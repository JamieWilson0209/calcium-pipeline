"""
Auto Radius Optimisation
========================

Automatically selects the detection radius parameters that maximise the
median Otsu inter-class variance across all detected blobs.

Algorithm
---------
1. Compute projections once (shared read-only across all candidates)
2. Run N candidate radius settings in parallel, each:
   a. Blob detection + contour fitting
   b. Collect per-blob Otsu inter-class variance from diagnostics
   c. Score as median variance across successful contours only
3. Select the candidate with the highest median inter-class variance

Scoring rationale
-----------------
Otsu thresholding maximises the inter-class variance between foreground
(cell body) and background pixels:

    σ²_between = w_bg × w_fg × (μ_bg − μ_fg)²

A radius setting that matches the true cell size produces clean bimodal
ROI histograms with high inter-class variance.  A mismatched radius
(too small: ROI clips the cell; too large: background dilutes the
foreground) blurs the histogram and reduces separation.  This metric is
purely geometric — it is independent of neural activity levels and
unaffected by the noise-averaging artefact that biases SNR-based scoring
toward larger ROIs.

Parallelisation
---------------
Candidates are independent — they share only read-only inputs (movie,
projections) — so they map cleanly onto a joblib worker pool.  The same
code runs on Eddie (fork, multi-core SGE job) and locally (spawn on Mac,
fork on Linux) without modification.  Pool size is capped at the number
of available cores so the job respects its SGE allocation.
"""

import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# SINGLE CANDIDATE EVALUATION
# =============================================================================

def _evaluate_candidate(
    movie: np.ndarray,
    min_radius: float,
    max_radius: float,
    smooth_sigma: float,
    max_seeds: int,
    precomputed_projections,
) -> Dict:
    """
    Run detection for one radius candidate and return its median Otsu
    inter-class variance across all blobs with successful contours.

    Designed to be called inside a joblib worker — all inputs are read-only
    and the return value is a plain dict of scalars.
    """
    try:
        from contour_seed_detection import detect_seeds_with_contours
    except ImportError:
        from .contour_seed_detection import detect_seeds_with_contours

    t0 = time.time()

    try:
        seeds = detect_seeds_with_contours(
            movie,
            min_radius=min_radius,
            max_radius=max_radius,
            intensity_threshold=0.18,
            correlation_threshold=0.12,
            border_margin=10,
            max_seeds=max_seeds,
            smooth_sigma=smooth_sigma,
            n_peak_frames=10,
            peak_percentile=90,
            precomputed_projections=precomputed_projections,
        )
    except Exception as exc:
        logger.warning(f"    Detection failed at r=[{min_radius}, {max_radius}]: {exc}")
        return _empty_result(min_radius, max_radius)

    if seeds.n_seeds == 0:
        return _empty_result(min_radius, max_radius)

    # Collect inter-class variance from each blob's diagnostics.
    # Only blobs where Otsu succeeded are included — failures are excluded
    # entirely rather than penalised with a zero.
    variances = [
        d['otsu_inter_class_variance']
        for d in seeds.diagnostics.get('per_blob', [])
        if d.get('success') and 'otsu_inter_class_variance' in d
    ]

    median_variance = float(np.median(variances)) if variances else 0.0

    return {
        'min_radius':      min_radius,
        'max_radius':      max_radius,
        'n_seeds':         seeds.n_seeds,
        'n_contours':      seeds.n_contours,
        'median_variance': median_variance,
        'variances':       variances,
        'elapsed':         time.time() - t0,
    }


def _empty_result(min_r: float, max_r: float) -> Dict:
    return {
        'min_radius':      min_r,
        'max_radius':      max_r,
        'n_seeds':         0,
        'n_contours':      0,
        'median_variance': 0.0,
        'variances':       [],
        'elapsed':         0.0,
    }


# =============================================================================
# CANDIDATE GENERATION
# =============================================================================

def _generate_candidates(
    radius_range: Tuple[float, float],
    n_candidates: int,
) -> List[Tuple[float, float]]:
    """
    Generate evenly spaced (min_radius, max_radius) candidate pairs across
    the sweep range.  Each candidate uses max_radius = 2× min_radius.
    Duplicates are removed.
    """
    r_min, r_max = radius_range
    centers = np.linspace(r_min * 1.5, r_max * 0.7, n_candidates)

    seen = set()
    candidates = []
    for c in centers:
        mn = round(max(r_min, c * 0.5), 1)
        mx = round(min(r_max, c * 2.0), 1)
        if mx <= mn:
            mx = round(mn * 2.0, 1)
        if (mn, mx) not in seen:
            seen.add((mn, mx))
            candidates.append((mn, mx))

    return candidates


# =============================================================================
# MAIN OPTIMISATION
# =============================================================================

def optimise_radius(
    movie: np.ndarray,
    smooth_sigma: float = 4.0,
    n_candidates: int = 5,
    radius_range: Tuple[float, float] = (3.0, 35.0),
    max_seeds: int = 500,
    precomputed_projections=None,
) -> Dict:
    """
    Sweep radius candidates in parallel and select the one with the highest
    median Otsu inter-class variance across detected blobs.

    Parameters
    ----------
    movie : ndarray (T, d1, d2)
    smooth_sigma : float
        Gaussian smoothing sigma — should match the value used in the main
        pipeline so projections are comparable.
    n_candidates : int
        Number of radius settings to test.
    radius_range : (float, float)
        Min and max bounds of the radius sweep.
    max_seeds : int
        Detection cap per candidate — controls runtime.
    precomputed_projections : ProjectionSet, optional
        Reuse projections already computed upstream.  Passed read-only to
        all worker processes.

    Returns
    -------
    dict with keys:
        'best_min_radius', 'best_max_radius'
        'best_median_variance'
        'reliable' : bool  (True if best candidate found >= 3 contours)
        'candidates' : list of per-candidate summary dicts
        'all_results' : full result list (includes variances, for figures)
    """
    candidates = _generate_candidates(radius_range, n_candidates)
    n_workers  = min(len(candidates), os.cpu_count() or 1)

    logger.info(f"  Auto-radius: {len(candidates)} candidates, "
                f"{n_workers} parallel workers")

    # Shared projections
    if precomputed_projections is not None:
        logger.info("  Reusing precomputed projections")
        shared_projections = precomputed_projections
    else:
        try:
            from contour_seed_detection import compute_projections
        except ImportError:
            from .contour_seed_detection import compute_projections
        logger.info("  Computing projections for radius sweep...")
        t0 = time.time()
        shared_projections = compute_projections(
            movie, smooth_sigma=smooth_sigma, compute_correlation=True,
        )
        logger.info(f"  Projections ready ({time.time() - t0:.1f}s)")

    # Parallel candidate evaluation
    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_workers, prefer='threads')(
            delayed(_evaluate_candidate)(
                movie, mn, mx, smooth_sigma, max_seeds, shared_projections,
            )
            for mn, mx in candidates
        )
    except ImportError:
        logger.warning("  joblib not found — running candidates sequentially")
        results = [
            _evaluate_candidate(
                movie, mn, mx, smooth_sigma, max_seeds, shared_projections,
            )
            for mn, mx in candidates
        ]

    for res in results:
        logger.info(
            f"    r=[{res['min_radius']:.1f}, {res['max_radius']:.1f}]  "
            f"seeds={res['n_seeds']}  contours={res['n_contours']}  "
            f"median_variance={res['median_variance']:.1f}  "
            f"({res['elapsed']:.1f}s)"
        )

    # Select best candidate by median inter-class variance
    best = max(results, key=lambda r: r['median_variance'])

    logger.info(
        f"  Auto-radius: BEST r=[{best['min_radius']:.1f}, "
        f"{best['max_radius']:.1f}]  "
        f"median_variance={best['median_variance']:.1f}  "
        f"n_contours={best['n_contours']}"
    )

    return {
        'best_min_radius':      best['min_radius'],
        'best_max_radius':      best['max_radius'],
        'best_median_variance': best['median_variance'],
        'reliable':             best['n_contours'] >= 3,
        'candidates': [
            {k: v for k, v in r.items() if k != 'variances'}
            for r in results
        ],
        'all_results': results,
    }


# =============================================================================
# DIAGNOSTIC FIGURE
# =============================================================================

def generate_radius_figure(
    radius_result: Dict,
    output_path: str,
) -> str:
    """
    Save a 2-panel diagnostic figure for the radius optimisation sweep.

    Panel A — median Otsu inter-class variance per candidate (scoring metric)
    Panel B — variance distributions per candidate (CDF)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results  = radius_result['all_results']
    best_idx = next(
        i for i, r in enumerate(results)
        if r['min_radius'] == radius_result['best_min_radius']
        and r['max_radius'] == radius_result['best_max_radius']
    )

    x      = np.arange(len(results))
    labels = [f"[{r['min_radius']:.0f},{r['max_radius']:.0f}]" for r in results]

    BEST_COLOUR = '#e94560'
    BASE_COLOUR = 'steelblue'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, panel in zip(axes, ['A', 'B']):
        ax.text(-0.1, 1.05, panel, transform=ax.transAxes,
                fontsize=24, fontweight='bold', va='top', ha='right')

    # Panel A: Median inter-class variance per candidate
    ax = axes[0]
    colours = [BEST_COLOUR if i == best_idx else BASE_COLOUR
               for i in range(len(results))]
    ax.bar(x, [r['median_variance'] for r in results],
           color=colours, edgecolor='black', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, fontsize=11)
    ax.set_xlabel('Radius range [min, max] (px)')
    ax.set_ylabel('Median Otsu inter-class variance', fontsize=12)
    ax.set_title('Otsu Separation by Radius', fontsize=14)

    # Panel B: Variance CDFs
    ax = axes[1]
    for i, res in enumerate(results):
        variances = res['variances']
        if not variances:
            continue
        sorted_v = np.sort(variances)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.plot(sorted_v, cdf,
                color=BEST_COLOUR if i == best_idx else BASE_COLOUR,
                alpha=0.9 if i == best_idx else 0.4,
                linewidth=2 if i == best_idx else 1,
                label=labels[i])
    ax.set_xlabel('Otsu inter-class variance', fontsize=12)
    ax.set_ylabel('Cumulative fraction', fontsize=12)
    ax.set_title('Variance Distributions', fontsize=14)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10, loc='lower right')

    best = results[best_idx]
    fig.suptitle(
        f"Auto-Radius — Best: [{best['min_radius']:.0f}, {best['max_radius']:.0f}] px  "
        f"median variance={best['median_variance']:.1f}  "
        f"n_contours={best['n_contours']}",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path
