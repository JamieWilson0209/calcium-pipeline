"""
Auto Radius Optimisation
========================

Automatically selects the detection radius parameters that maximise the
number of high-SNR calcium traces in a dataset.

Unlike simple blob-counting approaches, this module optimises for what
actually matters: clean traces with clear activity spikes.

Algorithm
---------
1. Compute projections from the movie (fast)
2. For each candidate radius in the sweep range:
   a. Run blob detection + contour fitting
   b. Build spatial footprints
   c. Extract traces (fast weighted-average, no neuropil)
   d. Score each trace's SNR (peak amplitude / MAD noise)
   e. Count traces exceeding SNR threshold
3. Select the radius that yields the most high-SNR detections
4. Generate diagnostic figure comparing all candidates

This runs ~5 candidate radii in parallel-friendly fashion.  Total time
is roughly 5× the detection stage (typically 30–120 s per dataset).
"""

import numpy as np
import logging
import time
from typing import Dict, Tuple
from scipy.sparse import issparse

logger = logging.getLogger(__name__)

# Default SNR threshold: traces must have peak-to-noise ratio above this
# to count as "high quality".  5.0 is a common minimum in the literature
# for reliable transient detection (CaImAn, Suite2p use similar).
DEFAULT_SNR_THRESHOLD = 5.0


def estimate_trace_snr(traces: np.ndarray) -> np.ndarray:
    """
    Compute per-trace SNR as robust peak amplitude / MAD noise.

    Parameters
    ----------
    traces : array (N, T)

    Returns
    -------
    snr : array (N,)
        Zero for traces with negligible noise floor.
    """
    diff = np.diff(traces, axis=1)                          # (N, T-1)
    medians = np.median(diff, axis=1, keepdims=True)        # (N, 1)
    mad = np.median(np.abs(diff - medians), axis=1)         # (N,)
    noise = 1.4826 * mad / np.sqrt(2)                       # (N,)

    peak = (np.percentile(traces, 95, axis=1)
            - np.percentile(traces, 5, axis=1))             # (N,)

    snr = np.zeros(traces.shape[0])
    valid = noise > 1e-10
    snr[valid] = peak[valid] / noise[valid]
    return snr


def _run_candidate(
    movie: np.ndarray,
    min_radius: float,
    max_radius: float,
    smooth_sigma: float,
    snr_threshold: float,
    max_seeds: int = 500,
    precomputed_projections=None,
) -> Dict:
    """Run detection + quick trace extraction at one radius setting."""
    try:
        from .contour_seed_detection import (
            detect_seeds_with_contours,
            contours_to_spatial_footprints,
        )
    except ImportError:
        from contour_seed_detection import (
            detect_seeds_with_contours,
            contours_to_spatial_footprints,
        )

    T, d1, d2 = movie.shape
    dims = (d1, d2)

    t0 = time.time()

    # Detection
    try:
        seeds = detect_seeds_with_contours(
            movie,
            min_radius=min_radius,
            max_radius=max_radius,
            intensity_threshold=0.18,
            correlation_threshold=0.12,
            border_margin=10,
            max_seeds=max_seeds,
            contour_method='otsu',
            smooth_sigma=smooth_sigma,
            use_temporal_projection=True,
            n_peak_frames=10,
            peak_percentile=90,
            precomputed_projections=precomputed_projections,
        )
    except Exception as e:
        logger.warning(f"    Detection failed at r=[{min_radius},{max_radius}]: {e}")
        return _empty_result(min_radius, max_radius)

    n_seeds = seeds.n_seeds
    if n_seeds == 0:
        return _empty_result(min_radius, max_radius)

    # Build footprints
    A = contours_to_spatial_footprints(seeds, dims, contour_fallback=True)

    # Quick trace extraction (weighted average, no neuropil, no chunking)
    A_dense = A.toarray().astype(np.float32) if issparse(A) else A.astype(np.float32)
    weights = A_dense.sum(axis=0)
    weights[weights == 0] = 1e-10

    # Extract on a subset of frames for speed (every Nth frame)
    stride = max(1, T // 500)
    Y_sub = movie[::stride].reshape(-1, d1 * d2).T  # (pixels, T_sub)
    C_sub = (A_dense.T @ Y_sub) / weights[:, np.newaxis]  # (N, T_sub)

    # SNR
    snr = estimate_trace_snr(C_sub)
    n_good = int(np.sum(snr >= snr_threshold))
    mean_snr = float(np.mean(snr)) if len(snr) > 0 else 0.0
    median_snr = float(np.median(snr)) if len(snr) > 0 else 0.0

    # Also compute top-quartile SNR (quality of the best detections)
    if len(snr) >= 4:
        top_q_snr = float(np.mean(np.sort(snr)[-max(1, len(snr) // 4):]))
    else:
        top_q_snr = mean_snr

    elapsed = time.time() - t0

    return {
        'min_radius': min_radius,
        'max_radius': max_radius,
        'n_seeds': n_seeds,
        'n_good': n_good,
        'mean_snr': mean_snr,
        'median_snr': median_snr,
        'top_quartile_snr': top_q_snr,
        'snr_values': snr,
        'radii': seeds.radii,
        'elapsed': elapsed,
    }


def _empty_result(min_r, max_r):
    return {
        'min_radius': min_r,
        'max_radius': max_r,
        'n_seeds': 0,
        'n_good': 0,
        'mean_snr': 0.0,
        'median_snr': 0.0,
        'top_quartile_snr': 0.0,
        'snr_values': np.array([]),
        'radii': np.array([]),
        'elapsed': 0.0,
    }


def optimise_radius(
    movie: np.ndarray,
    smooth_sigma: float = 4.0,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    n_candidates: int = 5,
    radius_range: Tuple[float, float] = (3.0, 35.0),
    max_seeds: int = 500,
    precomputed_projections=None,
) -> Dict:
    """
    Sweep radius values and select the one producing the most high-SNR traces.

    Parameters
    ----------
    movie : array (T, d1, d2)
    smooth_sigma : float
        Gaussian smoothing sigma for hotspot suppression.
    snr_threshold : float
        Minimum trace SNR to count as "good" (default 5.0).
    n_candidates : int
        Number of radius settings to test (default 5).
    radius_range : (float, float)
        Min and max of the radius sweep range.
    max_seeds : int
        Max seeds per candidate (keeps it fast).
    precomputed_projections : ProjectionSet, optional
        If provided, skip the internal projection computation and reuse these
        across all candidates.  Useful when the caller will also use the same
        projections downstream.

    Returns
    -------
    dict with:
        'best_min_radius', 'best_max_radius' : selected values
        'best_n_good' : number of high-SNR detections at best radius
        'candidates' : list of per-candidate results
        'reliable' : bool
    """
    r_min, r_max = radius_range

    # Generate candidate radius pairs
    # Each candidate has a min_radius and max_radius = min_radius * 2.5
    # (detections are typically 1–2.5× the minimum radius)
    centers = np.linspace(r_min * 1.5, r_max * 0.7, n_candidates)
    candidates = []
    for c in centers:
        mn = max(r_min, c * 0.5)
        mx = min(r_max, c * 2.0)
        if mx <= mn:
            mx = mn * 2.0
        candidates.append((round(mn, 1), round(mx, 1)))

    # Deduplicate
    seen = set()
    unique = []
    for pair in candidates:
        key = (pair[0], pair[1])
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    candidates = unique

    logger.info(f"  Auto-radius: testing {len(candidates)} candidates "
                f"(SNR threshold={snr_threshold})")

    # Compute projections once — they're invariant across radius candidates.
    # Reuse the caller's projections if provided.
    if precomputed_projections is not None:
        logger.info("  Reusing precomputed projections")
        shared_projections = precomputed_projections
    else:
        try:
            from .contour_seed_detection import compute_projections_extended
        except ImportError:
            from contour_seed_detection import compute_projections_extended
        logger.info("  Computing projections once for all candidates...")
        proj_t0 = time.time()
        shared_projections = compute_projections_extended(
            movie, compute_correlation=True, smooth_sigma=smooth_sigma,
        )
        logger.info(f"  Projections ready ({time.time() - proj_t0:.1f}s)")

    results = []
    for mn, mx in candidates:
        logger.info(f"    r=[{mn:.1f}, {mx:.1f}] ...")
        res = _run_candidate(
            movie, mn, mx, smooth_sigma, snr_threshold, max_seeds,
            precomputed_projections=shared_projections,
        )
        results.append(res)
        logger.info(f"      → {res['n_seeds']} seeds, {res['n_good']} good "
                     f"(SNR≥{snr_threshold}), mean SNR={res['mean_snr']:.1f}, "
                     f"{res['elapsed']:.1f}s")

    # Select best: maximise n_good, break ties with top-quartile SNR
    best_idx = 0
    best_score = (-1, -1.0)
    for i, res in enumerate(results):
        score = (res['n_good'], res['top_quartile_snr'])
        if score > best_score:
            best_score = score
            best_idx = i

    best = results[best_idx]
    reliable = best['n_good'] >= 3

    logger.info(f"  Auto-radius: BEST r=[{best['min_radius']:.1f}, "
                f"{best['max_radius']:.1f}] → {best['n_good']} good detections "
                f"(mean SNR={best['mean_snr']:.1f})")

    return {
        'best_min_radius': best['min_radius'],
        'best_max_radius': best['max_radius'],
        'best_n_good': best['n_good'],
        'best_mean_snr': best['mean_snr'],
        'best_top_quartile_snr': best['top_quartile_snr'],
        'snr_threshold': snr_threshold,
        'candidates': [
            {k: v for k, v in r.items()
             if k not in ('snr_values', 'radii')}
            for r in results
        ],
        'reliable': reliable,
        'all_results': results,  # kept for figure generation
    }


def generate_radius_figure(
    radius_result: Dict,
    output_path: str,
) -> str:
    """Save diagnostic figure showing the radius optimisation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results = radius_result['all_results']
    snr_thresh = radius_result['snr_threshold']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel subheadings

    panel_labels = ['A', 'B', 'C']
    for ax, label in zip(axes, panel_labels):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=24, fontweight='bold', va='top', ha='right')

    # 1. N good detections vs radius
    ax = axes[0]
    radii_labels = [f"[{r['min_radius']:.0f},{r['max_radius']:.0f}]" for r in results]
    n_good = [r['n_good'] for r in results]
    n_total = [r['n_seeds'] for r in results]

    x = np.arange(len(results))
    bars = ax.bar(x, n_good, color='steelblue', edgecolor='black', alpha=0.7,
                  label=f'SNR ≥ {snr_thresh}')
    ax.bar(x, [t - g for t, g in zip(n_total, n_good)], bottom=n_good,
           color='lightgray', edgecolor='black', alpha=0.5, label='Below threshold')

    best_idx = max(range(len(results)), key=lambda i: (results[i]['n_good'], results[i]['top_quartile_snr']))
    bars[best_idx].set_color('#e94560')

    ax.set_xticks(x)
    ax.set_xticklabels(radii_labels, rotation=30, fontsize=16)
    ax.set_xlabel('Radius range [min, max] (px)')
    ax.set_ylabel('Number of detections', fontsize=16)
    ax.set_title('High-SNR Detections by Radius', fontsize=18)
    ax.legend(fontsize=16)

    # 2. SNR distributions per candidate
    ax = axes[1]
    for i, res in enumerate(results):
        snr = res['snr_values']
        if len(snr) > 0:
            color = '#e94560' if i == best_idx else 'steelblue'
            alpha = 0.9 if i == best_idx else 0.4
            sorted_snr = np.sort(snr)
            cdf = np.arange(1, len(sorted_snr) + 1) / len(sorted_snr)
            ax.plot(sorted_snr, cdf, color=color, alpha=alpha,
                    linewidth=2 if i == best_idx else 1,
                    label=radii_labels[i])

    ax.axvline(snr_thresh, color='red', linestyle='--', alpha=0.5,
               label=f'Threshold={snr_thresh}')
    ax.set_xlabel('Trace SNR', fontsize=16)
    ax.set_ylabel('Cumulative fraction', fontsize=16)
    ax.set_title('SNR Distributions', fontsize=18)
    ax.legend(fontsize=16, loc='lower right')
    ax.set_xlim(0, min(50, ax.get_xlim()[1]))

    # 3. Mean SNR and top-quartile SNR
    ax = axes[2]
    mean_snrs = [r['mean_snr'] for r in results]
    top_snrs = [r['top_quartile_snr'] for r in results]

    ax.bar(x - 0.15, mean_snrs, width=0.3, color='steelblue', edgecolor='black',
           alpha=0.7, label='Mean SNR')
    ax.bar(x + 0.15, top_snrs, width=0.3, color='darkorange', edgecolor='black',
           alpha=0.7, label='Top quartile SNR')

    ax.set_xticks(x)
    ax.set_xticklabels(radii_labels, rotation=30, fontsize=16)
    ax.set_xlabel('Radius range [min, max] (px)')
    ax.set_ylabel('SNR', fontsize=16)
    ax.set_title('SNR Quality', fontsize=18)
    ax.legend(fontsize=16)

    best_r = results[best_idx]
    fig.suptitle(
        f"Auto-Radius Optimisation — Best: [{best_r['min_radius']:.0f}, "
        f"{best_r['max_radius']:.0f}] px, "
        f"{best_r['n_good']} detections with SNR ≥ {snr_thresh}",
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path
