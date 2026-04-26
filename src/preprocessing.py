"""
Preprocessing — Trace-Level Baseline Correction
================================================

Converts raw fluorescence traces to ΔF/F₀ using a rolling low-percentile
baseline.  Two methods are provided:

- **global_dff** (default): per-trace rolling 8th-percentile baseline.
  Standard approach used by Suite2p and CaImAn.

- **local_background**: tissue-masked annulus around each ROI provides a
  local F₀(t).  Designed for organoid data where bright tissue is
  surrounded by dark medium.

These operate on extracted traces (N × T), not on the movie itself.
Movie-level preprocessing (motion correction) is handled separately.
"""

import numpy as np
import logging
from typing import Tuple
from scipy import ndimage

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_window(T: int, fraction: float = 0.25,
                     min_window: int = 50, max_window: int = 500) -> int:
    """Compute an odd rolling-window size from recording length."""
    window = int(T * fraction)
    window = max(min_window, min(max_window, window))
    return window if window % 2 == 1 else window + 1


def _detect_edge_artefacts(
    traces: np.ndarray, T: int,
) -> Tuple[int, int]:
    """
    Detect boundary artefacts at the start/end of traces.

    Returns (trim_start, trim_end) — the number of frames to exclude
    from baseline estimation at each end.  Zero if no artefact detected.
    """
    edge_check = max(5, T // 10)
    trace_medians = np.median(traces, axis=1)
    safe_medians = np.maximum(trace_medians, 1.0)

    start_ratios = np.median(traces[:, :edge_check], axis=1) / safe_medians
    end_ratios = np.median(traces[:, -edge_check:], axis=1) / safe_medians

    trim_start = edge_check if np.mean(start_ratios < 0.90) > 0.15 else 0
    trim_end = edge_check if np.mean(end_ratios < 0.90) > 0.15 else 0

    return trim_start, trim_end


def _rolling_baseline(
    trace: np.ndarray,
    percentile: float,
    window: int,
    trim_start: int = 0,
    trim_end: int = 0,
) -> np.ndarray:
    """
    Compute a rolling-percentile baseline for a single 1-D trace.

    If trim_start/trim_end are nonzero, the baseline is estimated on the
    interior and extrapolated to the edges by repeating the boundary value.
    The baseline is floored at 1% of the trace median (minimum 1.0) to
    prevent division-by-near-zero for dim ROIs.

    Parameters
    ----------
    trace : 1-D array (float64)
    percentile : float
    window : int (odd)
    trim_start, trim_end : int
        Frames to exclude from each end.

    Returns
    -------
    baseline : 1-D array (float64), same length as trace
    """
    T = len(trace)

    if trim_start > 0 or trim_end > 0:
        t_s = trim_start
        t_e = T - trim_end if trim_end > 0 else T
        interior = trace[t_s:t_e]
        bl_interior = ndimage.percentile_filter(
            interior, percentile,
            size=min(window, len(interior)), mode='reflect',
        )
        baseline = np.empty(T, dtype=np.float64)
        baseline[t_s:t_e] = bl_interior
        if trim_start > 0:
            baseline[:t_s] = bl_interior[0]
        if trim_end > 0:
            baseline[t_e:] = bl_interior[-1]
    else:
        baseline = ndimage.percentile_filter(
            trace, percentile, size=window, mode='reflect',
        )

    # Floor at 1% of trace median to avoid division by near-zero
    trace_median = np.median(trace[trace > 0]) if np.any(trace > 0) else 1.0
    floor = max(trace_median * 0.01, 1.0)
    np.maximum(baseline, floor, out=baseline)

    return baseline


def _compute_dff_stats(C_dff, C_raw, baseline_drift_pcts, n_clipped, T):
    """Compute summary statistics shared by both correction methods."""
    median_drift = float(np.median(baseline_drift_pcts))
    median_dff = float(np.median(C_dff))

    chunk = max(1, T // 10)
    residual_drifts = (np.mean(C_dff[:, -chunk:], axis=1)
                       - np.mean(C_dff[:, :chunk], axis=1))

    logger.info(f"  Median baseline drift: {median_drift:.1f}%")
    logger.info(f"  ΔF/F₀ range: [{C_dff.min():.4f}, {C_dff.max():.4f}], "
                f"median={median_dff:.4f}")
    if n_clipped > 0:
        logger.info(f"  Clamped {n_clipped} extreme values (|ΔF/F₀| > 50)")
    logger.info(f"  Residual drift: median={np.median(residual_drifts):.4f}, "
                f"p10={np.percentile(residual_drifts, 10):.4f}, "
                f"p90={np.percentile(residual_drifts, 90):.4f}")

    return {
        'median_baseline_drift_pct': median_drift,
        'median_dff': median_dff,
        'n_clipped': n_clipped,
        'residual_drift_median': float(np.median(residual_drifts)),
        'residual_drift_p10': float(np.percentile(residual_drifts, 10)),
        'residual_drift_p90': float(np.percentile(residual_drifts, 90)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API: PER-TRACE ΔF/F₀
# ─────────────────────────────────────────────────────────────────────────────

def compute_dff_traces(
    C_raw: np.ndarray,
    frame_rate: float = 2.0,
    percentile: float = 8.0,
    window_fraction: float = 0.25,
    min_window: int = 50,
    max_window: int = 500,
    edge_trim: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert raw fluorescence traces to ΔF/F₀ using per-trace rolling baseline.

    Standard approach (Suite2p, CaImAn): a rolling low-percentile filter
    estimates F₀(t) for each trace, then ΔF/F₀ = (F − F₀) / F₀.

    Parameters
    ----------
    C_raw : array (N, T)
        Raw fluorescence traces.
    frame_rate : float
        Acquisition rate in Hz (logging only).
    percentile : float
        Baseline percentile (default 8.0).
    window_fraction : float
        Rolling window as fraction of T.
    min_window, max_window : int
        Window size bounds in frames.
    edge_trim : bool
        Detect and exclude boundary artefacts from baseline estimation.

    Returns
    -------
    C_dff : array (N, T)
        ΔF/F₀ traces.
    C_raw : array (N, T)
        Pass-through of input traces.
    info : dict
        Diagnostics.
    """
    N, T = C_raw.shape
    window = _adaptive_window(T, window_fraction, min_window, max_window)

    logger.info(f"Computing per-trace ΔF/F₀: {N} traces, {T} frames, "
                f"percentile={percentile}, window={window}")

    # Edge trim
    trim_start, trim_end = (0, 0)
    if edge_trim:
        trim_start, trim_end = _detect_edge_artefacts(C_raw, T)
        if trim_start or trim_end:
            logger.info(f"  Edge trim: {trim_start} start, {trim_end} end frames")
        else:
            logger.info("  Edge trim enabled but no artefacts detected")

    C_dff = np.zeros_like(C_raw, dtype=np.float32)
    drift_pcts = np.zeros(N)
    n_clipped = 0

    for i in range(N):
        trace = C_raw[i].astype(np.float64)
        baseline = _rolling_baseline(trace, percentile, window,
                                     trim_start, trim_end)

        dff = (trace - baseline) / baseline

        if baseline[0] > 0:
            drift_pcts[i] = 100.0 * (baseline[0] - baseline[-1]) / baseline[0]

        n_extreme = int(np.sum(np.abs(dff) > 50.0))
        n_clipped += n_extreme
        C_dff[i] = np.clip(dff, -1.0, 100.0).astype(np.float32)

    stats = _compute_dff_stats(C_dff, C_raw, drift_pcts, n_clipped, T)
    info = {
        'method': 'per_trace_rolling_percentile',
        'percentile': percentile,
        'window_frames': window,
        **stats,
    }
    return C_dff, C_raw, info


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API: LOCAL TISSUE-MASKED BACKGROUND
# ─────────────────────────────────────────────────────────────────────────────

def compute_dff_local_background(
    movie: np.ndarray,
    A: np.ndarray,
    frame_rate: float = 2.0,
    percentile: float = 8.0,
    window_fraction: float = 0.25,
    min_window: int = 50,
    max_window: int = 500,
    annulus_inner_gap: int = 2,
    annulus_outer_radius: int = 20,
    tissue_threshold_method: str = 'otsu',
    min_annulus_pixels: int = 50,
    edge_trim: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute ΔF/F₀ using a local, tissue-masked background for each ROI.

    For organoid data where bright tissue is surrounded by dark medium.
    A local annulus around each ROI (masked to tissue pixels only)
    provides the baseline normalisation denominator, while the ROI's
    own rolling baseline provides the numerator:

        ΔF/F₀ = (F_roi − F0_roi) / F0_local

    This keeps the baseline near zero (F0_roi tracks resting level)
    while normalising amplitudes by local tissue brightness.

    Parameters
    ----------
    movie : array (T, d1, d2)
    A : sparse or dense (n_pixels, N)
        Spatial footprints.
    frame_rate : float
    percentile : float
    window_fraction, min_window, max_window : float, int, int
    annulus_inner_gap : int
        Pixels between ROI boundary and annulus start.
    annulus_outer_radius : int
        Outer radius of annulus beyond ROI.
    tissue_threshold_method : str
        'otsu' or 'percentile'.
    min_annulus_pixels : int
        Expand annulus if fewer tissue pixels available.
    edge_trim : bool

    Returns
    -------
    C_dff : array (N, T)
    C_raw : array (N, T)
    info : dict
    """
    from scipy.ndimage import binary_dilation
    from scipy.sparse import issparse

    T, d1, d2 = movie.shape
    dims = (d1, d2)

    A_dense = (A.toarray().astype(np.float32) if issparse(A)
               else np.asarray(A, dtype=np.float32))
    N = A_dense.shape[1]
    window = _adaptive_window(T, window_fraction, min_window, max_window)

    logger.info(f"Computing local-background ΔF/F₀: {N} ROIs, {T} frames, "
                f"annulus={annulus_inner_gap}+{annulus_outer_radius}px, "
                f"percentile={percentile}, window={window}")

    # ── Tissue mask ──────────────────────────────────────────────────────
    mean_proj = np.mean(movie, axis=0)
    if tissue_threshold_method == 'otsu':
        try:
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(mean_proj[mean_proj > 0])
        except Exception:
            thresh = np.percentile(mean_proj, 25)
    else:
        thresh = np.percentile(mean_proj, 25)

    tissue_mask = mean_proj > thresh
    tissue_frac = tissue_mask.sum() / tissue_mask.size
    logger.info(f"  Tissue mask: {tissue_frac:.1%} of FOV (threshold={thresh:.1f})")

    # ── Per-ROI annulus masks ────────────────────────────────────────────
    annulus_masks = []
    n_small = 0

    for i in range(N):
        roi_mask = (A_dense[:, i] > 0).reshape(dims)
        inner_zone = binary_dilation(roi_mask, iterations=annulus_inner_gap)
        outer_zone = binary_dilation(roi_mask,
                                     iterations=annulus_inner_gap + annulus_outer_radius)
        annulus = outer_zone & ~inner_zone & tissue_mask
        annulus_pixels = np.flatnonzero(annulus)

        if len(annulus_pixels) < min_annulus_pixels:
            expanded = binary_dilation(
                roi_mask,
                iterations=annulus_outer_radius * 2 + annulus_inner_gap,
            )
            annulus_pixels = np.flatnonzero(expanded & ~inner_zone & tissue_mask)

        if len(annulus_pixels) < 10:
            annulus_pixels = np.flatnonzero(tissue_mask)
            n_small += 1

        annulus_masks.append(annulus_pixels)

    if n_small > 0:
        logger.info(f"  {n_small}/{N} ROIs fell back to global tissue background")

    median_annulus = np.median([len(m) for m in annulus_masks])
    logger.info(f"  Annulus size: median {median_annulus:.0f} pixels")

    # ── Extract ROI + local background traces ────────────────────────────
    weights = A_dense.sum(axis=0)
    weights[weights == 0] = 1e-10

    C_raw = np.zeros((N, T), dtype=np.float32)
    F_local = np.zeros((N, T), dtype=np.float32)

    chunk_size = 500
    n_chunks = (T + chunk_size - 1) // chunk_size

    for chunk_i in range(n_chunks):
        t0 = chunk_i * chunk_size
        t1 = min(t0 + chunk_size, T)
        Y = movie[t0:t1].reshape(t1 - t0, -1).T

        C_raw[:, t0:t1] = (A_dense.T @ Y) / weights[:, np.newaxis]

        for i in range(N):
            px = annulus_masks[i]
            if len(px) > 0:
                F_local[i, t0:t1] = np.percentile(Y[px, :], percentile, axis=0)

    logger.info(f"  Extracted {N} ROI traces and local backgrounds")

    # ── Edge trim ────────────────────────────────────────────────────────
    trim_start, trim_end = (0, 0)
    if edge_trim:
        trim_start, trim_end = _detect_edge_artefacts(C_raw, T)
        if trim_start or trim_end:
            logger.info(f"  Edge trim: {trim_start} start, {trim_end} end frames")

    # ── Smooth local background baselines ────────────────────────────────
    F0_local = np.zeros_like(F_local)
    for i in range(N):
        F0_local[i] = _rolling_baseline(F_local[i].astype(np.float64),
                                        percentile, window,
                                        trim_start, trim_end)

    # ── Compute ΔF/F₀ = (F_roi − F0_roi) / F0_local ────────────────────
    C_dff = np.zeros((N, T), dtype=np.float32)
    drift_pcts = np.zeros(N)
    n_clipped = 0

    for i in range(N):
        roi_trace = C_raw[i].astype(np.float64)
        f0_roi = _rolling_baseline(roi_trace, percentile, window,
                                   trim_start, trim_end)

        dff = (roi_trace - f0_roi) / F0_local[i]

        if F0_local[i, 0] > 0:
            drift_pcts[i] = 100.0 * (F0_local[i, 0] - F0_local[i, -1]) / F0_local[i, 0]

        n_extreme = int(np.sum(np.abs(dff) > 50.0))
        n_clipped += n_extreme
        C_dff[i] = np.clip(dff, -1.0, 100.0).astype(np.float32)

    stats = _compute_dff_stats(C_dff, C_raw, drift_pcts, n_clipped, T)
    info = {
        'method': 'local_background_tissue_masked',
        'percentile': percentile,
        'window_frames': window,
        'annulus_inner_gap': annulus_inner_gap,
        'annulus_outer_radius': annulus_outer_radius,
        'tissue_fraction': float(tissue_frac),
        'tissue_threshold': float(thresh),
        'median_annulus_pixels': float(median_annulus),
        'n_fallback_global': n_small,
        'edge_trim_start': trim_start,
        'edge_trim_end': trim_end,
        **stats,
    }
    return C_dff, C_raw, info


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def generate_dff_diagnostics(
    C_raw_fluorescence: np.ndarray,
    C_dff: np.ndarray,
    dff_info: dict,
    output_dir: str,
    method: str = 'global_dff',
    frame_rate: float = 2.0,
    movie: np.ndarray = None,
    A=None,
    n_example_traces: int = 8,
):
    """
    Generate diagnostic figures for ΔF/F₀ baseline correction.

    Produces:
    1. dff_overview.png — Summary statistics panel
    2. dff_example_traces.png — Example traces showing raw + ΔF/F₀
    3. dff_local_background.png — (local_background only) Tissue mask + annulus
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)
    N, T = C_dff.shape
    t_axis = np.arange(T) / frame_rate

    METHOD_LABELS = {
        'global_dff': 'Global Per-Trace ΔF/F₀',
        'local_dff': 'Local Per-Event ΔF/F₀',
        'local_background': 'Local Tissue-Masked Background',
    }
    method_label = METHOD_LABELS.get(method, method)

    # ── Figure 1: Overview statistics ────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'ΔF/F₀ Baseline Correction — {method_label}',
                 fontsize=14, fontweight='bold')

    # 1a: Raw trace medians
    ax = axes[0, 0]
    raw_medians = np.median(C_raw_fluorescence, axis=1)
    ax.hist(raw_medians, bins=50, color='#4472C4', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Median raw fluorescence')
    ax.set_ylabel('Count')
    ax.set_title('Raw Trace Medians')
    ax.axvline(np.median(raw_medians), color='red', linestyle='--',
               label=f'median={np.median(raw_medians):.0f}')
    ax.legend(fontsize=8)

    # 1b: ΔF/F₀ medians
    ax = axes[0, 1]
    dff_medians = np.median(C_dff, axis=1)
    ax.hist(dff_medians, bins=50, color='#ED7D31', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Median ΔF/F₀')
    ax.set_ylabel('Count')
    ax.set_title('ΔF/F₀ Trace Medians')
    ax.axvline(np.median(dff_medians), color='red', linestyle='--',
               label=f'median={np.median(dff_medians):.4f}')
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax.legend(fontsize=8)

    # 1c: Residual drift
    ax = axes[0, 2]
    chunk = max(1, T // 10)
    drifts = np.mean(C_dff[:, -chunk:], axis=1) - np.mean(C_dff[:, :chunk], axis=1)
    ax.hist(drifts, bins=50, color='#70AD47', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Residual drift (last 10% − first 10%)')
    ax.set_ylabel('Count')
    ax.set_title('Residual Drift After Correction')
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(np.median(drifts), color='red', linestyle='--',
               label=f'median={np.median(drifts):.4f}')
    ax.legend(fontsize=8)

    # 1d: Amplitude distribution
    ax = axes[1, 0]
    amplitudes = np.percentile(C_dff, 95, axis=1) - np.percentile(C_dff, 5, axis=1)
    ax.hist(amplitudes, bins=50, color='#A855F7', alpha=0.7, edgecolor='white')
    ax.set_xlabel('ΔF/F₀ amplitude (p95 − p5)')
    ax.set_ylabel('Count')
    ax.set_title('Trace Amplitude Distribution')
    ax.axvline(np.median(amplitudes), color='red', linestyle='--',
               label=f'median={np.median(amplitudes):.4f}')
    ax.legend(fontsize=8)

    # 1e: Raw vs corrected drift
    ax = axes[1, 1]
    raw_drifts = (np.mean(C_raw_fluorescence[:, -chunk:], axis=1)
                  - np.mean(C_raw_fluorescence[:, :chunk], axis=1))
    raw_drifts_pct = raw_drifts / np.maximum(np.median(C_raw_fluorescence, axis=1), 1.0) * 100
    ax.scatter(raw_drifts_pct, drifts, s=8, alpha=0.4, c='#4472C4')
    ax.set_xlabel('Raw drift (% of median)')
    ax.set_ylabel('Corrected drift (ΔF/F₀)')
    ax.set_title('Drift Correction Effectiveness')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)

    # 1f: Info text
    ax = axes[1, 2]
    ax.axis('off')
    info_lines = [
        f"Method: {method_label}",
        f"N traces: {N}",
        f"T frames: {T}  ({T/frame_rate:.1f}s)",
        f"",
        f"Median baseline drift: {dff_info.get('median_baseline_drift_pct', 'N/A')}%",
        f"Median ΔF/F₀: {dff_info.get('median_dff', 'N/A'):.4f}",
        f"Window: {dff_info.get('window_frames', 'N/A')} frames",
        f"Clipped: {dff_info.get('n_clipped', 0)}",
        f"",
        f"Residual drift:",
        f"  median: {dff_info.get('residual_drift_median', 'N/A')}",
        f"  p10: {dff_info.get('residual_drift_p10', 'N/A')}",
        f"  p90: {dff_info.get('residual_drift_p90', 'N/A')}",
    ]
    if method == 'local_background':
        info_lines += [
            f"",
            f"Tissue fraction: {dff_info.get('tissue_fraction', 'N/A'):.1%}",
            f"Median annulus: {dff_info.get('median_annulus_pixels', 'N/A'):.0f} px",
            f"Global fallback: {dff_info.get('n_fallback_global', 0)} ROIs",
        ]
    ax.text(0.05, 0.95, '\n'.join(info_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, 'dff_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {path}")

    # ── Figure 2: Example traces ─────────────────────────────────────────
    amp_order = np.argsort(amplitudes)
    n_ex = min(n_example_traces, N)
    indices = amp_order[np.linspace(0, N - 1, n_ex, dtype=int)]

    fig, axes = plt.subplots(n_ex, 1, figsize=(16, 2.5 * n_ex), sharex=True)
    if n_ex == 1:
        axes = [axes]
    fig.suptitle(f'Example Traces — {method_label}',
                 fontsize=13, fontweight='bold')

    for row, idx in enumerate(indices):
        ax = axes[row]
        raw = C_raw_fluorescence[idx]
        dff = C_dff[idx]

        color_raw = '#4472C4'
        color_dff = '#ED7D31'
        ax.plot(t_axis, raw, color=color_raw, alpha=0.6, linewidth=0.5)
        ax.set_ylabel(f'ROI {idx}\nRaw F', fontsize=8, color=color_raw)
        ax.tick_params(axis='y', labelcolor=color_raw, labelsize=7)

        ax2 = ax.twinx()
        ax2.plot(t_axis, dff, color=color_dff, alpha=0.8, linewidth=0.7)
        ax2.set_ylabel('ΔF/F₀', fontsize=8, color=color_dff)
        ax2.tick_params(axis='y', labelcolor=color_dff, labelsize=7)
        ax2.axhline(0, color='gray', linestyle=':', alpha=0.4)

        amp_val = amplitudes[idx]
        drift_val = drifts[idx]
        ax.set_title(f'ROI {idx}  |  amplitude={amp_val:.4f}  |  drift={drift_val:.4f}',
                     fontsize=8, loc='left')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    path = os.path.join(output_dir, 'dff_example_traces.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {path}")

    # ── Figure 3: Local background diagnostics ───────────────────────────
    if method == 'local_background' and movie is not None and A is not None:
        _generate_local_background_diagnostic(
            movie, A, C_raw_fluorescence, C_dff, dff_info,
            output_dir, frame_rate)


def _generate_local_background_diagnostic(
    movie, A, C_raw, C_dff, dff_info, output_dir, frame_rate,
):
    """Tissue mask + annulus diagnostic figure for local background method."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation
    from scipy.sparse import issparse
    import os

    T, d1, d2 = movie.shape
    dims = (d1, d2)

    A_dense = (A.toarray().astype(np.float32) if issparse(A)
               else np.asarray(A, dtype=np.float32))
    N = A_dense.shape[1]

    mean_proj = np.mean(movie[:min(100, T)], axis=0)
    try:
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(mean_proj[mean_proj > 0])
    except Exception:
        thresh = np.percentile(mean_proj, 25)
    tissue_mask = mean_proj > thresh

    # Select 6 example ROIs spread across the FOV
    centers = []
    for i in range(N):
        fp = A_dense[:, i].reshape(dims)
        ys, xs = np.where(fp > 0)
        if len(ys) > 0:
            centers.append((i, np.mean(ys), np.mean(xs)))
    centers.sort(key=lambda c: c[1] * 1000 + c[2])
    n_examples = min(6, len(centers))
    example_idx = [centers[i][0] for i in
                   np.linspace(0, len(centers) - 1, n_examples, dtype=int)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Local Background Diagnostics — Tissue Mask & Annulus',
                 fontsize=13, fontweight='bold')

    # Full FOV with tissue mask
    ax = axes[0, 0]
    ax.imshow(mean_proj, cmap='gray',
              vmin=np.percentile(mean_proj, 1),
              vmax=np.percentile(mean_proj, 99))
    mask_overlay = np.zeros((*dims, 4))
    mask_overlay[tissue_mask] = [0, 1, 0, 0.15]
    mask_overlay[~tissue_mask] = [1, 0, 0, 0.1]
    ax.imshow(mask_overlay)
    ax.set_title(f'Tissue Mask ({tissue_mask.sum()/tissue_mask.size:.0%} tissue)',
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Zoom panels for example ROIs
    for panel_idx, roi_idx in enumerate(example_idx[:5]):
        row = (panel_idx + 1) // 3
        col = (panel_idx + 1) % 3
        ax = axes[row, col]

        fp = A_dense[:, roi_idx].reshape(dims)
        roi_mask = fp > 0
        ys, xs = np.where(roi_mask)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))

        margin = 40
        y0, y1 = max(0, cy - margin), min(d1, cy + margin)
        x0, x1 = max(0, cx - margin), min(d2, cx + margin)

        inner = binary_dilation(roi_mask, iterations=2)
        outer = binary_dilation(roi_mask, iterations=22)
        annulus = outer & ~inner & tissue_mask

        crop = mean_proj[y0:y1, x0:x1]
        ax.imshow(crop, cmap='gray',
                  vmin=np.percentile(mean_proj, 1),
                  vmax=np.percentile(mean_proj, 99))

        overlay = np.zeros((y1 - y0, x1 - x0, 4))
        overlay[roi_mask[y0:y1, x0:x1]] = [0, 0.5, 1, 0.4]
        overlay[annulus[y0:y1, x0:x1]] = [0, 1, 0, 0.25]
        overlay[~tissue_mask[y0:y1, x0:x1]] = [1, 0, 0, 0.1]
        ax.imshow(overlay)

        n_ann = int(annulus.sum())
        amp = float(np.percentile(C_dff[roi_idx], 95)
                    - np.percentile(C_dff[roi_idx], 5))
        ax.set_title(f'ROI {roi_idx}  |  annulus={n_ann}px  |  amp={amp:.3f}',
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(output_dir, 'dff_local_background.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {path}")
