"""
Diagnostic Figures
==================

Single home for every diagnostic / inspection figure produced by the
pipeline.  Each function is a pure side-effect that takes already-computed
results and writes one or more PNGs to disk.

This module does **not** compute any pipeline quantities.  It consumes:

- result dataclasses / dicts returned by other stages
- already-extracted traces, footprints, and movies

…and produces matplotlib figures.

Sections
--------
- Motion correction   :func:`generate_motion_figure`
- Auto-radius sweep   :func:`generate_radius_figure`
- ΔF/F₀ baseline      :func:`generate_local_background_diagnostic`
- Deconvolution       :func:`generate_deconvolution_figure`,
                      :func:`save_roi_trace_figures`,
                      :func:`generate_decay_diagnostics`
- Per-ROI inspection  :func:`generate_per_roi_pngs`

Each callable below is invoked once from ``run_full_pipeline.py`` at the
appropriate stage; nothing in this module is called recursively from
within other ``src/`` modules (with one exception: the local-background
helper is a private continuation of the ΔF/F₀ entry point).
"""

import os
import logging

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# MOTION CORRECTION
# =============================================================================

def generate_motion_figure(result, output_path: str) -> str:
    """Save motion correction diagnostic figure.

    Parameters
    ----------
    result : MotionCorrectionResult
        From :mod:`motion_correction`.
    output_path : str
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 1. Shifts over time
    ax = axes[0, 0]
    ax.plot(result.shifts[:, 0], label='Y shift', alpha=0.7)
    ax.plot(result.shifts[:, 1], label='X shift', alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Shift (px)')
    ax.set_title(f'Motion Shifts ({result.mode})')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Shift magnitude
    ax = axes[0, 1]
    magnitude = np.sqrt(result.shifts[:, 0]**2 + result.shifts[:, 1]**2)
    ax.plot(magnitude, color='steelblue', alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Magnitude (px)')
    ax.set_title(f'Shift Magnitude (max={magnitude.max():.1f} px)')
    ax.grid(alpha=0.3)

    # 3. Template
    ax = axes[1, 0]
    ax.imshow(result.template, cmap='gray')
    ax.set_title('Reference Template')
    ax.axis('off')

    # 4. Correlations
    ax = axes[1, 1]
    ax.plot(result.correlations[:min(len(result.correlations), 500)],
            color='steelblue', alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Correlation')
    ax.set_title('Frame-Template Correlation')
    ax.grid(alpha=0.3)

    fig.suptitle(f'Motion Correction — {result.mode}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


# =============================================================================
# AUTO-RADIUS SWEEP
# =============================================================================

def generate_radius_figure(radius_result: dict, output_path: str) -> str:
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


# =============================================================================
# ΔF/F₀ BASELINE CORRECTION (local-background method only)
# =============================================================================
# The local-background ΔF/F method depends on a per-ROI annulus falling on
# tissue but not on neighbouring ROIs.  This figure visualises the tissue
# mask and several example annuli so the geometry can be sanity-checked.

def generate_local_background_diagnostic(
    movie, A, C_raw, C_dff, dff_info, output_dir, frame_rate,
):
    """Tissue mask + annulus diagnostic figure for local background method."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation
    from scipy.sparse import issparse

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


# =============================================================================
# DECONVOLUTION
# =============================================================================
# All three deconvolution figures share the MAD-of-differences noise
# estimator from :mod:`deconvolution`.  Importing it here keeps the noise
# definition single-sourced.

from deconvolution import _mad_noise as _deconv_mad_noise  # noqa: E402


def save_roi_trace_figures(
    C: np.ndarray,
    deconv_result: dict,
    frame_rate: float,
    output_dir: str,
    n_rois: int = 20,
    dpi: int = 150,
) -> str:
    """
    Save one presentation-quality figure per ROI showing the raw ΔF/F₀
    trace (top panel) and the OASIS deconvolved trace with spike markers
    (bottom panel).

    ROIs are selected automatically: the top ``n_rois`` by SNR among those
    with at least one detected spike, so every saved figure is informative.

    Parameters
    ----------
    C : array (N, T)
        ΔF/F₀ traces passed into deconvolution (or raw fluorescence — the
        function uses ``deconv_result['C_dff']`` if available, falling back
        to C).
    deconv_result : dict
        Output of ``deconvolve_traces``.
    frame_rate : float
        Sampling rate in Hz — used to label the time axis.
    output_dir : str
        Parent output directory.  Figures are written to
        ``<output_dir>/figures/roi_traces/``.
    n_rois : int
        Number of ROIs to save (default 20).
    dpi : int
        Figure resolution (default 150).

    Returns
    -------
    str  Path to the roi_traces directory.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    S      = deconv_result['S']
    C_den  = deconv_result['C_denoised']
    n_spikes = deconv_result['n_spikes']
    C_raw  = deconv_result.get('C_dff', C)   # prefer the ΔF/F₀ that OASIS saw

    N, T = C_raw.shape
    t_ax = np.arange(T) / frame_rate

    # ── Select ROIs: top n_rois by SNR among those with spikes ──────────
    snr = np.zeros(N)
    for i in range(N):
        noise = _deconv_mad_noise(C_raw[i])
        if noise > 0:
            snr[i] = (np.percentile(C_raw[i], 95) - np.percentile(C_raw[i], 5)) / noise

    active_idx = np.where(n_spikes > 0)[0]
    if len(active_idx) == 0:
        logger.warning("  No ROIs with detected spikes — skipping ROI trace figures")
        return output_dir

    selected = active_idx[np.argsort(snr[active_idx])[::-1]][:n_rois]

    traces_dir = os.path.join(output_dir, 'figures', 'roi_traces')
    os.makedirs(traces_dir, exist_ok=True)

    for rank, roi_idx in enumerate(selected):
        raw   = C_raw[roi_idx]
        den   = C_den[roi_idx]
        spike_frames = np.where(S[roi_idx] > 0)[0]

        fig, (ax_raw, ax_den) = plt.subplots(
            2, 1, figsize=(14, 4),
            sharex=True,
            gridspec_kw={'hspace': 0.08},
        )

        # ── Top: raw ΔF/F₀ ───────────────────────────────────────────
        ax_raw.plot(t_ax, raw, color='#aaaaaa', linewidth=0.8, zorder=2)
        ax_raw.set_ylabel('ΔF/F₀', fontsize=10)
        ax_raw.set_xlim(t_ax[0], t_ax[-1])
        ax_raw.spines[['top', 'right']].set_visible(False)
        ax_raw.tick_params(axis='x', labelbottom=False)

        # ── Bottom: deconvolved trace + spike markers ────────────────
        ax_den.plot(t_ax, den, color='#2196F3', linewidth=1.2, zorder=2,
                    label='Deconvolved')
        if len(spike_frames) > 0:
            spike_times = spike_frames / frame_rate
            ax_den.vlines(spike_times,
                          ymin=0, ymax=den[spike_frames],
                          color='#F44336', linewidth=1.2,
                          zorder=3, label='Spikes')
            ax_den.scatter(spike_times, den[spike_frames],
                           color='#F44336', s=18, zorder=4)

        ax_den.axhline(0, color='#555555', linewidth=0.5, linestyle='--')
        ax_den.set_ylabel('Deconvolved', fontsize=10)
        ax_den.set_xlabel('Time (s)', fontsize=10)
        ax_den.set_xlim(t_ax[0], t_ax[-1])
        ax_den.spines[['top', 'right']].set_visible(False)

        fig.suptitle(
            f'ROI {roi_idx}   |   SNR {snr[roi_idx]:.1f}   |   '
            f'{n_spikes[roi_idx]} spikes   |   s_min = 0.1 ΔF/F₀',
            fontsize=11, fontweight='bold', y=1.01,
        )

        fname = os.path.join(traces_dir, f'roi_{roi_idx:04d}_rank{rank+1:03d}.png')
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    logger.info(f"  Saved {len(selected)} ROI trace figures → {traces_dir}/")
    return traces_dir


def generate_deconvolution_figure(
    C: np.ndarray,
    deconv_result: dict,
    frame_rate: float,
    output_path: str,
    n_examples: int = 8,
    C_filtered=None,
) -> str:
    """Generate diagnostic figure showing deconvolution results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    S = deconv_result['S']
    C_den = deconv_result['C_denoised']
    n_spikes = deconv_result['n_spikes']

    # Use the ΔF/F₀ version for display if available
    C_plot = deconv_result.get('C_dff', C)

    N, T = C_plot.shape
    t_ax = np.arange(T) / frame_rate

    # Pick examples: prioritise neurons WITH detected spikes, sorted by SNR.
    # Show a mix of active neurons (to verify detection quality) and
    # at most 2 inactive ones (to verify they're correctly silent).
    snr = np.zeros(N)
    for i in range(N):
        noise = _deconv_mad_noise(C_plot[i])
        if noise > 0:
            snr[i] = (np.percentile(C_plot[i], 95) - np.percentile(C_plot[i], 5)) / noise

    has_spikes = n_spikes > 0
    active_idx = np.where(has_spikes)[0]
    inactive_idx = np.where(~has_spikes)[0]

    # Sort each group by SNR descending
    active_sorted = active_idx[np.argsort(snr[active_idx])[::-1]] if len(active_idx) > 0 else np.array([], dtype=int)
    inactive_sorted = inactive_idx[np.argsort(snr[inactive_idx])[::-1]] if len(inactive_idx) > 0 else np.array([], dtype=int)

    # Take up to (n_examples - 2) active, then fill with up to 2 inactive
    n_active_show = min(len(active_sorted), max(n_examples - 2, n_examples))
    n_inactive_show = min(len(inactive_sorted), max(0, n_examples - n_active_show))
    examples = np.concatenate([active_sorted[:n_active_show], inactive_sorted[:n_inactive_show]]).astype(int)
    examples = examples[:n_examples]

    fig, axes = plt.subplots(n_examples, 1, figsize=(16, n_examples * 2.2),
                             sharex=True)
    if n_examples == 1:
        axes = [axes]

    for ax, idx in zip(axes, examples):
        # Raw ΔF/F₀ trace
        ax.plot(t_ax, C_plot[idx], color='#888', alpha=0.4, linewidth=0.5,
                label='ΔF/F₀')

        # Denoised trace
        ax.plot(t_ax, C_den[idx], color='#00e676', linewidth=1.2,
                label='Denoised')

        # Spike times
        spike_frames = np.where(S[idx] > 0)[0]
        if len(spike_frames) > 0:
            spike_times = spike_frames / frame_rate
            ax.scatter(spike_times, C_den[idx, spike_frames],
                       color='red', s=15, zorder=5, label='Spikes')

        ax.set_ylabel(f'ROI {idx}', fontsize=8)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)

        info = f'SNR={snr[idx]:.1f}, {n_spikes[idx]} spikes'
        ax.text(0.99, 0.95, info, transform=ax.transAxes, fontsize=7,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    axes[0].legend(fontsize=8, ncol=4, loc='upper left')
    axes[-1].set_xlabel('Time (s)')

    method = deconv_result['method']
    total = int(n_spikes.sum())
    median = float(np.median(n_spikes))
    fig.suptitle(f'Deconvolution ({method}) — {total} total spikes, '
                 f'median {median:.0f}/neuron',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def generate_decay_diagnostics(
    deconv_result: dict,
    C_dff: np.ndarray,
    frame_rate: float,
    decay_time_initial: float,
    output_dir: str,
):
    """
    Generate diagnostic figure for deconvolution decay parameters.

    Produces decay_diagnostics.png with 6 panels:
      A — Histogram of fitted g values vs initial g
      B — Histogram of implied tau values vs initial tau
      C — Per-ROI scatter of fitted tau vs trace SNR
      D–F — Example transient overlays (fastest / median / slowest decay)

    Parameters
    ----------
    deconv_result : dict
        Output from deconvolve_traces()
    C_dff : np.ndarray (N, T)
        Input ΔF/F₀ traces
    frame_rate, decay_time_initial : float
    output_dir : str
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    N, T = C_dff.shape
    dt = 1.0 / frame_rate
    g_init = np.exp(-dt / decay_time_initial)

    g_fitted = deconv_result['g'][:, 0]
    g_valid = g_fitted[g_fitted > 0]

    if len(g_valid) == 0:
        logger.warning("No valid g values — skipping decay diagnostics")
        return None

    tau_fitted = -dt / np.log(np.clip(g_valid, 1e-10, 1 - 1e-10))
    tau_all = -dt / np.log(np.clip(g_fitted, 1e-10, 1 - 1e-10))
    tau_cap = np.percentile(tau_fitted, 99)

    C_den = deconv_result['C_denoised']
    S = deconv_result['S']
    noise = deconv_result['noise']

    # SNR per ROI
    snr = np.zeros(N)
    for i in range(N):
        n = noise[i]
        if n > 1e-10:
            snr[i] = (np.percentile(C_dff[i], 95) - np.percentile(C_dff[i], 5)) / n

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Deconvolution decay parameter diagnostics', fontsize=20,
                 fontweight='bold')

    # ── A: g histogram ───────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.hist(g_valid, bins=50, color='#4472C4', alpha=0.7, edgecolor='white')
    ax.axvline(g_init, color='red', linestyle='--', linewidth=2,
               label=f'Initial g = {g_init:.4f}\n(τ = {decay_time_initial}s)')
    ax.axvline(np.median(g_valid), color='orange', linewidth=2,
               label=f'Median fitted = {np.median(g_valid):.4f}')
    ax.set_xlabel('Fitted g (AR1 coefficient)', fontsize=18)
    ax.set_ylabel('Count', fontsize=18)
    ax.set_title('A — Fitted decay coefficient', fontsize=24, fontweight='bold')
    ax.legend(fontsize=18, loc='upper right')
    ax.tick_params(labelsize=18)

    # ── B: tau histogram ─────────────────────────────────────────────────
    ax = axes[0, 1]
    tau_clip = np.clip(tau_fitted, 0, tau_cap)
    ax.hist(tau_clip, bins=50, color='#ED7D31', alpha=0.7, edgecolor='white')
    ax.axvline(decay_time_initial, color='red', linestyle='--', linewidth=2,
               label=f'Initial τ = {decay_time_initial}s')
    ax.axvline(np.median(tau_fitted), color='blue', linewidth=2,
               label=f'Median fitted = {np.median(tau_fitted):.2f}s')
    ax.set_xlabel('Implied decay time τ (seconds)', fontsize=18)
    ax.set_ylabel('Count', fontsize=18)
    ax.set_title('B — Implied decay time', fontsize=24, fontweight='bold')
    ax.legend(fontsize=18, loc='upper right')
    ax.tick_params(labelsize=18)

    # ── C: tau vs SNR ────────────────────────────────────────────────────
    ax = axes[0, 2]
    valid = g_fitted > 0
    tau_plot = np.clip(tau_all, 0, tau_cap)
    ax.scatter(snr[valid], tau_plot[valid], s=15, alpha=0.4, c='#4472C4')
    ax.axhline(decay_time_initial, color='red', linestyle='--', alpha=0.7,
               label=f'Initial τ = {decay_time_initial}s')
    ax.axhline(np.median(tau_fitted), color='orange', linestyle='-', alpha=0.7,
               label=f'Median fitted = {np.median(tau_fitted):.2f}s')
    ax.set_xlabel('Trace SNR (p95−p5 / noise)', fontsize=18)
    ax.set_ylabel('Fitted τ (s)', fontsize=18)
    ax.set_title('C — Decay time vs trace quality', fontsize=24, fontweight='bold')
    ax.legend(fontsize=18, loc='upper right')
    ax.tick_params(labelsize=18)

    # ── D–F: Example transients ──────────────────────────────────────────
    n_spikes_per = np.array([int(np.sum(S[i] > 0)) for i in range(N)])
    active = np.where((n_spikes_per >= 3) & (g_fitted > 0))[0]
    t_ax = np.arange(T) / frame_rate

    if len(active) >= 3:
        tau_active = tau_all[active]
        order = np.argsort(tau_active)
        examples = [active[order[0]],
                    active[order[len(order) // 2]],
                    active[order[-1]]]
        labels = ['D — Fastest decay', 'E — Median decay', 'F — Slowest decay']
    elif len(active) > 0:
        examples = list(active[:min(3, len(active))])
        labels = [f'{chr(68+j)} — Example' for j in range(len(examples))]
    else:
        examples, labels = [], []

    for j, (roi, label) in enumerate(zip(examples, labels)):
        ax = axes[1, j]
        ax.plot(t_ax, C_dff[roi], color='#aaa', alpha=0.4, linewidth=0.5,
                label='Raw ΔF/F₀')
        ax.plot(t_ax, C_den[roi], color='#00c853', linewidth=1.2,
                label='OASIS denoised')

        spk = np.where(S[roi] > 0)[0]
        if len(spk) > 0:
            ax.scatter(spk / frame_rate, C_den[roi, spk],
                       color='red', s=25, zorder=5, label='Events')

        g_r = g_fitted[roi]
        tau_r = tau_all[roi]
        ax.set_title(f'{label}',
                     fontsize=24, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=18)
        ax.set_ylabel('ΔF/F₀', fontsize=18)
        ax.legend(fontsize=18, loc='upper right')
        ax.tick_params(labelsize=18)
        ax.grid(alpha=0.15)

    for j in range(len(examples), 3):
        axes[1, j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'decay_diagnostics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved decay diagnostics: {path}")
    return path


# =============================================================================
# PER-ROI INSPECTION PNGS
# =============================================================================
# These helpers were previously private inside ``run_full_pipeline.py``.
# They are used only by :func:`generate_per_roi_pngs`.

def _get_roi_weights(A, roi_idx, dims):
    """Extract spatial weight map for an ROI."""
    if hasattr(A, 'toarray'):
        w = A[:, roi_idx].toarray().flatten()
    else:
        w = A[:, roi_idx].flatten()
    return w.reshape(dims)


def _get_roi_centroid(weights):
    """Return (cy, cx) weighted centroid of footprint."""
    ys, xs = np.where(weights > 0)
    if len(ys) == 0:
        return None
    cy = float(np.average(ys, weights=weights[ys, xs]))
    cx = float(np.average(xs, weights=weights[ys, xs]))
    return cy, cx


def _pad_crop(frame, cy, cx, half_size):
    """Crop a square window around (cy, cx), padding with frame minimum."""
    H, W = frame.shape
    size = 2 * half_size
    out = np.full((size, size), frame.min(), dtype=frame.dtype)
    src_y0 = int(round(cy)) - half_size
    src_x0 = int(round(cx)) - half_size
    dst_y0 = max(0, -src_y0)
    dst_x0 = max(0, -src_x0)
    sy0 = max(0, src_y0); sy1 = min(H, src_y0 + size)
    sx0 = max(0, src_x0); sx1 = min(W, src_x0 + size)
    h = sy1 - sy0; w = sx1 - sx0
    if h > 0 and w > 0:
        out[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] = frame[sy0:sy1, sx0:sx1]
    return out


def _pad_crop_mask(mask2d, cy, cx, half_size):
    return _pad_crop(mask2d.astype(np.float32), cy, cx, half_size)


def _find_peak_frame(trace, frame_rate):
    from scipy.signal import find_peaks
    F0 = np.percentile(trace, 20)
    scale = np.percentile(trace, 95) - F0 + 1e-10
    norm = (trace - F0) / scale
    peaks, _ = find_peaks(norm, height=0.3, distance=int(frame_rate), prominence=0.2)
    if len(peaks) == 0:
        return int(np.argmax(trace))
    return int(peaks[np.argmax(trace[peaks])])


def _find_baseline_frame(trace):
    F0 = np.percentile(trace, 20)
    thresh = F0 + 0.1 * (trace.max() - F0)
    quiet = np.where(trace <= thresh)[0]
    if len(quiet) == 0:
        return int(np.argmin(trace))
    mid = len(trace) // 2
    return int(quiet[np.argmin(np.abs(quiet - mid))])


def _compute_dff(trace):
    F0 = np.percentile(trace, 20)
    if F0 > 1e-6:
        return (trace - F0) / F0 * 100, 'ΔF/F (%)'
    return trace - trace.min(), 'F - F_min'


def generate_per_roi_pngs(
    A_final, C_final, movie, dims, projections, frame_rate,
    output_dir,
):
    """Generate per-ROI inspection PNGs (peak-frame sequence + trace per ROI)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as mpl_gridspec

    n_final = A_final.shape[1]
    inspection_dir = os.path.join(output_dir, 'inspection')
    os.makedirs(inspection_dir, exist_ok=True)

    # Sort by mean activity (descending) so the highest-signal ROIs come first
    activity = np.array([np.percentile(C_final[i], 95) - np.percentile(C_final[i], 5)
                         for i in range(n_final)])
    sort_idx = np.argsort(activity)[::-1]

    logger.info(f"  Generating {n_final} per-ROI inspection PNGs...")

    image_paths = []
    T_movie = len(movie)

    for plot_i, roi_idx in enumerate(sort_idx):
        try:
            weights = _get_roi_weights(A_final, roi_idx, dims)
            centroid = _get_roi_centroid(weights)
            if centroid is None:
                continue

            cy, cx = centroid
            w_thresh = max(weights.max() * 0.2, 1e-10)
            TIGHT = 60

            trace = C_final[roi_idx, :]
            dff, ylabel = _compute_dff(trace)
            peak_t = _find_peak_frame(trace, frame_rate)
            base_t = _find_baseline_frame(trace)

            T_trace = len(trace)
            peak_f = min(int(peak_t * T_movie / T_trace), T_movie - 1)

            title = f"ROI #{roi_idx}  (Rank {plot_i + 1}/{n_final})"

            fig = plt.figure(figsize=(20, 6), facecolor='#1a1a2e')
            gs_roi = mpl_gridspec.GridSpec(
                2, 11, height_ratios=[1, 1.5], hspace=0.4, wspace=0.08)
            fig.suptitle(title, fontsize=12, fontweight='bold', color='#ccc')

            offsets = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
            w_tight = _pad_crop_mask(weights, cy, cx, TIGHT)

            sequence_crops = [
                _pad_crop(movie[max(0, min(T_movie - 1, peak_f + offset))],
                          cy, cx, TIGHT)
                for offset in offsets
            ]

            vlo = np.percentile(sequence_crops, 1)
            vhi = np.percentile(sequence_crops, 99.5)
            if vhi <= vlo:
                vhi = vlo + 1

            for i, (offset, crop) in enumerate(zip(offsets, sequence_crops)):
                ax = fig.add_subplot(gs_roi[0, i])
                ax.set_facecolor('#1a1a2e')
                ax.imshow(crop, cmap='gray', vmin=vlo, vmax=vhi, interpolation='none')
                ax.contour(w_tight, levels=[w_thresh],
                           colors=['cyan'], linewidths=1.2, alpha=0.8)
                if offset == 0:
                    ax.set_title(f"Peak (t={peak_t / frame_rate:.1f}s)",
                                 fontsize=10, color='lime', fontweight='bold')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('lime'); spine.set_linewidth(2)
                else:
                    label = f"+{offset}" if offset > 0 else str(offset)
                    ax.set_title(f"Peak {label}", fontsize=9, color='#ccc')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#444')
                ax.axis('off')

            ax_trace = fig.add_subplot(gs_roi[1, :])
            ax_trace.set_facecolor('#111')
            t_ax = np.arange(len(dff)) / frame_rate
            ax_trace.plot(t_ax, dff, color='#4fc3f7', linewidth=1.0)
            ax_trace.axhline(0, color='#555', linewidth=0.5)
            ax_trace.axvline(base_t / frame_rate, color='cyan',
                             linestyle='--', linewidth=1.2, label='Baseline')
            ax_trace.axvline(peak_t / frame_rate, color='lime',
                             linestyle='--', linewidth=1.2, label='Peak')
            win_lo = max(0, (peak_f - 5)) / frame_rate
            win_hi = min(T_movie - 1, (peak_f + 5)) / frame_rate
            ax_trace.axvspan(win_lo, win_hi, alpha=0.2, color='lime',
                             label='Image Sequence Window')
            ax_trace.set_xlabel('Time (s)', color='#aaa', fontsize=9)
            ax_trace.set_ylabel(ylabel, color='#aaa', fontsize=9)
            ax_trace.set_title('Calcium Trace (Processed)',
                               fontsize=10, color='#ccc')
            ax_trace.legend(fontsize=8, framealpha=0.3,
                            labelcolor='white', facecolor='white',
                            loc='upper right')
            ax_trace.set_xlim(0, t_ax[-1])
            ax_trace.grid(True, alpha=0.2, color='#444')
            ax_trace.tick_params(colors='#aaa')
            for spine in ax_trace.spines.values():
                spine.set_edgecolor('#444')

            img_path = os.path.join(inspection_dir, f'roi_{roi_idx:04d}.png')
            plt.savefig(img_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
            image_paths.append(img_path)

        except Exception as roi_err:
            logger.warning(f"    ROI #{roi_idx} image failed: {roi_err}")
            plt.close('all')
            continue

        if (plot_i + 1) % 50 == 0:
            logger.info(f"    Generated {plot_i + 1}/{n_final} ROI images")

    logger.info(f"  Generated {len(image_paths)} per-ROI inspection PNGs")
    return image_paths
