"""
Calcium Trace Deconvolution
============================

Infers spike trains from noisy ΔF/F calcium traces using constrained
deconvolution.

Primary method: CaImAn's OASIS (Online Active Set method to Infer Spikes),
which is fast, parameter-light, and handles the AR(1)/AR(2) calcium
indicator models.

Fallback: a simple threshold-based peak detection if CaImAn is unavailable.

Also provides optional temporal filtering (low-pass Butterworth) as a
preprocessing step before deconvolution.

References:
- Friedrich et al., "Fast online deconvolution of calcium imaging data",
  PLoS Comp Bio 2017.
- Pachitariu et al., "Suite2p", bioRxiv 2017.
"""

import os
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)


# =============================================================================
# TEMPORAL FILTERING
# =============================================================================

def temporal_filter(
    C: np.ndarray,
    frame_rate: float,
    cutoff_hz: float = 2.0,
    order: int = 3,
) -> np.ndarray:
    """
    Low-pass Butterworth filter for calcium traces.

    Parameters
    ----------
    C : array (N, T)
        Trace matrix.
    frame_rate : float
        Sampling rate in Hz.
    cutoff_hz : float
        Cutoff frequency in Hz (default 2.0).  For Fluo-4 (τ≈400ms) at
        2 Hz, real transient content is below ~2 Hz.
    order : int
        Filter order (default 3).

    Returns
    -------
    C_filtered : array (N, T)
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = frame_rate / 2.0
    if cutoff_hz >= nyquist:
        logger.warning(f"Cutoff {cutoff_hz} Hz >= Nyquist {nyquist} Hz — skipping filter")
        return C.copy()

    sos = butter(order, cutoff_hz / nyquist, btype='low', output='sos')

    N, T = C.shape
    C_filt = np.zeros_like(C)

    for i in range(N):
        try:
            C_filt[i] = sosfiltfilt(sos, C[i])
        except Exception:
            C_filt[i] = C[i]

    logger.info(f"  Temporal filter: Butterworth LP, cutoff={cutoff_hz} Hz, "
                f"order={order}, frame_rate={frame_rate} Hz")

    return C_filt


# =============================================================================
# DECONVOLUTION — OASIS (CaImAn)
# =============================================================================

def deconvolve_traces(
    C: np.ndarray,
    frame_rate: float,
    decay_time: float = 0.4,
    method: str = 'oasis',
    penalty: float = 0,
    optimize_g: bool = True,
    noise_method: str = 'mean',
) -> Dict[str, np.ndarray]:
    """
    Deconvolve calcium traces to infer spike trains.

    Parameters
    ----------
    C : array (N, T)
        Trace matrix (raw fluorescence or ΔF/F₀).  If traces appear to be
        raw fluorescence (median > 1), they are automatically converted to
        ΔF/F₀ before deconvolution.
    frame_rate : float
        Sampling rate in Hz.
    decay_time : float
        Indicator decay time constant in seconds (Fluo-4: ~0.4s).
    method : str
        'oasis' (default, recommended) or 'threshold' (fallback).
    penalty : float
        Sparsity penalty (L1). 0 = auto-tune (recommended for OASIS).
    optimize_g : bool
        Whether OASIS should optimise the AR coefficient from data.
    noise_method : str
        How OASIS estimates noise: 'mean', 'median', or 'logmexp'.

    Returns
    -------
    dict with:
        'C_denoised'  : (N, T)  — denoised calcium traces (in ΔF/F₀)
        'S'           : (N, T)  — inferred spike trains (≥s_min and ≥3.5σ noise)
        'bl'          : (N,)    — estimated baselines
        'noise'       : (N,)    — estimated noise levels
        'g'           : (N, p)  — AR coefficients per neuron
        'method'      : str     — method used
        'n_spikes'    : (N,)    — number of inferred spikes per neuron
        'C_dff'       : (N, T)  — ΔF/F₀ input used for deconvolution
    """
    N, T = C.shape

    # ── Ensure traces are ΔF/F₀ ──────────────────────────────────────────
    # OASIS expects small-valued ΔF/F₀ traces (values near 0, transients
    # as positive bumps of ~0.05–0.5).  If traces are raw fluorescence
    # (values in hundreds/thousands/millions), convert them first.
    C_dff = _ensure_dff(C)

    logger.info(f"Deconvolution: method={method}, {N} traces, "
                f"decay={decay_time}s, frame_rate={frame_rate} Hz")
    logger.info(f"  Input range: [{C_dff.min():.4f}, {C_dff.max():.4f}], "
                f"median={np.median(C_dff):.4f}")

    if method == 'oasis':
        try:
            result = _deconvolve_oasis(
                C_dff, frame_rate, decay_time,
                penalty=penalty, optimize_g=optimize_g,
                noise_method=noise_method,
            )
        except Exception as e:
            logger.warning(f"OASIS failed: {e}")
            logger.info("Falling back to threshold deconvolution")
            result = _deconvolve_threshold(C_dff, frame_rate, decay_time)
    elif method == 'threshold':
        result = _deconvolve_threshold(C_dff, frame_rate, decay_time)
    else:
        raise ValueError(f"Unknown deconvolution method: {method}")

    result['C_dff'] = C_dff
    return result


def _ensure_dff(C: np.ndarray) -> np.ndarray:
    """
    Ensure traces are ΔF/F₀ for OASIS deconvolution.

    If data is raw fluorescence (median > 1.0): apply per-trace rolling
    percentile baseline correction and convert to ΔF/F₀.

    If data is already ΔF/F₀ (median ≤ 1.0): pass through without
    modification.  OASIS internally estimates its own baseline (the `bl`
    parameter in constrained_foopsi), so additional baseline subtraction
    here can interfere with its estimation and degrade spike detection.
    """
    N, T = C.shape
    median_val = np.median(C)

    if median_val <= 1.0:
        logger.info(f"  Traces appear to be ΔF/F₀ already (median={median_val:.4f})")
        logger.info(f"  Passing through without baseline adjustment — "
                    f"OASIS will estimate its own baseline internally")
        return C.copy().astype(np.float32)

    logger.info(f"  Traces appear to be raw fluorescence (median={median_val:.1f}), "
                f"converting to ΔF/F₀...")

    from scipy.ndimage import percentile_filter
    window = max(int(T * 0.25), 50)
    window = min(window, T)

    C_dff = np.zeros_like(C, dtype=np.float64)

    for i in range(N):
        trace = C[i].astype(np.float64)

        # Rolling percentile baseline
        baseline = percentile_filter(trace, percentile=8, size=window, mode='reflect')
        # Floor at 1% of the trace median (not 1e-6) to avoid
        # division-by-zero artefacts from zero-padded border pixels
        trace_median = np.median(trace[trace > 0]) if np.any(trace > 0) else 1.0
        baseline = np.maximum(baseline, max(trace_median * 0.01, 1e-2))

        C_dff[i] = np.clip((trace - baseline) / baseline, -1.0, 100.0)

    logger.info(f"  Converted: range [{C_dff.min():.4f}, {C_dff.max():.4f}], "
                f"median={np.median(C_dff):.4f}")

    return C_dff.astype(np.float32)


def _deconvolve_oasis(
    C, frame_rate, decay_time, penalty, optimize_g, noise_method,
) -> Dict[str, np.ndarray]:
    """Run CaImAn's OASIS deconvolution.

    s_min=0.1 is passed to constrained_foopsi, instructing OASIS to suppress
    any inferred spike whose amplitude is below 10% ΔF/F₀ relative to
    baseline.  This is the principled way to gate on transient amplitude —
    letting the solver itself discard sub-threshold events rather than
    re-filtering its output post-hoc.
    """
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi

    N, T = C.shape
    dt = 1.0 / frame_rate

    # Initial AR(1) coefficient from decay time:  g = exp(-dt/τ)
    g_init = np.exp(-dt / decay_time)
    logger.info(f"  OASIS: g_init={g_init:.4f} (τ={decay_time}s, dt={dt:.4f}s), "
                f"s_min=0.1 (10% ΔF/F₀), noise gate=3.5σ")

    C_denoised = np.zeros((N, T), dtype=np.float32)
    S = np.zeros((N, T), dtype=np.float32)
    baselines = np.zeros(N, dtype=np.float32)
    noise_levels = np.zeros(N, dtype=np.float32)
    g_values = np.zeros((N, 1), dtype=np.float32)
    n_spikes = np.zeros(N, dtype=np.int32)

    n_success = 0
    n_failed = 0

    for i in range(N):
        trace = C[i].astype(np.float64)

        try:
            # constrained_foopsi returns: (c, bl, c1, g, sn, sp, lam)
            # s_min=0.1 tells OASIS to zero out any spike whose reconstructed
            # amplitude is below 0.1 (i.e. 10% ΔF/F₀), matching our
            # definition of a real transient.
            c, bl, c1, g, sn, sp, lam = constrained_foopsi(
                trace,
                g=[g_init] if not optimize_g else None,
                noise_method=noise_method,
                p=1,       # AR(1) model
                s_min=0.1, # minimum spike amplitude: 10% ΔF/F₀
            )

            C_denoised[i] = c.astype(np.float32)
            S[i] = sp.astype(np.float32)
            baselines[i] = float(bl)
            noise_levels[i] = float(sn)
            g_values[i, 0] = float(g[0]) if len(g) > 0 else g_init
            n_success += 1

        except Exception as e:
            # OASIS failed for this trace — use simple peak detection.
            # Zero the denoised trace so downstream knows this wasn't
            # properly deconvolved.
            C_denoised[i] = 0.0
            S[i] = _simple_spike_detect(trace, frame_rate, decay_time)
            noise_levels[i] = _mad_noise(trace)
            g_values[i, 0] = g_init
            n_failed += 1
            if n_failed <= 3:
                logger.warning(f"    OASIS failed on trace {i}: {e}")

        if (i + 1) % max(1, N // 5) == 0:
            logger.info(f"    Deconvolved {i + 1}/{N} traces")

    logger.info(f"  OASIS complete: {n_success} success, {n_failed} fallback")

    # ── Decay parameter diagnostics ──────────────────────────────────────
    dt = 1.0 / frame_rate
    g_fitted = g_values[:, 0]
    g_valid = g_fitted[g_fitted > 0]
    if len(g_valid) > 0:
        tau_fitted = -dt / np.log(np.clip(g_valid, 1e-10, 1 - 1e-10))
        logger.info(f"  ── DECAY PARAMETER DIAGNOSTICS ──")
        logger.info(f"  Initial g (from τ={decay_time}s): {g_init:.4f}")
        logger.info(f"  Fitted g:  median={np.median(g_valid):.4f}, "
                    f"mean={np.mean(g_valid):.4f}, "
                    f"range=[{np.min(g_valid):.4f}, {np.max(g_valid):.4f}]")
        logger.info(f"  Implied τ: median={np.median(tau_fitted):.3f}s, "
                    f"mean={np.mean(tau_fitted):.3f}s, "
                    f"range=[{np.min(tau_fitted):.3f}s, {np.max(tau_fitted):.3f}s]")
        logger.info(f"  Ratio τ_fitted/τ_initial: "
                    f"median={np.median(tau_fitted)/decay_time:.1f}×, "
                    f"range=[{np.min(tau_fitted)/decay_time:.1f}×, "
                    f"{np.max(tau_fitted)/decay_time:.1f}×]")
        pcts = np.percentile(tau_fitted, [5, 25, 50, 75, 95])
        logger.info(f"  τ percentiles: p5={pcts[0]:.3f}s, p25={pcts[1]:.3f}s, "
                    f"p50={pcts[2]:.3f}s, p75={pcts[3]:.3f}s, p95={pcts[4]:.3f}s")

    # ── Noise-relative spike gate ─────────────────────────────────────────
    # Each spike that survived s_min=0.1 must also exceed 3.5× the trace
    # noise floor (OASIS's own sn estimate). This ensures only clearly
    # visible transients are counted — spikes between s_min and 3.5σ are
    # large enough in absolute terms but sit within the noise band of noisier
    # traces and produce ambiguous detections.
    noise_gate_sigma = 3.5
    n_noise_rejected = 0
    for i in range(N):
        sn = float(noise_levels[i])
        if sn <= 0:
            continue
        threshold = noise_gate_sigma * sn
        spike_frames = np.where(S[i] > 0)[0]
        for pk in spike_frames:
            if S[i, pk] < threshold:
                S[i, pk] = 0.0
                n_noise_rejected += 1

    if n_noise_rejected > 0:
        logger.info(f"  Noise gate (>{noise_gate_sigma}σ): "
                    f"removed {n_noise_rejected} sub-threshold spikes")

    # ── Transient duration gate ────────────────────────────────────────────
    # Reject traces where the longest continuous run of spike frames
    # exceeds max_transient_seconds.  Genuine calcium transients decay
    # within a few seconds (fitted τ ≈ 2s median); sustained events
    # lasting 80+ seconds indicate loading artefacts, slow baseline
    # shifts, or non-neuronal signals that OASIS has fitted as activity.
    max_transient_seconds = 80.0
    max_transient_frames = int(max_transient_seconds * frame_rate)
    n_duration_rejected = 0
    for i in range(N):
        if np.sum(S[i] > 0) == 0:
            continue
        spike_binary = (S[i] > 0).astype(np.int32)
        diffs = np.diff(np.concatenate([[0], spike_binary, [0]]))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        if len(starts) > 0:
            longest_run = int(np.max(ends - starts))
            if longest_run > max_transient_frames:
                C_denoised[i, :] = 0.0
                S[i, :] = 0.0
                n_duration_rejected += 1

    if n_duration_rejected > 0:
        logger.info(f"  Duration gate (>{max_transient_seconds:.0f}s): "
                    f"zeroed {n_duration_rejected} traces with sustained events")

    # ── Edge guard ────────────────────────────────────────────────────────
    # Suppress spikes at recording boundaries where baseline estimation is
    # unreliable regardless of the amplitude gate.
    edge_frames = max(2, int(frame_rate * 0.5))
    for i in range(N):
        S[i, :edge_frames] = 0.0
        S[i, -edge_frames:] = 0.0

    n_spikes = np.array([int(np.sum(S[i] > 0)) for i in range(N)], dtype=np.int32)

    total_spikes  = int(n_spikes.sum())
    median_spikes = float(np.median(n_spikes))
    logger.info(f"  Final: {total_spikes} spikes, median {median_spikes:.0f}/neuron")

    return {
        'C_denoised': C_denoised,
        'S':          S,
        'bl':         baselines,
        'noise':      noise_levels,
        'g':          g_values,
        'method':     'oasis',
        'n_spikes':   n_spikes,
    }


def _deconvolve_threshold(C, frame_rate, decay_time) -> Dict[str, np.ndarray]:
    """Simple threshold-based spike detection fallback.

    Uses a 3.5σ MAD noise threshold to match the noise gate applied to the
    OASIS path, so the two methods are consistent when OASIS is unavailable.
    """
    from scipy.signal import find_peaks

    N, T = C.shape
    logger.info(f"  Threshold deconvolution: {N} traces")

    S = np.zeros((N, T), dtype=np.float32)
    baselines = np.zeros(N, dtype=np.float32)
    noise_levels = np.zeros(N, dtype=np.float32)
    n_spikes = np.zeros(N, dtype=np.int32)

    min_distance = max(1, int(decay_time * frame_rate * 0.5))

    for i in range(N):
        trace = C[i]
        noise = _mad_noise(trace)
        baseline = np.percentile(trace, 20)

        noise_levels[i] = noise
        baselines[i] = baseline

        if noise > 0:
            # 3.5σ threshold matches the noise gate on the OASIS path
            height = baseline + 3.5 * noise
            peaks, _ = find_peaks(
                trace, height=height, distance=min_distance,
            )
            for pk in peaks:
                S[i, pk] = trace[pk] - baseline
            n_spikes[i] = len(peaks)

    logger.info(f"  Threshold deconvolution: median {np.median(n_spikes):.0f} "
                f"spikes/neuron")

    return {
        'C_denoised': C.copy(),
        'S':          S,
        'bl':         baselines,
        'noise':      noise_levels,
        'g':          np.full((N, 1), np.exp(-1.0 / (frame_rate * decay_time))),
        'method':     'threshold',
        'n_spikes':   n_spikes,
    }


def _simple_spike_detect(trace, frame_rate, decay_time):
    """Quick spike detection for a single trace (OASIS per-trace fallback).

    Uses 3.5σ threshold to match the noise gate on the main OASIS path.
    """
    from scipy.signal import find_peaks
    S = np.zeros_like(trace)
    noise = _mad_noise(trace)
    baseline = np.percentile(trace, 20)
    if noise > 0:
        height = baseline + 3.5 * noise
        peaks, _ = find_peaks(trace, height=height,
                              distance=max(1, int(decay_time * frame_rate * 0.5)))
        for pk in peaks:
            S[pk] = trace[pk] - baseline
    return S


def _mad_noise(trace):
    """MAD-based noise estimate."""
    diff = np.diff(trace)
    return 1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2)


# =============================================================================
# DIAGNOSTIC FIGURE
# =============================================================================

def save_roi_trace_figures(
    C: np.ndarray,
    deconv_result: Dict,
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
        noise = _mad_noise(C_raw[i])
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
    deconv_result: Dict,
    frame_rate: float,
    output_path: str,
    n_examples: int = 8,
    C_filtered: Optional[np.ndarray] = None,
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
        noise = _mad_noise(C_plot[i])
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
    deconv_result: Dict,
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
    import os

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
