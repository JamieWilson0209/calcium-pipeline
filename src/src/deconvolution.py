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
