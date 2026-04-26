"""
Network Spectral Analysis — Detection-Free
===========================================

Computes population-level spectral properties from the raw calcium
imaging movie without requiring ROI detection.  Handles the common
organoid imaging problem of variable empty-space fractions by using
variance-weighted pixel aggregation.

Key outputs:
    - Variance-weighted global trace (detection-free population signal)
    - Welch power spectral density of the global trace
    - Burst detection on the global trace (threshold crossings)
    - Per-recording spectral summary metrics

Author: Calcium Pipeline Project
"""

import logging
import os
import json

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


# ─── Core computation ────────────────────────────────────────────────────

def compute_variance_weights(movie, percentile_floor=5.0):
    """
    Compute per-pixel temporal standard deviation weights.

    Pixels in empty/black space have near-zero temporal variance and
    receive near-zero weight.  Active tissue pixels receive weight
    proportional to their temporal standard deviation.

    Parameters
    ----------
    movie : np.ndarray, shape (T, Y, X)
        Raw fluorescence movie.
    percentile_floor : float
        Pixels below this percentile of std are zeroed to suppress
        sensor noise in truly empty regions.

    Returns
    -------
    weights : np.ndarray, shape (Y, X)
        Normalised weights (sum to 1).  Zero for empty-space pixels.
    std_map : np.ndarray, shape (Y, X)
        Raw temporal standard deviation per pixel.
    """
    T, Y, X = movie.shape
    # Compute std in chunks to limit memory for large movies
    chunk = min(T, 500)
    n_chunks = (T + chunk - 1) // chunk

    sum_sq = np.zeros((Y, X), dtype=np.float64)
    sum_val = np.zeros((Y, X), dtype=np.float64)

    for i in range(n_chunks):
        s = i * chunk
        e = min(s + chunk, T)
        block = movie[s:e].astype(np.float64)
        sum_val += block.sum(axis=0)
        sum_sq += (block ** 2).sum(axis=0)

    mean_map = sum_val / T
    var_map = sum_sq / T - mean_map ** 2
    var_map = np.maximum(var_map, 0)
    std_map = np.sqrt(var_map)

    # Floor: suppress pixels below percentile threshold
    floor = np.percentile(std_map, percentile_floor)
    weights = std_map.copy()
    weights[weights < floor] = 0.0

    # Normalise
    total = weights.sum()
    if total > 0:
        weights /= total

    logger.info(f"  Variance weights: {np.count_nonzero(weights)} / {Y * X} pixels active "
                f"({100 * np.count_nonzero(weights) / (Y * X):.1f}%)")

    return weights, std_map


def compute_global_trace(movie, weights):
    """
    Compute variance-weighted global fluorescence trace.

    Parameters
    ----------
    movie : np.ndarray, shape (T, Y, X)
        Raw fluorescence movie.
    weights : np.ndarray, shape (Y, X)
        Per-pixel weights (from compute_variance_weights).

    Returns
    -------
    global_trace : np.ndarray, shape (T,)
        Weighted mean fluorescence per frame.
    """
    T = movie.shape[0]
    global_trace = np.zeros(T, dtype=np.float64)
    weights_flat = weights.ravel()

    for t in range(T):
        global_trace[t] = np.dot(movie[t].ravel().astype(np.float64), weights_flat)

    # Convert to ΔF/F₀ using rolling 20th percentile baseline
    win = max(11, T // 10)
    if win % 2 == 0:
        win += 1

    # Compute rolling percentile (manual for portability)
    baseline = np.zeros(T, dtype=np.float64)
    half = win // 2
    for t in range(T):
        s = max(0, t - half)
        e = min(T, t + half + 1)
        baseline[t] = np.percentile(global_trace[s:e], 20)

    # Avoid division by zero
    baseline[baseline < 1e-6] = 1e-6
    global_dff = (global_trace - baseline) / baseline

    logger.info(f"  Global trace: range [{global_dff.min():.4f}, {global_dff.max():.4f}], "
                f"std={global_dff.std():.4f}")

    return global_dff


def compute_welch_psd(trace, frame_rate, nperseg=None):
    """
    Compute Welch power spectral density of a 1D trace.

    Parameters
    ----------
    trace : np.ndarray, shape (T,)
    frame_rate : float
        Sampling rate in Hz.
    nperseg : int or None
        Segment length for Welch.  None = T // 4 (good default for
        short recordings).

    Returns
    -------
    freqs : np.ndarray
        Frequency axis in Hz.
    psd : np.ndarray
        Power spectral density.
    """
    T = len(trace)
    if nperseg is None:
        nperseg = min(T // 2, max(32, T // 4))

    freqs, psd = scipy_signal.welch(
        trace,
        fs=frame_rate,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend='linear',
        scaling='density',
    )

    return freqs, psd


def detect_global_bursts(global_trace, frame_rate, threshold_sd=2.0,
                         min_gap_seconds=3.0):
    """
    Detect population burst events on the global trace.

    A burst is defined as a contiguous region where the global trace
    exceeds threshold_sd standard deviations above the median.
    Bursts closer than min_gap_seconds are merged.

    Parameters
    ----------
    global_trace : np.ndarray, shape (T,)
    frame_rate : float
    threshold_sd : float
        Threshold in units of MAD-estimated standard deviation.
    min_gap_seconds : float
        Minimum inter-burst interval; closer bursts are merged.

    Returns
    -------
    bursts : list of dict
        Each burst has 'onset_frame', 'offset_frame', 'onset_s',
        'offset_s', 'duration_s', 'peak_frame', 'peak_amplitude'.
    """
    # MAD-based robust std
    median_val = np.median(global_trace)
    mad = np.median(np.abs(global_trace - median_val))
    robust_std = 1.4826 * mad
    threshold = median_val + threshold_sd * robust_std

    above = global_trace > threshold
    min_gap_frames = int(min_gap_seconds * frame_rate)

    # Find contiguous regions
    bursts = []
    in_burst = False
    onset = 0

    for t in range(len(global_trace)):
        if above[t] and not in_burst:
            onset = t
            in_burst = True
        elif not above[t] and in_burst:
            bursts.append({'onset_frame': onset, 'offset_frame': t - 1})
            in_burst = False

    if in_burst:
        bursts.append({'onset_frame': onset, 'offset_frame': len(global_trace) - 1})

    # Merge close bursts
    if len(bursts) > 1:
        merged = [bursts[0]]
        for b in bursts[1:]:
            if b['onset_frame'] - merged[-1]['offset_frame'] < min_gap_frames:
                merged[-1]['offset_frame'] = b['offset_frame']
            else:
                merged.append(b)
        bursts = merged

    # Enrich with timing and amplitude
    for b in bursts:
        b['onset_s'] = b['onset_frame'] / frame_rate
        b['offset_s'] = b['offset_frame'] / frame_rate
        b['duration_s'] = b['offset_s'] - b['onset_s']
        segment = global_trace[b['onset_frame']:b['offset_frame'] + 1]
        b['peak_frame'] = b['onset_frame'] + int(np.argmax(segment))
        b['peak_amplitude'] = float(np.max(segment))

    logger.info(f"  Detected {len(bursts)} population bursts "
                f"(threshold={threshold:.4f}, {threshold_sd}\u00D7SD)")

    return bursts


def compute_spectral_summary(freqs, psd, bursts, recording_duration_s):
    """
    Compute summary metrics from spectral and burst analysis.

    Returns
    -------
    dict with:
        dominant_freq_hz, dominant_period_s, spectral_entropy,
        total_power, band_power_fractions, burst_rate_per_min,
        mean_burst_duration_s, mean_ibi_s, cv_ibi
    """
    # Dominant frequency (excluding DC)
    mask = freqs > 0.005  # skip very low freq / DC
    if mask.sum() > 0:
        peak_idx = np.argmax(psd[mask])
        dom_freq = freqs[mask][peak_idx]
        dom_period = 1.0 / dom_freq if dom_freq > 0 else np.inf
    else:
        dom_freq = 0.0
        dom_period = np.inf

    # Spectral entropy (normalised, 0 = pure sinusoid, 1 = white noise)
    psd_norm = psd[mask] / psd[mask].sum() if mask.sum() > 0 and psd[mask].sum() > 0 else np.array([1.0])
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd_norm)) if len(psd_norm) > 1 else 0.0

    # Band power fractions
    total_power = float(np.trapz(psd[mask], freqs[mask])) if mask.sum() > 1 else 0.0
    bands = {
        'ultra_slow_0_01': (0.005, 0.01),
        'slow_0_01_0_05': (0.01, 0.05),
        'mid_0_05_0_2': (0.05, 0.2),
        'fast_0_2_1': (0.2, 1.0),
    }
    band_power = {}
    for name, (lo, hi) in bands.items():
        band_mask = (freqs >= lo) & (freqs < hi)
        if band_mask.sum() > 1 and total_power > 0:
            band_power[name] = float(np.trapz(psd[band_mask], freqs[band_mask]) / total_power)
        else:
            band_power[name] = 0.0

    # Burst statistics
    n_bursts = len(bursts)
    burst_rate = n_bursts / (recording_duration_s / 60) if recording_duration_s > 0 else 0.0
    durations = [b['duration_s'] for b in bursts]
    mean_dur = float(np.mean(durations)) if durations else 0.0

    ibis = []
    for i in range(1, len(bursts)):
        ibis.append(bursts[i]['onset_s'] - bursts[i - 1]['onset_s'])
    mean_ibi = float(np.mean(ibis)) if ibis else 0.0
    cv_ibi = float(np.std(ibis) / np.mean(ibis)) if ibis and np.mean(ibis) > 0 else 0.0

    return {
        'dominant_freq_hz': float(dom_freq),
        'dominant_period_s': float(dom_period),
        'spectral_entropy': float(spectral_entropy),
        'total_power': total_power,
        'band_power_fractions': band_power,
        'n_bursts': n_bursts,
        'burst_rate_per_min': float(burst_rate),
        'mean_burst_duration_s': mean_dur,
        'mean_ibi_s': mean_ibi,
        'cv_ibi': cv_ibi,
    }


# ─── Figures ─────────────────────────────────────────────────────────────

def generate_spectral_figures(global_trace, freqs, psd, bursts, std_map,
                              frame_rate, output_dir, dataset_name=''):
    """Generate diagnostic figures for spectral analysis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    T = len(global_trace)
    time_axis = np.arange(T) / frame_rate

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1.2, 1, 1]})
    fig.suptitle(f'Network spectral analysis — {dataset_name}', fontsize=13, fontweight='bold')

    # Panel 1: Global trace with burst markers
    ax = axes[0]
    ax.plot(time_axis, global_trace, color='#2B5797', linewidth=0.6, alpha=0.8)
    for b in bursts:
        ax.axvspan(b['onset_s'], b['offset_s'], alpha=0.15, color='#E24B4A')
        ax.plot(b['peak_frame'] / frame_rate, b['peak_amplitude'],
                'v', color='#E24B4A', markersize=6)
    ax.set_ylabel('\u0394F/F\u2080 (weighted)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Variance-weighted global trace', fontsize=11)
    ax.set_xlim(0, time_axis[-1])

    # Panel 2: Power spectral density
    ax = axes[1]
    ax.semilogy(freqs, psd, color='#1D9E75', linewidth=1.2)
    # Mark dominant frequency
    mask = freqs > 0.005
    if mask.sum() > 0:
        peak_idx = np.argmax(psd[mask])
        dom_f = freqs[mask][peak_idx]
        ax.axvline(dom_f, color='#E24B4A', linestyle='--', alpha=0.7,
                   label=f'Dominant: {dom_f:.3f} Hz ({1/dom_f:.1f}s)')
        ax.legend(fontsize=9)
    ax.set_ylabel('Power spectral density')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Welch PSD of global trace', fontsize=11)
    ax.set_xlim(0, frame_rate / 2)

    # Panel 3: Variance weight map (tissue mask proxy)
    ax = axes[2]
    im = ax.imshow(std_map, cmap='inferno', aspect='equal')
    ax.set_title('Temporal std map (variance weight source)', fontsize=11)
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Temporal std')

    plt.tight_layout()
    path = os.path.join(output_dir, 'network_spectral_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Spectral figure saved: {path}")
    return path


# ─── Main entry point ────────────────────────────────────────────────────

def run_network_spectral(movie, frame_rate, output_dir, dataset_name='',
                         percentile_floor=5.0, burst_threshold_sd=2.0):
    """
    Full detection-free spectral analysis pipeline.

    Parameters
    ----------
    movie : np.ndarray, shape (T, Y, X)
        Raw (or motion-corrected) fluorescence movie.
    frame_rate : float
        Acquisition rate in Hz.
    output_dir : str
        Directory for outputs.
    dataset_name : str
        Label for figures.
    percentile_floor : float
        Percentile threshold for variance weights (suppress empty space).
    burst_threshold_sd : float
        SD threshold for burst detection on global trace.

    Returns
    -------
    result : dict
        Contains global_trace, freqs, psd, bursts, spectral_summary,
        weights, std_map, figure_path.
    """
    logger.info("=" * 60)
    logger.info("DEV: Network Spectral Analysis (detection-free)")
    logger.info("=" * 60)

    T, Y, X = movie.shape
    recording_duration = T / frame_rate
    logger.info(f"  Movie: {T} frames, {Y}x{X} px, {recording_duration:.1f}s at {frame_rate} Hz")

    # Step 1: Variance weights
    logger.info("  Computing variance weights...")
    weights, std_map = compute_variance_weights(movie, percentile_floor)

    # Step 2: Global trace
    logger.info("  Computing variance-weighted global trace...")
    global_trace = compute_global_trace(movie, weights)

    # Step 3: Welch PSD
    logger.info("  Computing Welch PSD...")
    freqs, psd = compute_welch_psd(global_trace, frame_rate)

    # Step 4: Burst detection
    logger.info("  Detecting population bursts...")
    bursts = detect_global_bursts(global_trace, frame_rate, burst_threshold_sd)

    # Step 5: Summary metrics
    summary = compute_spectral_summary(freqs, psd, bursts, recording_duration)
    logger.info(f"  Dominant frequency: {summary['dominant_freq_hz']:.4f} Hz "
                f"(period {summary['dominant_period_s']:.1f}s)")
    logger.info(f"  Spectral entropy: {summary['spectral_entropy']:.3f} "
                f"(0=periodic, 1=white noise)")
    logger.info(f"  Bursts: {summary['n_bursts']} detected, "
                f"{summary['burst_rate_per_min']:.1f}/min, "
                f"mean IBI {summary['mean_ibi_s']:.1f}s")

    # Step 6: Figures
    logger.info("  Generating figures...")
    dev_dir = os.path.join(output_dir, 'dev_network')
    fig_path = generate_spectral_figures(
        global_trace, freqs, psd, bursts, std_map,
        frame_rate, dev_dir, dataset_name,
    )

    # Step 7: Save data
    np.save(os.path.join(dev_dir, 'global_trace.npy'), global_trace.astype(np.float32))
    np.save(os.path.join(dev_dir, 'psd_freqs.npy'), freqs.astype(np.float32))
    np.save(os.path.join(dev_dir, 'psd_power.npy'), psd.astype(np.float32))
    np.save(os.path.join(dev_dir, 'variance_weights.npy'), weights.astype(np.float32))

    with open(os.path.join(dev_dir, 'spectral_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    burst_data = [
        {k: (int(v) if isinstance(v, (np.integer,)) else
             float(v) if isinstance(v, (np.floating, float)) else v)
         for k, v in b.items()}
        for b in bursts
    ]
    with open(os.path.join(dev_dir, 'bursts.json'), 'w') as f:
        json.dump(burst_data, f, indent=2)

    logger.info(f"  Network spectral analysis complete. Outputs in {dev_dir}/")

    return {
        'global_trace': global_trace,
        'freqs': freqs,
        'psd': psd,
        'bursts': bursts,
        'spectral_summary': summary,
        'weights': weights,
        'std_map': std_map,
        'figure_path': fig_path,
    }
