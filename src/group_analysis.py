"""
Dataset Comparison & Quality Analysis Module
=============================================

Comparative analysis of calcium imaging datasets from iPSC-derived
brain organoids.  Presents data as-is without forced clustering or
group assignments — the researcher interprets the patterns.

Approach
--------
1. Extract a feature vector per dataset from pipeline outputs
2. Select top-N neurons per dataset by quality scoring
3. Flag suspicious ROIs (anomalously high amplitude or frequency)
4. Compute population-level metrics (synchrony, pairwise correlation)
5. Generate publication-quality figures for visual comparison

Input
-----
Pipeline output directories from batch-processed datasets, each containing:
- temporal_traces.npy     (N × T)
- confidence_scores.npy   (N,)
- diagnostics.npz         (transient_count, activity_fraction, etc.)
- spatial_footprints.npz  (sparse, d1*d2 × N)
- run_info.json           (frame_rate, dims, etc.)

Output
------
- dataset_features.csv          Per-dataset feature matrix
- figures/                      All figures
- analysis_results.json         Full results with per-dataset metrics

Usage
-----
    python -m src.group_analysis \\
        --results-dir /path/to/batch_results \\
        --output /path/to/analysis \\
        --confidence-threshold 0.5

Author: Calcium Pipeline
"""

import os
import json
import logging
import warnings
import csv
from pathlib import Path

# Redirect matplotlib cache to scratch (HPC home quota is often limited)
if 'MPLCONFIGDIR' not in os.environ:
    _scratch = os.environ.get('SCRATCH_DIR', os.getcwd())
    _mpl_cache = os.path.join(_scratch, '.cache', 'matplotlib')
    os.makedirs(_mpl_cache, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = _mpl_cache
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def _abbrev(name: str, max_len: int = 20) -> str:
    """Create a short, readable dataset label for figure legends.

    Naming convention: {organoid}_{passage}_{date}_{region} - Denoised

    Examples
    --------
    'D109_3-63_040226_R7 - Denoised' -> 'D109_040226_R7'
    'D115_0-3_050326_R3 - Denoised'  -> 'D115_050326_R3'
    """
    # Strip common suffixes
    s = name.replace(' - Denoised', '').replace(' - denoised', '').strip()

    parts = s.split('_')
    line_id = parts[0] if parts else s[:4]  # 'D109'

    # Find the date part (6 digits, typically ddmmyy)
    date_part = ''
    for p in parts:
        if len(p) == 6 and p.isdigit():
            date_part = p
            break

    # Find the R part (region/repeat)
    r_part = ''
    for p in parts:
        if p.startswith('R') and len(p) <= 3 and p[1:].isdigit():
            r_part = p
            break

    # Build label: always include date for disambiguation
    if date_part and r_part:
        label = f"{line_id}_{date_part}_{r_part}"
    elif r_part:
        label = f"{line_id}_{r_part}"
    elif date_part:
        label = f"{line_id}_{date_part}"
    else:
        label = line_id

    return label[:max_len]


def _trace_snr(trace: np.ndarray) -> float:
    """
    Compute trace SNR following standard calcium imaging convention.

    SNR = peak ΔF/F amplitude / σ_baseline

    where σ_baseline is the MAD-based noise estimate from frame-to-frame
    differences (robust to transients).

    Returns values typically in the range 2–20 for good neurons.
    """
    diff = np.diff(trace)
    mad = np.median(np.abs(diff - np.median(diff)))
    noise = 1.4826 * mad / np.sqrt(2)  # MAD → σ
    if noise <= 0:
        return 0.0
    signal = np.percentile(trace, 95) - np.percentile(trace, 5)
    return float(signal / noise)


def _ensure_traces_dff(C: np.ndarray) -> np.ndarray:
    """
    Ensure traces are in ΔF/F₀ space for display and SNR computation.

    If median absolute value > 1.0, traces are likely raw fluorescence
    (values ~1e3–1e7) rather than ΔF/F₀ (values ~0, with transients
    at 0.01–5.0).  Convert per-trace using rolling baseline.

    This guards against traces_denoised.npy files generated before
    the _ensure_dff fix was added to deconvolution.
    """
    median_abs = np.median(np.abs(C))
    if median_abs <= 1.0:
        return C  # already ΔF/F₀

    logger.info(f"  _ensure_traces_dff: median |trace| = {median_abs:.1f}, "
                f"converting to ΔF/F₀")

    N, T = C.shape
    C_out = np.zeros_like(C, dtype=np.float64)
    win = max(T // 10, 50)

    for i in range(N):
        trace = C[i].astype(np.float64)
        from scipy.ndimage import minimum_filter1d
        baseline = minimum_filter1d(trace, size=win)
        # Smooth baseline to avoid division artefacts
        from scipy.ndimage import uniform_filter1d
        baseline = uniform_filter1d(baseline, size=win // 2)
        # Floor at 1% of trace median or 1st percentile, whichever is larger
        trace_median = np.median(trace[trace > 0]) if np.any(trace > 0) else 1.0
        baseline = np.maximum(baseline, max(np.percentile(trace, 1), trace_median * 0.01))
        C_out[i] = np.clip((trace - baseline) / baseline, -1.0, 100.0)

    return C_out.astype(np.float32)


def _load_valid_mask(result_path) -> np.ndarray:
    """
    Return a boolean mask of valid ROIs for a dataset.

    Boundary-touching ROIs are excluded at detection time (v2.1+),
    so all saved ROIs are valid.  For backwards compatibility with
    older results that include boundary ROIs, the boundary_touching
    file is checked if present.
    """
    from pathlib import Path
    result_path = Path(result_path)

    # Try to infer N from spike trains
    spikes_path = result_path / 'spike_trains.npy'
    traces_path = result_path / 'temporal_traces.npy'
    if spikes_path.exists():
        N = np.load(spikes_path, mmap_mode='r').shape[0]
    elif traces_path.exists():
        N = np.load(traces_path, mmap_mode='r').shape[0]
    else:
        return None

    mask = np.ones(N, dtype=bool)

    # Backwards compat: exclude boundary ROIs from older pipeline runs
    boundary_path = result_path / 'boundary_touching.npy'
    if boundary_path.exists():
        boundary = np.load(boundary_path).astype(bool)
        if len(boundary) == N:
            mask &= ~boundary

    return mask


def _has_corrupted_values(trace: np.ndarray, threshold: float = 1e6) -> bool:
    """Check if a trace has corrupted/overflow values (e.g. 1e9+)."""
    return bool(np.any(np.abs(trace) > threshold))


@dataclass
class DatasetMetrics:
    """Per-dataset summary metrics extracted from pipeline outputs."""
    name: str
    filepath: str

    # Neuron counts
    n_neurons: int = 0
    n_confident: int = 0
    n_selected: int = 0
    n_hard_rejected: int = 0       # neurons rejected by hard gates (deconv failures)
    n_overlap_removed: int = 0     # duplicate ROIs removed by ≥50% spatial overlap
    n_distance_removed: int = 0    # ROIs removed by minimum centroid distance filter

    # Selected neuron info (for transparency)
    selected_indices: Optional[np.ndarray] = None    # indices into confident set
    selected_roi_indices: Optional[np.ndarray] = None  # original ROI indices (into full array)
    selected_quality: Optional[np.ndarray] = None    # quality scores
    selected_traces: Optional[np.ndarray] = None     # (n_selected, T) denoised
    selected_raw_traces: Optional[np.ndarray] = None # (n_selected, T) raw fluorescence
    selected_spikes: Optional[np.ndarray] = None     # (n_selected, T) spike trains
    selected_roi_crops: Optional[List] = None        # list of (crop_img, footprint_img) tuples per neuron

    # Per-neuron arrays (all confident neurons)
    all_quality_scores: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None

    # Per-neuron arrays (selected neurons) — precomputed to avoid storing full traces
    neuron_spike_rates: Optional[np.ndarray] = None      # (n_selected,) events/10s
    neuron_spike_amplitudes: Optional[np.ndarray] = None  # (n_selected,) mean dF/F per neuron
    neuron_is_active: Optional[np.ndarray] = None         # (n_selected,) bool: ≥1 validated transient

    # Dataset-level metrics (computed from SELECTED neurons only)
    mean_spike_rate: float = 0.0
    median_spike_rate: float = 0.0
    mean_spike_amplitude: float = 0.0
    n_active: int = 0                    # active selected neurons (rate > 0)
    active_fraction: float = 0.0         # n_active / n_neurons (total detections)
    pairwise_correlation_mean: float = 0.0
    synchrony_index: float = 0.0
    mean_iei: float = 0.0
    cv_iei: float = 0.0
    n_network_bursts: int = 0
    burst_rate: float = 0.0
    mean_burst_participation: float = 0.0
    mean_quality_score: float = 0.0

    # Temporal
    frame_rate: float = 2.0
    n_frames: int = 0
    duration_seconds: float = 0.0

    # Motion quality
    motion_max_shift: float = 0.0
    motion_mean_shift: float = 0.0
    motion_residual_std: float = 0.0
    motion_excluded: bool = False

    # Baseline drift quality
    baseline_drift: float = 0.0       # population-mean drift ratio (Q4-Q1)/std
    baseline_drift_excluded: bool = False

    # Amplitude tracking (from run_info.json, populated if available)
    amplitude_tracking: Optional[List] = None  # per-stage amplitude diagnostics

    # Genotype (v2.0)
    genotype: str = ''                 # 'Control', 'Mutant', or 'Unknown'
    
    # Manual override
    manually_inactive: bool = False    # True if visually confirmed no activity
    line_id: str = ''                  # e.g. '3-63', '1-12'


# Feature names used for clustering (must match DatasetMetrics attributes)
# Focused on biological metrics — removed imaging-quality confounds
FEATURE_NAMES = [
    ('mean_spike_rate',          'Event Rate (events/10s)'),
    ('median_spike_rate',        'Median Event Rate (events/10s)'),
    ('mean_spike_amplitude',     'Mean Transient Amplitude (ΔF/F₀)'),
    ('pairwise_correlation_mean','Mean Pairwise Corr. (r)'),
    ('synchrony_index',          'Synchrony Index'),
    ('mean_iei',                 'Mean IEI (s)'),
    ('cv_iei',                   'IEI CV'),
    ('n_network_bursts',         'Network Bursts'),
    ('burst_rate',               'Burst Rate (bursts/10s)'),
    ('mean_burst_participation', 'Burst Participation'),
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset_metrics(
    result_dir: str,
    name: str,

    frame_rate_override: Optional[float] = None,
    min_roi_distance: float = 15.0,
) -> Optional[DatasetMetrics]:
    """Load pipeline outputs, score neuron quality, select by threshold.

    Parameters
    ----------
        If > 0, select this many top neurons (legacy mode).
        Minimum quality score (0–1) for neuron inclusion. Default 0.8.
    min_roi_distance : float
        Minimum distance in pixels between selected neuron centroids.
        Pairs closer than this are deduplicated, keeping the higher-quality
        one.  Default 15.0 pixels.  Set to 0 to disable.
    """
    result_path = Path(result_dir)

    denoised_path = result_path / 'traces_denoised.npy'
    spikes_path = result_path / 'spike_trains.npy'
    traces_path = result_path / 'temporal_traces.npy'
    conf_path = result_path / 'confidence_scores.npy'

    if not traces_path.exists():
        logger.warning(f"  {name}: temporal_traces.npy not found, skipping")
        return None

    C_raw = np.load(traces_path)
    confidence = np.load(conf_path) if conf_path.exists() else np.ones(C_raw.shape[0])

    # Load raw fluorescence traces (for local ΔF/F amplitude measurement)
    raw_fluor_path = result_path / 'temporal_traces_raw.npy'
    C_raw_fluorescence = np.load(raw_fluor_path) if raw_fluor_path.exists() else None

    # Check which amplitude method was used for this dataset
    amplitude_method = 'global_dff'  # default
    pipeline_json_path = result_path / 'pipeline_results.json'
    if pipeline_json_path.exists():
        try:
            import json as _json
            with open(pipeline_json_path) as _f:
                _pres = _json.load(_f)
            amplitude_method = _pres.get('amplitude_method', 'global_dff')
        except Exception:
            pass

    # Load deconvolved data
    has_deconv = denoised_path.exists() and spikes_path.exists()
    if has_deconv:
        C_denoised = _ensure_traces_dff(np.load(denoised_path))
        S = np.load(spikes_path)
    else:
        logger.warning(f"  {name}: no deconvolution data — quality selection not possible, skipping")
        return None

    # Load and apply edge ROI exclusion
    boundary_path = result_path / 'boundary_touching.npy'
    if boundary_path.exists():
        boundary = np.load(boundary_path).astype(bool)
        n_edge = int(boundary.sum())
        if n_edge > 0:
            logger.info(f"  {name}: excluding {n_edge} edge ROIs")
    else:
        boundary = np.zeros(C_denoised.shape[0], dtype=bool)

    info_path = result_path / 'run_info.json'
    frame_rate = frame_rate_override or 2.0
    amp_tracking_data = None
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
            config = info.get('config', {})
            frame_rate = frame_rate_override or config.get('frame_rate', 2.0)
            amp_tracking_data = info.get('amplitude_tracking', None)
            if amp_tracking_data:
                logger.info(f"  {name}: loaded amplitude tracking ({len(amp_tracking_data)} stages)")

    N, T = C_denoised.shape
    duration = T / frame_rate
    dur_min = duration / 60.0

    # ── Selection: include all ROIs that survived deconvolution ──────────
    # Boundary-touching seeds are excluded at detection time.
    # The deconvolution stage gates on s_min, 3.5σ noise, and transient
    # duration — traces that fail have C_denoised and S zeroed.
    # Selection here simply checks for deconvolution output.
    has_events = np.array([np.sum(S[i] > 0) > 0 for i in range(N)])
    has_signal = np.array([np.max(C_denoised[i]) - np.min(C_denoised[i]) > 1e-10
                           for i in range(N)])
    sel_mask = has_events & has_signal

    n_deconv_pass = int(sel_mask.sum())
    n_deconv_fail = N - n_deconv_pass
    logger.info(f"  {name}: {N} total ROIs, {n_deconv_pass} with deconvolved events, "
                f"{n_deconv_fail} deconvolution failures")

    if n_deconv_pass < 3:
        logger.warning(f"  {name}: only {n_deconv_pass} ROIs passed deconvolution, skipping")
        return None

    sel_idx = np.where(sel_mask)[0]
    original_roi_idx = sel_idx.copy()
    actual_n = len(sel_idx)

    C_sel = C_denoised[sel_idx]
    S_sel = S[sel_idx]
    R_sel = C_raw[sel_idx]

    # Raw fluorescence traces for local ΔF/F amplitude measurement
    R_fluor_sel = None
    if C_raw_fluorescence is not None:
        R_fluor_sel = C_raw_fluorescence[sel_idx]

    # ── Load spatial footprints and compute ROI crops ────────────────────
    # Each crop is a dict with 'max_proj', 'baseline', 'contour' arrays
    roi_crops = None
    A_sparse = None
    footprint_path = result_path / 'spatial_footprints.npz'
    max_proj_path = result_path / 'max_projection.npy'
    mean_proj_path = result_path / 'mean_projection.npy'
    info_path_spatial = result_path / 'run_info.json'
    try:
        if footprint_path.exists():
            from scipy.sparse import load_npz
            A_sparse = load_npz(footprint_path)

            # Get image dimensions
            dims = None
            if info_path_spatial.exists():
                with open(info_path_spatial) as _f:
                    _info = json.load(_f)
                    if 'dims' in _info:
                        dims = tuple(_info['dims'])
                    elif 'd1' in _info and 'd2' in _info:
                        dims = (int(_info['d1']), int(_info['d2']))

            # Load projection images if available
            max_proj = np.load(max_proj_path) if max_proj_path.exists() else None
            mean_proj = np.load(mean_proj_path) if mean_proj_path.exists() else None

            if dims is not None:
                d1, d2 = dims
                roi_crops = []
                crop_radius = 35  # pixels around centroid

                for sel_i in range(actual_n):
                    orig_roi = int(original_roi_idx[sel_i])
                    if orig_roi >= A_sparse.shape[1]:
                        roi_crops.append(None)
                        continue

                    # Get footprint as 2D
                    fp = A_sparse[:, orig_roi].toarray().ravel()
                    if len(fp) != d1 * d2:
                        roi_crops.append(None)
                        continue
                    fp_2d = fp.reshape(d1, d2)

                    # Find centroid
                    ys, xs = np.where(fp_2d > 0)
                    if len(ys) == 0:
                        roi_crops.append(None)
                        continue
                    cy, cx = int(np.mean(ys)), int(np.mean(xs))

                    # Crop region
                    y0 = max(0, cy - crop_radius)
                    y1 = min(d1, cy + crop_radius)
                    x0 = max(0, cx - crop_radius)
                    x1 = min(d2, cx + crop_radius)

                    # Binary contour of the footprint for overlay
                    fp_mask = (fp_2d > 0).astype(np.uint8)
                    contour_crop = fp_mask[y0:y1, x0:x1]

                    crop_data = {'contour': contour_crop}

                    # Crop from max projection (peak fluorescence)
                    if max_proj is not None and max_proj.shape == (d1, d2):
                        crop_data['max_proj'] = max_proj[y0:y1, x0:x1]

                    # Crop from mean projection (baseline)
                    if mean_proj is not None and mean_proj.shape == (d1, d2):
                        crop_data['baseline'] = mean_proj[y0:y1, x0:x1]

                    roi_crops.append(crop_data)

                n_with_crops = sum(1 for c in roi_crops if c is not None)
                has_projs = max_proj is not None or mean_proj is not None
                logger.info(f"  {name}: spatial crops for {n_with_crops}/{actual_n} neurons"
                            f" (projections: {'yes' if has_projs else 'no — run batch again to save projections'})")
            else:
                logger.info(f"  {name}: no image dims available, skipping spatial crops")
    except Exception as e:
        logger.warning(f"  {name}: spatial crop extraction failed: {e}")
        roi_crops = None

    # ── Distance deduplication: remove ROIs closer than min_roi_distance ──
    # Two detections with centres < min_roi_distance pixels apart are
    # effectively sampling the same structure.  Keeping both inflates n
    # and creates correlated duplicates.  The higher-SNR one is kept.
    roi_snr = np.array([_trace_snr(C_sel[j]) for j in range(actual_n)])
    n_distance_removed = 0
    if A_sparse is not None and min_roi_distance > 0 and actual_n >= 2:
        try:
            _dims = None
            _info_path = result_path / 'run_info.json'
            if _info_path.exists():
                with open(_info_path) as _f:
                    _info_data = json.load(_f)
                    if 'dims' in _info_data:
                        _dims = tuple(_info_data['dims'])
                    elif 'd1' in _info_data and 'd2' in _info_data:
                        _dims = (int(_info_data['d1']), int(_info_data['d2']))

            if _dims is not None:
                d1, d2 = _dims
                centroids = np.zeros((actual_n, 2))
                centroid_valid = np.ones(actual_n, dtype=bool)

                for ci in range(actual_n):
                    roi_col = int(original_roi_idx[ci])
                    if roi_col >= A_sparse.shape[1]:
                        centroid_valid[ci] = False
                        continue
                    fp = A_sparse[:, roi_col].toarray().ravel()
                    if len(fp) != d1 * d2:
                        centroid_valid[ci] = False
                        continue
                    fp_2d = fp.reshape(d1, d2)
                    ys, xs = np.where(fp_2d > 0)
                    if len(ys) == 0:
                        centroid_valid[ci] = False
                        continue
                    centroids[ci] = [np.mean(ys), np.mean(xs)]

                dist_keep = np.ones(actual_n, dtype=bool)
                for ci in range(actual_n):
                    if not dist_keep[ci] or not centroid_valid[ci]:
                        continue
                    for cj in range(ci + 1, actual_n):
                        if not dist_keep[cj] or not centroid_valid[cj]:
                            continue
                        dist = np.sqrt((centroids[ci, 0] - centroids[cj, 0])**2 +
                                       (centroids[ci, 1] - centroids[cj, 1])**2)
                        if dist < min_roi_distance:
                            if roi_snr[ci] >= roi_snr[cj]:
                                dist_keep[cj] = False
                            else:
                                dist_keep[ci] = False
                                break
                            n_distance_removed += 1

                if n_distance_removed > 0:
                    logger.info(f"  {name}: removing {n_distance_removed}/{actual_n} "
                                f"ROIs closer than {min_roi_distance:.0f}px")
                    sel_idx = sel_idx[dist_keep]
                    original_roi_idx = original_roi_idx[dist_keep]
                    C_sel = C_sel[dist_keep]
                    S_sel = S_sel[dist_keep]
                    R_sel = R_sel[dist_keep]
                    roi_snr = roi_snr[dist_keep]
                    if R_fluor_sel is not None:
                        R_fluor_sel = R_fluor_sel[dist_keep]
                    if roi_crops is not None:
                        roi_crops = [c for c, k in zip(roi_crops, dist_keep) if k]
                    actual_n = len(sel_idx)
                    if actual_n < 3:
                        logger.warning(f"  {name}: only {actual_n} neurons remain after "
                                       f"distance filter, skipping dataset")
                        return None
        except Exception as e:
            logger.warning(f"  {name}: centroid distance filter failed: {e}")

    logger.info(f"  {name}: selected {actual_n}/{N} ROIs "
                f"({n_deconv_fail} deconv failures, {n_distance_removed} distance-deduped)")

    # ── Compute metrics from SELECTED neurons ────────────────────────────
    # Spike rates (events per 10 seconds)
    spike_counts = np.array([np.sum(S_sel[j] > 0) for j in range(actual_n)])
    spike_rates = spike_counts / duration * 10.0 if duration > 0 else spike_counts
    mean_spike_rate = float(np.mean(spike_rates))
    median_spike_rate = float(np.median(spike_rates))

    # Spike amplitudes — method depends on pipeline configuration:
    #   direct / local_dff: measure each event as local ΔF/F from raw fluorescence
    #   global_dff / local_background: measure from corrected traces
    _use_local = amplitude_method in ('direct', 'local_dff')
    _amp_raw = R_fluor_sel if _use_local else None
    all_amps = _measure_transient_amplitudes(
        C_sel, S_sel, frame_rate, C_raw_fluorescence=_amp_raw)
    mean_spike_amp = float(np.mean(all_amps)) if all_amps else 0.0

    # Detailed per-neuron log for verification
    logger.info(f"  {name}: duration={duration:.1f}s ({dur_min:.2f} min), "
                f"spike counts per neuron: {spike_counts.tolist()}")
    logger.info(f"  {name}: rates/10s: {[f'{r:.1f}' for r in spike_rates]}")
    if all_amps:
        logger.info(f"  {name}: transient amplitudes (ΔF/F₀): {[f'{a:.3f}' for a in all_amps]}")

    # ── Correlation and synchrony ──────────────────────────────────────────
    # Uses DENOISED TRACES (C_sel) for correlation/synchrony - see docstrings
    # for rationale (trace correlations more reliable than spike correlations
    # at 2 Hz due to temporal discretization issues).
    # Minimum n=5: below this, pairwise correlation from too few pairs
    # (<10) is unreliable and excluded from statistical comparisons.
    MIN_N_CORR = 5
    if actual_n >= MIN_N_CORR:
        corr_mean, _ = _pairwise_correlations(C_sel, S=S_sel)
        sync_idx = _synchrony_index(C_sel, S=S_sel, frame_rate=frame_rate)
    else:
        corr_mean = float('nan')
        sync_idx  = float('nan')
        logger.info(f"  {name}: n_selected={actual_n} < {MIN_N_CORR} — "
                    f"correlation/synchrony set to NaN (insufficient pairs)")

    # ── IEI from SPIKE TRAINS ─────────────────────────────────────────────
    # Uses deconvolved spike events (S_sel > 0) for inter-event intervals
    mean_iei, cv_iei = _inter_event_intervals_from_spikes(S_sel, frame_rate)

    # ── Network bursts from SPIKE TRAINS ──────────────────────────────────
    # Uses population spike synchrony (fraction of neurons with S > 0)
    n_bursts, burst_rate_val, burst_part = _network_bursts_from_spikes(S_sel, frame_rate)

    ds = DatasetMetrics(
        name=name, filepath=str(result_path),
        n_neurons=N, n_confident=n_deconv_pass, n_selected=actual_n,
        n_hard_rejected=n_deconv_fail, n_overlap_removed=0,
        n_distance_removed=n_distance_removed,
        selected_indices=sel_idx,
        selected_roi_indices=original_roi_idx,
        selected_quality=roi_snr,
        selected_traces=C_sel,
        selected_raw_traces=R_sel,
        selected_spikes=S_sel,
        selected_roi_crops=roi_crops,
        all_quality_scores=None,
        confidence_scores=None,
        mean_spike_rate=mean_spike_rate,
        median_spike_rate=median_spike_rate,
        mean_spike_amplitude=mean_spike_amp,
        pairwise_correlation_mean=corr_mean,
        synchrony_index=sync_idx,
        mean_iei=mean_iei, cv_iei=cv_iei,
        n_network_bursts=n_bursts,
        burst_rate=float(burst_rate_val),
        mean_burst_participation=burst_part,
        mean_quality_score=float(np.mean(roi_snr)),
        frame_rate=frame_rate,
        n_frames=T, duration_seconds=duration,
        genotype=_extract_genotype(name),
        line_id=_extract_line_id(name),
    )

    logger.info(f"    event_rate={mean_spike_rate:.1f}/10s, sync={sync_idx:.3f}, "
                f"mean_snr={np.mean(roi_snr):.3f}, "
                f"genotype={ds.genotype}, line={ds.line_id}")


    # ── Baseline drift (population-level) ────────────────────────────────
    # Measure drift on raw traces of selected neurons.
    # Compare mean fluorescence in first vs last quarter of recording.
    # High population-median drift ratio indicates bleach correction failure.
    drift_ratio = 0.0
    if R_sel is not None and R_sel.shape[1] > 10:
        T_drift = R_sel.shape[1]
        q1_slice = slice(0, T_drift // 4)
        q4_slice = slice(3 * T_drift // 4, T_drift)
        neuron_drifts = []
        for j in range(R_sel.shape[0]):
            trace = R_sel[j]
            q1_mean = np.mean(trace[q1_slice])
            q4_mean = np.mean(trace[q4_slice])
            trace_std = np.std(trace)
            if trace_std > 1e-10:
                neuron_drifts.append(abs(q4_mean - q1_mean) / trace_std)
        if neuron_drifts:
            drift_ratio = float(np.median(neuron_drifts))
    logger.info(f"    baseline_drift={drift_ratio:.3f}")

    # ── Motion quality ───────────────────────────────────────────────────
    shifts_path = result_path / 'motion_shifts.npy'
    mc_info = {}
    if info_path.exists():
        with open(info_path) as f:
            mc_info = json.load(f).get('motion_correction', {})

    motion_max = mc_info.get('max_shift_y', 0.0) + mc_info.get('max_shift_x', 0.0)
    motion_mean = mc_info.get('mean_shift_y', 0.0) + mc_info.get('mean_shift_x', 0.0)

    motion_residual = 0.0
    if shifts_path.exists():
        shifts = np.load(shifts_path)
        if shifts.ndim == 2 and shifts.shape[0] > 2:
            shift_diffs = np.diff(shifts, axis=0)
            motion_residual = float(np.std(np.sqrt(
                shift_diffs[:, 0]**2 + shift_diffs[:, 1]**2
            )))

    ds.motion_max_shift = float(motion_max)
    ds.motion_mean_shift = float(motion_mean)
    ds.motion_residual_std = motion_residual
    ds.baseline_drift = drift_ratio
    ds.amplitude_tracking = amp_tracking_data

    # ── Precompute per-neuron arrays (for genotype analysis) ─────────────
    # These allow us to free the full (N, T) trace matrices below.
    dur_s = ds.duration_seconds if ds.duration_seconds > 0 else 1.0
    ds.neuron_spike_rates = np.array([
        np.sum(S_sel[j] > 0) / dur_s * 10.0
        for j in range(actual_n)
    ])
    
    # Active fraction: neurons with ≥1 validated transient in the selected
    # set, as a proportion of ALL detections for this dataset.
    # This measures how many of the total detected ROIs survived quality
    # selection AND showed genuine activity.
    ds.neuron_is_active = ds.neuron_spike_rates > 0
    ds.n_active = int(ds.neuron_is_active.sum())
    ds.active_fraction = ds.n_active / max(N, 1)
    
    logger.info(f"  {name}: active fraction = {ds.n_active}/{N} "
                f"({ds.active_fraction:.1%}) "
                f"[{ds.n_active} active selected out of {N} total detections]")
    all_amps_list = _measure_transient_amplitudes(
        C_sel, S_sel, frame_rate, C_raw_fluorescence=_amp_raw)
    # _measure_transient_amplitudes skips neurons with no spikes, so its length
    # may be < actual_n. Build a per-neuron array with 0.0 for silent neurons.
    if len(all_amps_list) == actual_n:
        ds.neuron_spike_amplitudes = np.array(all_amps_list)
    else:
        # Recompute per-neuron to get correct alignment
        per_neuron_amps = np.zeros(actual_n)
        amp_idx = 0
        for j in range(actual_n):
            spike_frames = np.where(S_sel[j] > 0)[0]
            if len(spike_frames) > 0 and amp_idx < len(all_amps_list):
                per_neuron_amps[j] = all_amps_list[amp_idx]
                amp_idx += 1
        ds.neuron_spike_amplitudes = per_neuron_amps

    # ── Free large trace arrays to reduce memory ────────────────────────
    # The full (N, T) matrices are only needed for per-dataset diagnostic
    # figures; we keep them only for a limited number of datasets.
    # Dataset-level metrics are already computed above.
    ds.selected_traces = None
    ds.selected_raw_traces = None
    ds.selected_spikes = None

    return ds


# =============================================================================
# METRIC COMPUTATION HELPERS
# =============================================================================
#
# DATA FLOW AND CONSISTENCY NOTES
# ===============================
#
# All metrics are computed from the SELECTED neurons (top N by quality score).
# The data sources are:
#
#   C_sel (selected_traces)   - Denoised calcium traces from OASIS
#   S_sel (selected_spikes)   - Deconvolved spike trains from OASIS  
#   R_sel (selected_raw_traces) - Raw fluorescence (for QC only)
#
# METRIC SOURCE MAPPING:
#
#   Spike Rate          → S_sel (count of S > 0 events per second)
#   Transient amplitude     → S_sel (mean amplitude of S > 0 values)
#   Pairwise Correlation→ C_sel (trace correlations, NOT spike correlations)
#   Synchrony Index     → C_sel (population coupling + co-activation)
#   Mean IEI            → S_sel (inter-spike intervals)
#   CV IEI              → S_sel (firing regularity)
#   Network Bursts      → S_sel (population spike synchrony)
#   Burst Rate          → S_sel (bursts per 10s)
#   Burst Participation → S_sel (fraction of neurons in bursts)
#
# RATIONALE FOR TRACE-BASED CORRELATION/SYNCHRONY:
#
#   At 2 Hz imaging, the temporal resolution is too coarse for reliable
#   spike-train correlations. Two neurons firing within 50ms of each other
#   may appear on different frames, giving spuriously low correlation.
#   
#   The GCaMP6s indicator has ~1s decay time, which acts as a natural
#   temporal integration window. Denoised trace correlations therefore
#   capture functional coupling more reliably than spike-train methods.
#
# QUALITY CONTROL:
#
#   - Only neurons with confidence >= 0.3 are considered
#   - Edge-touching ROIs are excluded (boundary_touching.npy)
#   - Top N neurons by composite quality score are selected
#   - Quality score combines: SNR, spike discreteness, baseline stability,
#     amplitude consistency, and transient duration
#
# =============================================================================

def _get_neuron_rates(ds) -> np.ndarray:
    """Get per-neuron spike rates for ACTIVE neurons only (rate > 0)."""
    if ds.neuron_spike_rates is not None and len(ds.neuron_spike_rates) > 0:
        rates = ds.neuron_spike_rates
        return rates[rates > 0]
    if ds.selected_spikes is None:
        return np.array([])
    dur_s = ds.duration_seconds if ds.duration_seconds > 0 else 1.0
    rates = np.array([np.sum(ds.selected_spikes[j] > 0) / dur_s * 10.0
                     for j in range(ds.selected_spikes.shape[0])])
    return rates[rates > 0]


def _get_neuron_amplitudes(ds) -> np.ndarray:
    """Get per-neuron spike amplitudes for ACTIVE neurons only (amp > 0)."""
    if ds.neuron_spike_amplitudes is not None and len(ds.neuron_spike_amplitudes) > 0:
        amps = ds.neuron_spike_amplitudes
        return amps[amps > 0]
    if ds.selected_traces is None or ds.selected_spikes is None:
        return np.array([])
    amps = _measure_transient_amplitudes(
        ds.selected_traces, ds.selected_spikes, ds.frame_rate)
    amps = np.array(amps) if amps else np.array([])
    return amps[amps > 0] if len(amps) > 0 else amps


def _recording_metric(ds, attr: str):
    """Get a recording-level metric value, using active-neuron-only means
    for spike rate and amplitude (consistent across all figures).
    
    Returns float or None if no active neurons for rate/amplitude metrics.
    """
    if attr == 'mean_spike_rate':
        rates = _get_neuron_rates(ds)
        return float(np.mean(rates)) if len(rates) > 0 else None
    elif attr == 'mean_spike_amplitude':
        amps = _get_neuron_amplitudes(ds)
        return float(np.mean(amps)) if len(amps) > 0 else None
    elif attr == 'total_events':
        if ds.selected_spikes is not None:
            return float(np.sum(ds.selected_spikes > 0))
        rates = _get_neuron_rates(ds)
        if len(rates) > 0 and ds.duration_seconds > 0:
            return float(np.sum(rates) * ds.duration_seconds / 10.0)
        return 0.0
    else:
        return getattr(ds, attr, None)


def _pairwise_correlations(C: np.ndarray,
                           S: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Compute mean and median pairwise Pearson correlations.

    Uses **denoised calcium traces** (C) rather than spike trains (S).
    
    Rationale: For calcium imaging at low frame rates (2 Hz), the slow 
    GCaMP6s dynamics (~1s decay) provide a natural temporal integration 
    window that makes trace-based correlations a reliable measure of 
    functional co-activation. Spike-train correlations at this frame rate 
    are unreliable because co-firing neurons rarely spike on the exact 
    same frame due to temporal discretization.

    Parameters
    ----------
    C : array (N, T)
        Denoised calcium traces from OASIS deconvolution.
    S : array (N, T), optional
        Deconvolved spike trains (not used — kept for API compatibility).

    Returns
    -------
    mean_r : float
        Mean of all pairwise Pearson r values (upper triangle).
    median_r : float
        Median of all pairwise Pearson r values.
    
    Notes
    -----
    - Input should be the denoised traces (C) from OASIS, NOT raw fluorescence.
    - The denoised trace represents the estimated calcium concentration,
      which directly reflects underlying neural activity.
    """
    N = C.shape[0]
    if N < 2:
        return 0.0, 0.0
    if N < 5:
        logger.debug(f"  _pairwise_correlations: N={N} neurons — correlation "
                     f"from {N*(N-1)//2} pairs, treat with caution")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R = np.corrcoef(C)

    triu_idx = np.triu_indices(N, k=1)
    r_vals = R[triu_idx]
    r_vals = r_vals[np.isfinite(r_vals)]
    if len(r_vals) == 0:
        return 0.0, 0.0
    return float(np.mean(r_vals)), float(np.median(r_vals))


def _synchrony_index(C: np.ndarray, fraction_threshold: float = 0.20,
                     S: Optional[np.ndarray] = None,
                     frame_rate: float = 2.0) -> float:
    """Compute population synchrony index from denoised calcium traces.

    Uses two complementary approaches and returns their weighted mean:

    1. **Population coupling** (60% weight) — for each neuron, correlate its 
       denoised trace with the mean of all other traces (population mean-field).
       Average across neurons. High values mean neurons track population activity.

    2. **Co-activation fraction** (40% weight) — fraction of time when a 
       substantial proportion of neurons are simultaneously above baseline.
       Uses denoised traces because the slow calcium dynamics (~1s decay)
       provide an appropriate coincidence window for 2 Hz imaging.

    The result is scaled to [0, 1] where 0 = independent firing,
    1 = perfectly synchronised population.

    Parameters
    ----------
    C : array (N, T)
        Denoised calcium traces from OASIS deconvolution.
    fraction_threshold : float
        Minimum fraction of neurons co-active for a "synchronous" frame.
    S : array (N, T), optional
        Deconvolved spike trains (not used for primary metric).
    frame_rate : float
        Sampling rate in Hz.

    Returns
    -------
    float
        Synchrony index in [0, 1].
    
    Notes
    -----
    Using denoised traces (C) rather than spike trains (S) is intentional:
    - At 2 Hz, spike trains have poor temporal resolution for coincidence
    - The ~1s GCaMP6s decay acts as a natural integration window
    - Trace correlations capture functional coupling more reliably
    """
    N, T = C.shape
    if N < 3:
        return 0.0

    # ── 1. Population coupling ───────────────────────────────────────────
    # For each neuron, compute Pearson r with the mean of all OTHER neurons.
    # This measures how strongly each neuron tracks the population.
    pop_r = np.zeros(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N):
            others = np.delete(C, i, axis=0)
            pop_mean = others.mean(axis=0)

            std_i = np.std(C[i])
            std_p = np.std(pop_mean)
            if std_i < 1e-10 or std_p < 1e-10:
                pop_r[i] = 0.0
                continue

            r = np.corrcoef(C[i], pop_mean)[0, 1]
            pop_r[i] = r if np.isfinite(r) else 0.0

    # Clip negatives (anti-correlated neurons don't contribute to synchrony)
    pop_coupling = float(np.mean(np.maximum(pop_r, 0)))

    # ── 2. Co-activation fraction ────────────────────────────────────────
    # Use denoised traces: a neuron is "active" if above baseline + 2σ noise
    diffs = np.diff(C, axis=1)
    noise = np.median(np.abs(diffs), axis=1) / 0.6745
    noise[noise == 0] = 1e-10
    baseline = np.percentile(C, 20, axis=1, keepdims=True)
    active = (C - baseline) > (2.0 * noise[:, np.newaxis])

    frac_per_frame = active.mean(axis=0)
    coactivation = float((frac_per_frame >= fraction_threshold).mean())

    # ── Combine: weighted mean ───────────────────────────────────────────
    # Population coupling is more robust; co-activation captures burst events
    sync_index = 0.6 * pop_coupling + 0.4 * coactivation

    return float(np.clip(sync_index, 0, 1))


def _inter_event_intervals_from_spikes(
    S: np.ndarray, frame_rate: float
) -> Tuple[float, float]:
    """Compute inter-event interval (IEI) statistics from deconvolved spike trains.
    
    This is the PRIMARY IEI computation method, using the discrete spike events
    detected by OASIS deconvolution rather than calcium trace peaks.
    
    Parameters
    ----------
    S : array (N, T)
        Deconvolved spike trains from OASIS. Non-zero values indicate spike events.
        The amplitude of S reflects the estimated spike magnitude (Ca2+ transient size).
    frame_rate : float
        Sampling rate in Hz.
    
    Returns
    -------
    mean_iei : float
        Mean inter-event interval in seconds across all neurons.
    cv_iei : float
        Coefficient of variation (SD/mean) of IEIs. Higher values indicate
        more irregular/bursty firing patterns.
    
    Notes
    -----
    - Uses S > 0 to identify spike frames (OASIS outputs positive values for spikes)
    - Pools IEIs across all neurons to get population-level statistics
    - Requires at least 2 spikes per neuron to compute IEIs for that neuron
    - Returns (0, 0) if fewer than 3 IEIs total across all neurons
    """
    all_ieis = []
    for i in range(S.shape[0]):
        spike_frames = np.where(S[i] > 0)[0]
        if len(spike_frames) >= 2:
            ieis = np.diff(spike_frames) / frame_rate
            all_ieis.extend(ieis.tolist())
    if len(all_ieis) < 3:
        return 0.0, 0.0
    arr = np.array(all_ieis)
    m = float(np.mean(arr))
    return m, float(np.std(arr) / m) if m > 0 else 0.0


def _network_bursts_from_spikes(
    S: np.ndarray, frame_rate: float,
    threshold: float = 0.15, min_gap: float = 1.0,
) -> Tuple[int, float, float]:
    """Detect network bursts from deconvolved spike trains.
    
    This is the PRIMARY burst detection method, using discrete spike events
    from OASIS deconvolution rather than calcium trace thresholding.
    
    A network burst is defined as a time point where the fraction of neurons
    spiking exceeds `threshold`, with bursts separated by at least `min_gap`.
    
    Parameters
    ----------
    S : array (N, T)
        Deconvolved spike trains from OASIS. S[i,t] > 0 indicates neuron i
        fired at frame t.
    frame_rate : float
        Sampling rate in Hz.
    threshold : float
        Minimum fraction of neurons that must spike simultaneously to count
        as a network burst. Default 0.15 (15% of neurons).
    min_gap : float
        Minimum time in seconds between distinct burst peaks.
    
    Returns
    -------
    n_bursts : int
        Number of detected network bursts.
    burst_rate : float
        Burst rate in bursts per 10 seconds.
    mean_participation : float
        Mean fraction of neurons participating in detected bursts.
    
    Notes
    -----
    - Uses binary spike detection (S > 0) to compute population activity
    - Peak detection on population spike rate identifies burst events
    - Lower threshold (0.15) than trace-based method because spike detection
      is more temporally precise
    """
    from scipy.signal import find_peaks
    N, T = S.shape
    dur_s = T / frame_rate

    # Population spike rate per frame (fraction of neurons spiking)
    active = (S > 0).astype(float)
    pop_rate = active.mean(axis=0)

    peaks, _ = find_peaks(
        pop_rate, height=threshold,
        distance=max(1, int(min_gap * frame_rate)),
    )
    n = len(peaks)
    rate = n / dur_s * 10.0 if dur_s > 0 else 0.0
    part = float(np.mean(pop_rate[peaks])) if n > 0 else 0.0
    return n, rate, part


def _measure_transient_amplitudes(
    C: np.ndarray, S: np.ndarray, frame_rate: float,
    C_raw_fluorescence: Optional[np.ndarray] = None,
    baseline_window_s: float = 1.0,
    baseline_offset_s: float = 0.2,
    peak_window_s: float = 0.5,
) -> List[float]:
    """
    Measure calcium transient amplitudes as local ΔF/F₀ per event.
    
    For each detected spike event, computes the amplitude as:
        amplitude = (peak_F - baseline_F) / baseline_F
    
    where baseline_F and peak_F come from the raw fluorescence trace
    (if available) rather than the globally corrected ΔF/F₀ trace.
    This makes each event self-referenced: its amplitude reflects the
    actual calcium-driven fluorescence increase relative to the
    immediately preceding baseline, independent of any global drift,
    bleach correction, or baseline estimation artefacts.
    
    When raw fluorescence traces are not available, falls back to
    measuring peak − baseline on the denoised trace (legacy behaviour).
    
    Parameters
    ----------
    C : array (N, T)
        Denoised calcium traces (used for spike timing from S, and as
        fallback for amplitude measurement).
    S : array (N, T)
        Deconvolved spike trains from OASIS (used only for spike timing).
    frame_rate : float
        Imaging frame rate in Hz.
    C_raw_fluorescence : array (N, T), optional
        Raw fluorescence traces (NOT ΔF/F).  If provided, amplitudes are
        measured as local ΔF/F from these traces, giving robust event
        amplitudes independent of global baseline correction.
    baseline_window_s : float
        Duration of window before spike for baseline estimation (default 1.0s).
    baseline_offset_s : float
        Gap between baseline window end and spike time (default 0.2s).
    peak_window_s : float
        Duration of window after spike to search for peak (default 0.5s).
    
    Returns
    -------
    amplitudes : list of float
        Per-neuron mean transient amplitudes in ΔF/F₀ units.
        Returns empty list if no valid transients found.
    """
    N, T = C.shape
    use_raw = C_raw_fluorescence is not None and C_raw_fluorescence.shape == C.shape
    
    if use_raw:
        logger.debug("  Measuring transient amplitudes from raw fluorescence (local ΔF/F)")
    
    # Convert time windows to frames
    baseline_frames = max(1, int(baseline_window_s * frame_rate))
    offset_frames = max(1, int(baseline_offset_s * frame_rate))
    peak_frames = max(1, int(peak_window_s * frame_rate))
    
    neuron_amplitudes = []
    
    for j in range(N):
        trace_for_amp = C_raw_fluorescence[j] if use_raw else C[j]
        spikes = S[j]
        
        spike_frames = np.where(spikes > 0)[0]
        if len(spike_frames) == 0:
            continue
        
        transient_amps = []
        
        for t_spike in spike_frames:
            bl_end = t_spike - offset_frames
            bl_start = bl_end - baseline_frames
            pk_start = t_spike
            pk_end = min(T, t_spike + peak_frames)
            
            if bl_start < 0 or pk_end <= pk_start:
                continue
            
            baseline = np.median(trace_for_amp[bl_start:bl_end])
            peak_val = np.max(trace_for_amp[pk_start:pk_end])
            
            if use_raw:
                # Local ΔF/F: (peak - baseline) / baseline
                if baseline > 1e-6:
                    amp = (peak_val - baseline) / baseline
                else:
                    continue
            else:
                # Fallback: absolute difference on denoised trace
                amp = peak_val - baseline
            
            if amp > 0:
                transient_amps.append(amp)
        
        if len(transient_amps) > 0:
            neuron_amplitudes.append(float(np.mean(transient_amps)))
    
    return neuron_amplitudes


# =============================================================================
# FEATURE EXTRACTION & CLUSTERING
# =============================================================================

def build_feature_matrix(datasets: List[DatasetMetrics]) -> Tuple[np.ndarray, List[str]]:
    """Build (n_datasets, n_features) matrix from dataset metrics."""
    names = [d.name for d in datasets]
    X = np.zeros((len(datasets), len(FEATURE_NAMES)))
    for i, ds in enumerate(datasets):
        for j, (attr, _) in enumerate(FEATURE_NAMES):
            val = _recording_metric(ds, attr)
            X[i, j] = val if val is not None else 0.0

    # Replace NaN/inf with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad]) if (~bad).any() else 0.0

    return X, names



# =============================================================================
# FIGURE GENERATION
# =============================================================================

def _extract_organoid_id(name: str) -> str:
    """Extract organoid identifier from dataset name.

    'D109_3-63_040226_R7 - Denoised' → 'D109'
    'D115_0-3_040226_R3 - Denoised'  → 'D115'

    Falls back to the full abbreviated name if no clear organoid prefix found.
    """
    s = name.replace(' - Denoised', '').replace(' - denoised', '').strip()
    parts = s.split('_')
    if parts and (parts[0].startswith('D') or parts[0].startswith('d')):
        return parts[0].upper()
    return _abbrev(name)



def _extract_genotype(name: str, genotype_map: dict = None) -> str:
    """Extract genotype from dataset name using a configurable prefix map.

    Naming convention: ``{day}_{line}_{date}_{region} - Denoised``

    The line field (second underscore-delimited part) encodes genotype
    via a prefix before the hyphen.  The mapping from prefix to genotype
    label is defined by ``genotype_map``.

    Parameters
    ----------
    name : str
        Dataset name.
    genotype_map : dict, optional
        Maps line-prefix strings to genotype labels.  Any prefix not in the
        map is assigned to ``'default'`` if present, otherwise ``'Unknown'``.

        Example::

            {'3': 'Control', 'default': 'Mutant'}

        Default (if None): ``{'3': 'Control', 'default': 'Mutant'}``.

    Examples
    --------
    >>> _extract_genotype('D109_3-63_040226_R7 - Denoised')
    'Control'
    >>> _extract_genotype('D109_1-12_040226_R3 - Denoised')
    'Mutant'

    Returns 'Unknown' if the line field cannot be parsed.
    """
    if genotype_map is None:
        genotype_map = {'3': 'Control', 'default': 'Mutant'}

    s = name.replace(' - Denoised', '').replace(' - denoised', '').strip()
    parts = s.split('_')

    if len(parts) < 2:
        return 'Unknown'

    line_field = parts[1]  # e.g. '3-63', '1-12', '0-3'

    # The genotype prefix is the digit(s) before the hyphen
    if '-' in line_field:
        prefix = line_field.split('-')[0].strip()
    else:
        prefix = line_field.strip()

    if prefix in genotype_map:
        return genotype_map[prefix]
    elif 'default' in genotype_map:
        return genotype_map['default']
    else:
        return 'Unknown'


def _extract_line_id(name: str) -> str:
    """Extract the full line identifier from dataset name.

    'D109_3-63_040226_R7 - Denoised' -> '3-63'
    """
    s = name.replace(' - Denoised', '').replace(' - denoised', '').strip()
    parts = s.split('_')
    return parts[1] if len(parts) >= 2 else 'unknown'


# =============================================================================
# Z-SCORE NORMALISATION FOR CROSS-DATASET COMPARISON
# =============================================================================

def _zscore_within_dataset(values: np.ndarray) -> np.ndarray:
    """Z-score normalise values within a single dataset.

    Parameters
    ----------
    values : array (n_neurons,)
        Per-neuron metric values from one dataset.

    Returns
    -------
    z : array (n_neurons,)
        Z-scored values.  Returns zeros if std ~ 0.
    """
    mu = np.mean(values)
    sigma = np.std(values, ddof=1) if len(values) > 1 else 0.0
    if sigma < 1e-10:
        return np.zeros_like(values)
    return (values - mu) / sigma


def _flag_suspicious_neurons(spike_train: np.ndarray, dur_s: float,
                             z_thresh: float = 3.0):
    """Flag neurons with anomalously high amplitude or frequency within a dataset.

    Uses MAD-based modified Z-scores so that one or two extreme neurons
    don't inflate the scale and mask themselves.

    A threshold of 3.0 (modified Z-score) corresponds roughly to the top
    ~1% in a normal distribution.  This is intentionally conservative —
    we want to flag only genuinely suspicious neurons (neuropil, vessels,
    merged cells) rather than normal biological variation.

    Parameters
    ----------
    spike_train : array (n_neurons, T)
        Spike trains for the selected neurons.
    dur_s : float
        Recording duration in seconds.
    z_thresh : float
        Modified Z-score threshold for flagging (default 3.0).

    Returns
    -------
    dict with:
        'flagged'     : bool array (n_neurons,) — True = suspicious
        'amp_z'       : float array — per-neuron amplitude Z-scores
        'freq_z'      : float array — per-neuron frequency Z-scores
        'n_flagged'   : int
        'rates'       : float array — spike rates (events/10s)
        'amps'        : float array — mean spike amplitudes
    """
    n = spike_train.shape[0]
    rates = np.zeros(n)
    amps = np.zeros(n)

    for j in range(n):
        spk = spike_train[j]
        spike_frames = spk[spk > 0]
        rates[j] = len(spike_frames) / dur_s * 10.0 if dur_s > 0 else 0
        amps[j] = float(np.mean(spike_frames)) if len(spike_frames) > 0 else 0.0

    def _mad_z(arr):
        """Modified Z-score using median absolute deviation."""
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad < 1e-10:
            return np.zeros_like(arr)
        return 0.6745 * (arr - med) / mad

    amp_z = _mad_z(amps)
    freq_z = _mad_z(rates)

    # Flag neurons that are HIGH outliers on EITHER metric
    # Only flag high side — low amplitude / frequency is just quiet, not suspicious
    flagged = (amp_z > z_thresh) | (freq_z > z_thresh)

    return {
        'flagged': flagged,
        'amp_z': amp_z,
        'freq_z': freq_z,
        'n_flagged': int(flagged.sum()),
        'rates': rates,
        'amps': amps,
    }


def run_analysis(
    results_dir: str,
    output_dir: str,
    frame_rate_override: Optional[float] = None,
    motion_max_threshold: float = 15.0,
    motion_residual_threshold: float = 2.0,
    drift_threshold: float = 1.0,
    inactive_file: Optional[str] = None,
    min_roi_distance: float = 15.0,
    roi_peak_figures: bool = False,
    mutant_label: str = 'CEP41 R242H',
) -> Dict[str, Any]:
    """
    Run full analysis on batch results.

    Parameters
    ----------
    motion_max_threshold : float
        Datasets with max shift above this (in pixels) are excluded.
    motion_residual_threshold : float
        Datasets with residual jitter std above this are excluded.
    drift_threshold : float
        Datasets with population-median baseline drift ratio above this
        are excluded.  Drift ratio = |mean(Q4) - mean(Q1)| / std(trace),
        measured on raw fluorescence of selected neurons.  Default 1.0.
    inactive_file : str, optional
        Path to a text file listing dataset names (one per line) that were
        visually confirmed to have no activity.  These datasets are kept
        in the results but marked as inactive — all spikes are zeroed,
        active fraction set to 0, and they appear in figures with a
        distinct annotation.  Lines starting with # are ignored.

    Returns dict with datasets, features, and analysis results.
    """

    # =====================================================================
    # FIGURE GENERATION (imported from separate modules)
    # =====================================================================

    from .figures_overview import (
        generate_figures,
        fig_neuron_selection,
        fig_flagged_neurons,
        fig_n_selected_distribution,
        fig_quality_gating,
        fig_selected_traces,
    )
    from .figures_genotype import (
        run_genotype_comparison,
        run_activity_analysis,
        run_statistical_tests,
        run_dataset_overview,
        run_between_organoid_tests,
        generate_roi_peak_figures,
    )
    
    os.makedirs(output_dir, exist_ok=True)
    results_path = Path(results_dir)

    # ── Load inactive dataset list ───────────────────────────────────────
    inactive_names = set()
    if inactive_file and os.path.isfile(inactive_file):
        with open(inactive_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    inactive_names.add(line)
        logger.info(f"Loaded {len(inactive_names)} inactive datasets from {inactive_file}")
    elif inactive_file:
        logger.warning(f"Inactive file not found: {inactive_file}")

    # ── Load all datasets ────────────────────────────────────────────────
    logger.info(f"Loading datasets from {results_dir}")
    all_datasets = []
    for subdir in sorted(results_path.iterdir()):
        if not subdir.is_dir():
            continue
        if not (subdir / 'temporal_traces.npy').exists():
            continue
        ds = load_dataset_metrics(
            str(subdir), subdir.name,
            frame_rate_override=frame_rate_override,
            min_roi_distance=min_roi_distance,
        )
        if ds is not None:
            all_datasets.append(ds)

    if len(all_datasets) == 0:
        logger.error("No datasets loaded")
        return {}

    logger.info(f"\nLoaded {len(all_datasets)} datasets")

    # ── Mark manually inactive datasets ──────────────────────────────────
    # Datasets visually confirmed to have no activity: zero out spikes,
    # set active fraction to 0, but keep them in the analysis for
    # active fraction and demographic reporting.
    n_marked_inactive = 0
    for ds in all_datasets:
        # Match by exact name, or bidirectional substring (handles cases where
        # folder name is shorter or longer than the inactive list entry)
        is_inactive = (ds.name in inactive_names or
                       any(iname in ds.name or ds.name in iname 
                           for iname in inactive_names))
        if is_inactive:
            ds.n_active = 0
            ds.active_fraction = 0.0
            if ds.neuron_is_active is not None:
                ds.neuron_is_active[:] = False
            if ds.neuron_spike_rates is not None:
                ds.neuron_spike_rates[:] = 0.0
            if ds.neuron_spike_amplitudes is not None:
                ds.neuron_spike_amplitudes[:] = 0.0
            ds.mean_spike_rate = 0.0
            ds.median_spike_rate = 0.0
            ds.mean_spike_amplitude = 0.0
            ds.manually_inactive = True
            n_marked_inactive += 1
            logger.info(f"  Marked as inactive (no visible activity): {ds.name}")
        else:
            ds.manually_inactive = False

    if n_marked_inactive > 0:
        logger.info(f"  Total manually inactive: {n_marked_inactive}/{len(all_datasets)}")

    # ── Quality gating (motion + baseline drift) ──────────────────────────
    logger.info(f"\nQuality gating (max_shift<={motion_max_threshold}px, "
                f"residual_std<={motion_residual_threshold}px, "
                f"baseline_drift<={drift_threshold}):")

    datasets = []
    excluded = []
    for ds in all_datasets:
        reasons = []
        if ds.motion_max_shift > motion_max_threshold:
            reasons.append(f"max_shift={ds.motion_max_shift:.1f}px")
        if ds.motion_residual_std > motion_residual_threshold:
            reasons.append(f"residual_std={ds.motion_residual_std:.2f}px")
        if ds.baseline_drift > drift_threshold:
            reasons.append(f"baseline_drift={ds.baseline_drift:.2f}")
            ds.baseline_drift_excluded = True

        if reasons:
            ds.motion_excluded = True
            excluded.append(ds)
            logger.warning(f"  EXCLUDED {ds.name}: {', '.join(reasons)}")
        else:
            datasets.append(ds)
            logger.info(f"  OK {ds.name}: shift={ds.motion_max_shift:.1f}px, "
                        f"residual={ds.motion_residual_std:.2f}px, "
                        f"drift={ds.baseline_drift:.2f}")

    logger.info(f"\n{len(datasets)} included, {len(excluded)} excluded by quality gating")

    # Save exclusion report
    quality_report = {
        'threshold_max_shift': motion_max_threshold,
        'threshold_residual_std': motion_residual_threshold,
        'threshold_baseline_drift': drift_threshold,
        'included': [d.name for d in datasets],
        'excluded': [
            {'name': d.name, 'max_shift': d.motion_max_shift,
             'residual_std': d.motion_residual_std,
             'baseline_drift': d.baseline_drift}
            for d in excluded
        ],
    }
    
    # ── Create organized output directory structure ────────────────────────
    # analysis/
    # ├── figures/
    # │   ├── 1 - Main Results/        UMAP, heatmap, between-organoid
    # │   ├── 1b - Metrics/            Individual metric plots
    # │   ├── Correlation Graphs/      Correlation matrices  
    # │   ├── Full Overview/           Population activity, quality, neuron selection
    # │   ├── Results by Dataset/      Per-dataset folders
    # │   └── Temporal Visualisations/ Rasters, trace comparisons
    # └── data/
    #     ├── analysis_results.json
    #     ├── dataset_features.csv
    #     └── quality_gating.json
    
    fig_dir = os.path.join(output_dir, 'figures')
    data_dir = os.path.join(output_dir, 'data')
    
    # New directory structure (v2.0: added genotype comparison)
    main_results_dir = os.path.join(fig_dir, '1 - Main Results')
    metrics_dir = os.path.join(fig_dir, '1b - Metrics')
    genotype_dir = os.path.join(fig_dir, '2 - Genotype Comparison')
    correlation_dir = os.path.join(fig_dir, 'Correlation Graphs')
    overview_dir = os.path.join(fig_dir, 'Full Overview')
    per_dataset_dir = os.path.join(fig_dir, 'Results by Dataset')
    temporal_dir = os.path.join(fig_dir, 'Temporal Visualisations')
    
    for d in [fig_dir, data_dir, main_results_dir, metrics_dir, genotype_dir,
              correlation_dir, overview_dir, per_dataset_dir, temporal_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Save quality gating JSON to data/
    with open(os.path.join(data_dir, 'quality_gating.json'), 'w') as f:
        json.dump(quality_report, f, indent=2)

    # ── Generate motion quality figure ───────────────────────────────────
    fig_quality_gating(all_datasets, motion_max_threshold,
                        motion_residual_threshold, drift_threshold, overview_dir)

    if len(datasets) < 3:
        logger.error(f"Need at least 3 datasets after motion exclusion, "
                     f"got {len(datasets)}")
        return {}



    # ── Selected trace figures ───────────────────────────────────────────
    fig_selected_traces(datasets, output_dir)

    # ── ROI peak frame figures (optional, slow) ──────────────────────────
    if roi_peak_figures:
        try:
            generate_roi_peak_figures(datasets, output_dir)
        except Exception as e:
            logger.warning(f"ROI peak figures failed: {e}")
    else:
        logger.info("ROI peak frame figures disabled")

    # ── Neuron selection transparency ────────────────────────────────────
    fig_neuron_selection(datasets, fig_dir, per_dataset_dir=per_dataset_dir)

    # ── n_selected distribution ──────────────────────────────────────────
    try:
        fig_n_selected_distribution(datasets, output_dir, mutant_label=mutant_label)
    except Exception as e:
        logger.warning(f"n_selected distribution figure failed: {e}")

    # ── Build feature matrix ─────────────────────────────────────────────
    X, names = build_feature_matrix(datasets)
    feat_labels = [fl for _, fl in FEATURE_NAMES]

    # ── Save feature CSV ─────────────────────────────────────────────────
    csv_path = os.path.join(data_dir, 'dataset_features.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset'] + feat_labels)
        for i, name in enumerate(names):
            writer.writerow([name] + [f'{X[i, j]:.4f}' for j in range(X.shape[1])])
    logger.info(f"Saved feature matrix: {csv_path}")

    # ── Save per-ROI listing ─────────────────────────────────────────────
    roi_csv_path = os.path.join(data_dir, 'selected_rois.csv')
    with open(roi_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset', 'roi_index', 'quality_score', 'is_active',
            'n_spikes', 'spike_rate_per_10s', 'mean_amplitude',
            'genotype', 'organoid_day', 'manually_inactive',
        ])
        for ds in datasets:
            geno = _extract_genotype(ds.name)
            day = _extract_organoid_id(ds.name)
            n_sel = ds.n_selected
            for j in range(n_sel):
                roi_idx = int(ds.selected_roi_indices[j]) if ds.selected_roi_indices is not None and j < len(ds.selected_roi_indices) else j
                q = float(ds.selected_quality[j]) if ds.selected_quality is not None and j < len(ds.selected_quality) else 0.0
                is_active = bool(ds.neuron_is_active[j]) if ds.neuron_is_active is not None and j < len(ds.neuron_is_active) else False
                rate = float(ds.neuron_spike_rates[j]) if ds.neuron_spike_rates is not None and j < len(ds.neuron_spike_rates) else 0.0
                amp = float(ds.neuron_spike_amplitudes[j]) if ds.neuron_spike_amplitudes is not None and j < len(ds.neuron_spike_amplitudes) else 0.0
                n_spk = int(round(rate * ds.duration_seconds / 10.0)) if ds.duration_seconds > 0 else 0
                writer.writerow([
                    ds.name, roi_idx, f'{q:.3f}', is_active,
                    n_spk, f'{rate:.2f}', f'{amp:.4f}',
                    geno, day, ds.manually_inactive,
                ])
    logger.info(f"Saved selected ROI listing: {roi_csv_path}")

    # ── Save results JSON ────────────────────────────────────────────────
    results = {
        'n_datasets': len(datasets),
        'n_features': len(feat_labels),
        'motion_excluded': [d.name for d in excluded],
        'n_excluded': len(excluded),
        'neuron_selection': {
            'mode': 'deconv_pass + distance_dedup',
            'per_dataset': {
                d.name: {
                    'n_total': d.n_neurons,
                    'n_deconv_pass': d.n_confident,
                    'n_distance_removed': d.n_distance_removed,
                    'n_selected': d.n_selected,
                    'mean_snr': float(d.mean_quality_score),
                }
                for d in datasets
            },
        },
        'dataset_metrics': {
            d.name: {
                'mean_spike_rate': d.mean_spike_rate,
                'mean_spike_amplitude': d.mean_spike_amplitude,
                'pairwise_correlation': d.pairwise_correlation_mean,
                'synchrony_index': d.synchrony_index,
                'n_network_bursts': d.n_network_bursts,
                'burst_rate': d.burst_rate,
            }
            for d in datasets
        },
    }

    json_path = os.path.join(data_dir, 'analysis_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # ── Statistical tests ────────────────────────────────────────────────
    try:
        stat_results = run_statistical_tests(datasets, output_dir)
        results['statistical_tests'] = stat_results['tests']
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Statistical tests failed: {e}")
        import traceback; logger.error(traceback.format_exc())

    # ── Dataset overview ─────────────────────────────────────────────────
    try:
        overview_results = run_dataset_overview(datasets, output_dir)
        results['dataset_overview'] = overview_results
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Dataset overview failed: {e}")
        import traceback; logger.error(traceback.format_exc())

    # ── Between-organoid comparison ──────────────────────────────────────
    try:
        between_results = run_between_organoid_tests(datasets, output_dir)
        results['between_organoid'] = between_results
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Between-organoid tests failed: {e}")
        import traceback; logger.error(traceback.format_exc())

    # ── Genotype comparison ──────────────────────────────────────────────
    try:
        genotype_results = run_genotype_comparison(datasets, output_dir,
                                                    mutant_label=mutant_label)
        results['genotype_comparison'] = genotype_results
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Genotype comparison failed: {e}")
        import traceback; logger.error(traceback.format_exc())

    # ── Activity analysis ────────────────────────────────────────────────
    try:
        activity_results = run_activity_analysis(datasets, output_dir,
                                                  mutant_label=mutant_label)
        results['activity_analysis'] = activity_results
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Activity analysis failed: {e}")
        import traceback; logger.error(traceback.format_exc())

    # ── Core overview figures ────────────────────────────────────────────
    generate_figures(datasets, X, feat_labels, names, fig_dir)

    # ── Flagged neuron inspection ────────────────────────────────────────
    try:
        flagged_dir = os.path.join(fig_dir, 'per_dataset')
        os.makedirs(flagged_dir, exist_ok=True)
        fig_flagged_neurons(datasets, flagged_dir)
    except Exception as e:
        logger.error(f"Flagged neuron inspection failed: {e}")
        import traceback; logger.error(traceback.format_exc())

    return results


# =============================================================================
# CLI
# =============================================================================
if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    parser = argparse.ArgumentParser(
        description='Calcium Pipeline v2.1 — Unsupervised Dataset Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Compares datasets side by side based on their functional properties
(spike rates, amplitudes, correlations, synchrony) using the top-quality
neurons from each recording.

Example:
    python -m src.group_analysis \\
        --results-dir /path/to/batch_results \\
        --output /path/to/analysis
        """,
    )
    parser.add_argument('--results-dir', required=True,
                        help='Directory with per-dataset pipeline outputs')
    parser.add_argument('--output', required=True,
                        help='Output directory for analysis results')
    parser.add_argument('--config', default=None,
                        help='YAML configuration file (defaults to $PIPELINE_DIR/config/default.yaml)')

    # Optional CLI overrides — default to None so they don't clobber YAML.
    parser.add_argument('--frame-rate', type=float, default=None)
    parser.add_argument('--drift-threshold', type=float, default=None)
    parser.add_argument('--motion-max-threshold', type=float, default=None)
    parser.add_argument('--motion-residual-threshold', type=float, default=None)
    parser.add_argument('--min-roi-distance', type=float, default=None)
    parser.add_argument('--inactive-file', type=str, default=None,
                        help='Text file listing inactive datasets (one per line)')

    args = parser.parse_args()

    # ── Resolve config path ────────────────────────────────────────────
    import os
    pipeline_dir = os.environ.get(
        'PIPELINE_DIR',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    config_path = args.config or os.path.join(pipeline_dir, 'config', 'default.yaml')

    # ── Load YAML config with CLI overrides ────────────────────────────
    try:
        from .config_loader import load_config
    except ImportError:
        from config_loader import load_config

    overrides = {
        'imaging.frame_rate':              args.frame_rate,
        'quality.drift_threshold':         args.drift_threshold,
        'quality.motion_max_threshold':    args.motion_max_threshold,
        'quality.motion_residual_threshold': args.motion_residual_threshold,
        'quality.min_roi_distance':        args.min_roi_distance,
        'quality.inactive_file':           args.inactive_file,
    }
    cfg = load_config(config_path, overrides=overrides)

    results = run_analysis(
        results_dir=args.results_dir,
        output_dir=args.output,
        frame_rate_override=cfg.imaging.frame_rate,
        motion_max_threshold=cfg.quality.motion_max_threshold,
        motion_residual_threshold=cfg.quality.motion_residual_threshold,
        drift_threshold=cfg.quality.drift_threshold,
        inactive_file=cfg.quality.inactive_file,
        min_roi_distance=cfg.quality.min_roi_distance,
        roi_peak_figures=cfg.quality.roi_peak_figures,
        mutant_label=cfg.quality.mutant_label,
    )

    if results:
        print(f"\nAnalysis complete. Results in {args.output}")
        print(f"  Datasets: {results['n_datasets']}")
    else:
        print("Analysis failed — check logs")
