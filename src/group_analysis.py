"""
Dataset Comparison & Quality Analysis Module
=============================================

*Everything was built on top as needed during development, needs heavy refractoring

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

def _fmt_p(p):
    """Format p-value for display: avoid showing p=0.000."""
    if p < 0.0001:
        return f'p={p:.1e}'
    elif p < 0.001:
        return f'p={p:.4f}'
    else:
        return f'p={p:.3f}'


def _sig_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'


def _draw_sig_bracket(ax, x1, x2, y, h, text, fontsize=7, color='#333333'):
    """Draw a significance bracket between two x positions."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color,
            linewidth=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom',
            fontsize=fontsize, color=color, fontweight='bold')


def _fig_spike_rate_by_organoid(datasets, organoid_ids, unique_organoids,
                                 org_colors, output_dir):
    """Spike rate by organoid dot plot with per-day grouping."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from collections import OrderedDict
    import re

    def _extract_day_rec(name):
        date_match = re.search(r'(\d{6})', name)
        rec_match = re.search(r'(R\d+)', name)
        day = date_match.group(1) if date_match else 'unknown'
        rec = rec_match.group(1) if rec_match else ''
        return day, rec

    rng = np.random.default_rng(42)
    org_day_rates = OrderedDict()
    for oid in unique_organoids:
        org_day_rates[oid] = OrderedDict()

    for ds, oid in zip(datasets, organoid_ids):
        rates = _get_neuron_rates(ds)
        if len(rates) == 0:
            continue
        rec_mean = float(np.mean(rates))
        day, rec = _extract_day_rec(ds.name)
        if day not in org_day_rates[oid]:
            org_day_rates[oid][day] = []
        org_day_rates[oid][day].append((rec, rec_mean))

    total_days = sum(len(days) for days in org_day_rates.values())
    if total_days == 0:
        logger.warning("  No spike rate data — skipping spike rate by organoid figure")
        return

    n_org = len(unique_organoids)
    fig_width = max(18, total_days * 1.1 + n_org * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x_pos = 0
    x_ticks = []
    x_labels = []
    all_plotted_vals = []

    for oid in unique_organoids:
        days = org_day_rates[oid]
        for day_idx, (day, recordings) in enumerate(sorted(days.items())):
            if not recordings:
                continue
            day_vals = np.array([r[1] for r in recordings])
            all_plotted_vals.extend(day_vals.tolist())
            jitter = rng.uniform(-0.15, 0.15, len(day_vals))
            ax.scatter(x_pos + jitter, day_vals, c=org_colors[oid], s=55,
                      alpha=0.75, edgecolor='white', linewidth=0.6, zorder=5)
            if len(day_vals) > 1:
                ax.scatter(x_pos, np.mean(day_vals), c=org_colors[oid], s=150,
                          marker='_', linewidths=3.5, zorder=6)
            day_formatted = f'{day[2:4]}/{day[0:2]}' if len(day) == 6 else day
            x_ticks.append(x_pos)
            x_labels.append(day_formatted)
            x_pos += 1
        x_pos += 0.3

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha='right')
    ax.set_xlim(-0.8, x_pos - 0.3)

    if all_plotted_vals:
        y_floor = max(0.0, float(np.percentile(all_plotted_vals, 1)) - 0.02)
        y_ceil = float(np.percentile(all_plotted_vals, 99)) * 1.15
        ax.set_ylim(bottom=y_floor, top=max(y_ceil, 0.1))

    ax.set_ylabel('Event rate (events/10s)', fontsize=12)
    ax.set_xlabel('Recording Date (DD/MM)', fontsize=11)
    ax.set_title('Event rate by organoid  (per-recording averages)',
                 fontsize=14, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    if all_plotted_vals:
        global_mean = np.mean(all_plotted_vals)
        ax.axhline(global_mean, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(ax.get_xlim()[1], global_mean, f' mean={global_mean:.2f}',
               va='center', ha='left', fontsize=9, color='#666666')

    legend_elements = [Patch(facecolor=org_colors[oid], alpha=0.7, label=oid)
                      for oid in unique_organoids]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              title='Organoid', title_fontsize=11, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'figures', '1 - Main Results', 'spike_rate_by_organoid.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Event rate by organoid saved: {path}")


def _fig_correlation_by_organoid(datasets, organoid_ids, unique_organoids,
                                  org_colors, output_dir):
    """Pairwise correlation by organoid dot plot with per-day grouping."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from collections import OrderedDict
    import re

    def _extract_day_rec(name):
        date_match = re.search(r'(\d{6})', name)
        rec_match = re.search(r'(R\d+)', name)
        day = date_match.group(1) if date_match else 'unknown'
        rec = rec_match.group(1) if rec_match else ''
        return day, rec

    rng = np.random.default_rng(42)
    org_day_corr = OrderedDict()
    for oid in unique_organoids:
        org_day_corr[oid] = OrderedDict()

    all_corr_vals = []
    for ds, oid in zip(datasets, organoid_ids):
        corr_val = ds.pairwise_correlation_mean
        if not np.isfinite(corr_val):
            continue
        all_corr_vals.append(corr_val)
        day, rec = _extract_day_rec(ds.name)
        if day not in org_day_corr[oid]:
            org_day_corr[oid][day] = []
        org_day_corr[oid][day].append((rec, corr_val))

    total_days = sum(len(days) for days in org_day_corr.values())
    if total_days == 0:
        logger.warning("  No valid correlation data — skipping correlation by organoid figure")
        return

    n_org = len(unique_organoids)
    fig, ax = plt.subplots(figsize=(max(10, total_days * 0.6 + n_org * 0.8), 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x_pos = 0
    x_ticks = []
    x_labels = []

    for oid in unique_organoids:
        days = org_day_corr[oid]
        for day, recordings in sorted(days.items()):
            vals = [r[1] for r in recordings]
            day_formatted = f'{day[2:4]}/{day[0:2]}' if len(day) == 6 else day
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(x_pos + jitter, vals, c=org_colors[oid], s=60, alpha=0.7,
                      edgecolor='white', linewidth=0.5, zorder=5)
            if len(vals) > 0:
                ax.scatter(x_pos, np.mean(vals), c=org_colors[oid], s=120,
                          marker='_', linewidths=3, zorder=6)
            x_ticks.append(x_pos)
            x_labels.append(day_formatted)
            x_pos += 1
        x_pos += 0.3

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha='right')
    ax.set_xlim(-0.8, x_pos - 0.3)
    ax.set_ylabel('Pairwise Correlation (r)', fontsize=11)
    ax.set_xlabel('Recording Date (DD/MM)', fontsize=10)
    ax.set_title('Pairwise Correlation by Organoid', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    if all_corr_vals:
        mn_c = float(np.mean(all_corr_vals))
        ax.axhline(mn_c, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(ax.get_xlim()[1], mn_c, f' mean={mn_c:.3f}',
               va='center', ha='left', fontsize=8, color='#666666')

    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=org_colors[oid],
                              markersize=10, alpha=0.7, label=oid)
                      for oid in unique_organoids]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              title='Organoid', title_fontsize=10, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'figures', '1 - Main Results', 'correlation_by_organoid.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Correlation by organoid saved: {path}")


def _fig_synchrony_by_organoid(datasets, organoid_ids, unique_organoids,
                                org_colors, output_dir):
    """Synchrony index by organoid dot plot with per-day grouping."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from collections import OrderedDict
    import re

    def _extract_day_rec(name):
        date_match = re.search(r'(\d{6})', name)
        rec_match = re.search(r'(R\d+)', name)
        day = date_match.group(1) if date_match else 'unknown'
        rec = rec_match.group(1) if rec_match else ''
        return day, rec

    rng = np.random.default_rng(42)
    org_day_sync = OrderedDict()
    for oid in unique_organoids:
        org_day_sync[oid] = OrderedDict()

    all_sync_vals = []
    for ds, oid in zip(datasets, organoid_ids):
        sync_val = ds.synchrony_index
        if not np.isfinite(sync_val):
            continue
        all_sync_vals.append(sync_val)
        day, rec = _extract_day_rec(ds.name)
        if day not in org_day_sync[oid]:
            org_day_sync[oid][day] = []
        org_day_sync[oid][day].append((rec, sync_val))

    total_days = sum(len(days) for days in org_day_sync.values())
    if total_days == 0:
        logger.warning("  No valid synchrony data — skipping synchrony by organoid figure")
        return

    n_org = len(unique_organoids)
    fig, ax = plt.subplots(figsize=(max(10, total_days * 0.6 + n_org * 0.8), 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x_pos = 0
    x_ticks = []
    x_labels = []

    for oid in unique_organoids:
        days = org_day_sync[oid]
        for day, recordings in sorted(days.items()):
            vals = [r[1] for r in recordings]
            day_formatted = f'{day[2:4]}/{day[0:2]}' if len(day) == 6 else day
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(x_pos + jitter, vals, c=org_colors[oid], s=60, alpha=0.7,
                      edgecolor='white', linewidth=0.5, zorder=5)
            if len(vals) > 0:
                ax.scatter(x_pos, np.mean(vals), c=org_colors[oid], s=120,
                          marker='_', linewidths=3, zorder=6)
            x_ticks.append(x_pos)
            x_labels.append(day_formatted)
            x_pos += 1
        x_pos += 0.3

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha='right')
    ax.set_xlim(-0.8, x_pos - 0.3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Synchrony Index', fontsize=11)
    ax.set_xlabel('Recording Date (DD/MM)', fontsize=10)
    ax.set_title('Synchrony Index by Organoid', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    if all_sync_vals:
        mn_s = float(np.mean(all_sync_vals))
        ax.axhline(mn_s, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(ax.get_xlim()[1], mn_s, f' mean={mn_s:.3f}',
               va='center', ha='left', fontsize=8, color='#666666')

    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=org_colors[oid],
                              markersize=10, alpha=0.7, label=oid)
                      for oid in unique_organoids]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              title='Organoid', title_fontsize=10, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'figures', '1 - Main Results', 'synchrony_by_organoid.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Synchrony by organoid saved: {path}")


def run_statistical_tests(datasets: List[DatasetMetrics], output_dir: str) -> dict:
    """
    Statistical comparison across datasets.

    Produces three by-organoid figures (spike rate, correlation, synchrony)
    and computes Kruskal-Wallis + pairwise Mann-Whitney U tests on spike rates.
    """
    from scipy import stats as sp_stats
    from collections import OrderedDict

    n_ds = len(datasets)
    results = {
        'n_datasets': n_ds,
        'data_sources': {
            'spike_rate': 'Deconvolved spike trains (S > 0 events, OASIS output)',
            'pairwise_correlation': 'Denoised calcium traces (Pearson r, OASIS output)',
            'synchrony_index': 'Denoised calcium traces (population coupling)',
            'note': 'All metrics computed from top-N quality-selected neurons per dataset',
        },
        'tests': {},
    }

    # ── Per-neuron spike rates by dataset ────────────────────────────────
    per_ds_rates = []
    ds_names = []
    for ds in datasets:
        rates = _get_neuron_rates(ds)
        if len(rates) == 0:
            continue
        per_ds_rates.append(rates)
        ds_names.append(_abbrev(ds.name))

    per_ds_rate_means = np.array([float(np.mean(r)) for r in per_ds_rates]) if per_ds_rates else np.array([])

    # ── Kruskal-Wallis ───────────────────────────────────────────────────
    kw_result = None
    if len(per_ds_rates) >= 2 and all(len(r) >= 2 for r in per_ds_rates):
        H, p_kw = sp_stats.kruskal(*per_ds_rates)
        kw_result = {'H': float(H), 'p': float(p_kw), 'n_groups': len(per_ds_rates)}

    # ── Pairwise Mann-Whitney U ──────────────────────────────────────────
    n_pairs = len(per_ds_rates) * (len(per_ds_rates) - 1) // 2
    pairwise = []
    if n_pairs > 0:
        for i in range(len(per_ds_rates)):
            for j in range(i + 1, len(per_ds_rates)):
                U, p_raw = sp_stats.mannwhitneyu(
                    per_ds_rates[i], per_ds_rates[j], alternative='two-sided')
                p_bonf = min(p_raw * n_pairs, 1.0)
                pairwise.append({
                    'i': i, 'j': j,
                    'name_i': ds_names[i], 'name_j': ds_names[j],
                    'U': float(U), 'p_raw': float(p_raw), 'p_bonf': float(p_bonf),
                })

    results['tests'] = {
        'kruskal_wallis': kw_result,
        'pairwise_mannwhitney': pairwise,
        'spike_rate_summary': {
            'n_recordings': len(per_ds_rate_means),
            'mean': float(np.mean(per_ds_rate_means)) if len(per_ds_rate_means) > 0 else 0,
            'median': float(np.median(per_ds_rate_means)) if len(per_ds_rate_means) > 0 else 0,
            'sd': float(np.std(per_ds_rate_means, ddof=1)) if len(per_ds_rate_means) > 1 else 0,
        },
    }

    # Collect correlation and synchrony summary stats
    corr_vals = np.array([ds.pairwise_correlation_mean for ds in datasets])
    sync_vals = np.array([ds.synchrony_index for ds in datasets])
    corr_vals_valid = corr_vals[np.isfinite(corr_vals)]
    sync_vals_valid = sync_vals[np.isfinite(sync_vals)]

    for key, vals, label in [
        ('pairwise_correlation', corr_vals_valid, 'Pairwise Correlation (r)'),
        ('synchrony_index', sync_vals_valid, 'Synchrony Index'),
    ]:
        if len(vals) > 0:
            results['tests'][key] = {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0,
                'n': int(len(vals)),
            }

    # ── Organoid grouping (shared by all three figures) ──────────────────
    organoid_ids = [_extract_organoid_id(ds.name) for ds in datasets]
    unique_organoids = list(OrderedDict.fromkeys(organoid_ids))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    org_colors = {oid: palette[i % len(palette)] for i, oid in enumerate(unique_organoids)}

    # ── Generate figures (each in its own try/except) ────────────────────
    try:
        _fig_spike_rate_by_organoid(datasets, organoid_ids, unique_organoids,
                                     org_colors, output_dir)
    except Exception as e:
        logger.warning(f"  Spike rate by organoid figure failed: {e}")

    try:
        _fig_correlation_by_organoid(datasets, organoid_ids, unique_organoids,
                                      org_colors, output_dir)
    except Exception as e:
        logger.warning(f"  Correlation by organoid figure failed: {e}")

    try:
        _fig_synchrony_by_organoid(datasets, organoid_ids, unique_organoids,
                                    org_colors, output_dir)
    except Exception as e:
        logger.warning(f"  Synchrony by organoid figure failed: {e}")

    logger.info(f"  Statistical analysis complete")
    return results



def run_dataset_overview(datasets: List[DatasetMetrics], output_dir: str) -> dict:
    """
    Generate publication-quality visualizations for many-dataset comparisons.
    
    Produces separate figures:
    - dataset_umap.png — Clean UMAP projection (publication style)
    - dataset_heatmap.png — Clustered heatmap of standardized features  
    - metric_*.png — Individual metric plots grouped by organoid and day
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as path_effects
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    from collections import OrderedDict
    import re
    
    n_ds = len(datasets)
    results = {'n_datasets': n_ds, 'visualizations': []}
    
    # Create organized subdirectory structure
    summary_dir = os.path.join(output_dir, 'figures', '1 - Main Results')
    by_metric_dir = os.path.join(output_dir, 'figures', '1b - Metrics')
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(by_metric_dir, exist_ok=True)
    
    # ── Extract organoid IDs and build color map ──────────────────────────
    organoid_ids = [_extract_organoid_id(ds.name) for ds in datasets]
    unique_organoids = list(OrderedDict.fromkeys(organoid_ids))
    n_org = len(unique_organoids)
    
    # Professional color palette (colorblind-friendly)
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    org_colors = {oid: palette[i % len(palette)] for i, oid in enumerate(unique_organoids)}
    ds_colors = [org_colors[oid] for oid in organoid_ids]
    
    # ── Build feature matrix ──────────────────────────────────────────────
    feature_attrs = [
        ('mean_spike_rate', 'Event Rate'),
        ('pairwise_correlation_mean', 'Correlation'),
        ('synchrony_index', 'Synchrony'),
        ('burst_rate', 'Burst Rate'),
        ('mean_burst_participation', 'Burst Participation'),
        ('cv_iei', 'IEI Variability'),
    ]
    
    X = np.zeros((n_ds, len(feature_attrs)))
    for i, ds in enumerate(datasets):
        for j, (attr, _) in enumerate(feature_attrs):
            val = _recording_metric(ds, attr)
            X[i, j] = val if val is not None else np.nan

    # Impute NaN with column median (not zero — zero clusters excluded
    # recordings at the origin, distorting the UMAP embedding)
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad]) if (~bad).any() else 0.0

    X_std = StandardScaler().fit_transform(X)
    feat_labels = [fl for _, fl in feature_attrs]
    ds_names = [_abbrev(ds.name) for ds in datasets]
    # Keep original names for genotype extraction — _abbrev loses the line field
    ds_names_orig = [ds.name for ds in datasets]
    
    rng = np.random.default_rng(42)
    
    # =====================================================================
    # UMAP — removed (unreliable with small dataset counts)
    # =====================================================================
    
    # =====================================================================
    # FIGURE: Feature Heatmap (clustered)
    # =====================================================================
    try:
        fig_height = max(12, n_ds * 0.18)
        fig = plt.figure(figsize=(14, fig_height))
        fig.patch.set_facecolor('white')
        
        # Create gridspec with proper spacing for dendrogram
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(1, 3, width_ratios=[0.2, 1, 0.05], wspace=0.05)
        
        # Hierarchical clustering
        if n_ds > 2:
            linkage = hierarchy.linkage(pdist(X_std, 'euclidean'), method='ward')
            dendro = hierarchy.dendrogram(linkage, no_plot=True)
            row_order = dendro['leaves']
        else:
            row_order = list(range(n_ds))
        
        # Dendrogram
        ax_dendro = fig.add_subplot(gs[0])
        if n_ds > 2:
            hierarchy.dendrogram(linkage, orientation='left', ax=ax_dendro,
                                leaf_rotation=0, leaf_font_size=1,
                                above_threshold_color='#888888',
                                color_threshold=0)
        ax_dendro.set_xticks([])
        ax_dendro.set_yticks([])
        for spine in ax_dendro.spines.values():
            spine.set_visible(False)
        
        # Heatmap
        ax_heat = fig.add_subplot(gs[1])
        X_ordered = X_std[row_order, :]
        names_ordered = [ds_names[i] for i in row_order]
        colors_ordered = [ds_colors[i] for i in row_order]
        orgs_ordered = [organoid_ids[i] for i in row_order]
        
        im = ax_heat.imshow(X_ordered, aspect='auto', cmap='RdBu_r',
                            vmin=-2.5, vmax=2.5)
        
        ax_heat.set_xticks(range(len(feat_labels)))
        ax_heat.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=10)
        ax_heat.set_yticks(range(n_ds))
        
        # Y-tick labels with organoid color coding
        ylabels = [f'{orgs_ordered[i]}' for i in range(n_ds)]
        ax_heat.set_yticklabels(ylabels, fontsize=6)
        for i, (label, color) in enumerate(zip(ax_heat.get_yticklabels(), colors_ordered)):
            label.set_color(color)
            label.set_fontweight('bold')
        
        ax_heat.set_title('Standardized Features (hierarchically clustered)', 
                          fontsize=12, fontweight='bold', pad=10)
        
        # Colorbar
        ax_cbar = fig.add_subplot(gs[2])
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('Z-score', fontsize=10)
        
        plt.tight_layout()
        heatmap_path = os.path.join(summary_dir, 'feature_heatmap.png')
        plt.savefig(heatmap_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        results['visualizations'].append(heatmap_path)
        logger.info(f"Heatmap saved: {heatmap_path}")
        
    except Exception as e:
        logger.warning(f"Heatmap failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
    
    # =====================================================================
    # FIGURES 3+: Individual metric plots grouped by organoid with day structure
    # =====================================================================
    
    metrics_to_plot = [
        ('mean_spike_rate', 'Event Rate', 'events/10s'),
        ('mean_spike_amplitude', 'Transient Amplitude', 'ΔF/F₀'),
        ('pairwise_correlation_mean', 'Pairwise Correlation', 'r'),
        ('synchrony_index', 'Synchrony Index', ''),
        ('burst_rate', 'Burst Rate', 'bursts/10s'),
    ]
    
    for attr, title, unit in metrics_to_plot:
        try:
            # Pool data by organoid (like between_organoid_comparison)
            org_data = OrderedDict()
            for oid in unique_organoids:
                org_data[oid] = []
            
            for ds, oid in zip(datasets, organoid_ids):
                val = _recording_metric(ds, attr)
                if val is not None and np.isfinite(val):
                    org_data[oid].append(val)
            
            # Simple figure - one box per organoid
            fig_width = max(8, n_org * 1.2)
            fig, ax = plt.subplots(figsize=(fig_width, 6))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            # Prepare arrays for box plot
            data_arrays = [np.array(org_data[oid]) for oid in unique_organoids]
            
            # Box plots
            bp = ax.boxplot(
                data_arrays, positions=range(n_org), widths=0.5,
                patch_artist=True, showfliers=False,
                medianprops=dict(color='white', linewidth=1.5),
                whiskerprops=dict(color='#555555', linewidth=0.8),
                capprops=dict(color='#555555', linewidth=0.8),
            )
            
            for i, (patch, oid) in enumerate(zip(bp['boxes'], unique_organoids)):
                patch.set_facecolor(org_colors[oid])
                patch.set_alpha(0.6)
                patch.set_edgecolor(org_colors[oid])
                patch.set_linewidth(1.5)
            
            # Overlay individual recordings
            for i, oid in enumerate(unique_organoids):
                vals = org_data[oid]
                if len(vals) > 0:
                    jitter = rng.uniform(-0.15, 0.15, len(vals))
                    ax.scatter(i + jitter, vals,
                              color=org_colors[oid], s=40, alpha=0.7,
                              edgecolor='white', linewidth=0.5, zorder=5)
            
            # X-axis with organoid labels
            ax.set_xticks(range(n_org))
            ax.set_xticklabels(unique_organoids, fontsize=10, fontweight='bold')
            ax.set_xlim(-0.6, n_org - 0.4)
            
            # Color the x-tick labels
            for i, (tick_label, oid) in enumerate(zip(ax.get_xticklabels(), unique_organoids)):
                tick_label.set_color(org_colors[oid])
            
            # Labels and styling
            ylabel = f'{title} ({unit})' if unit else title
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlabel('Organoid', fontsize=10)
            ax.set_title(f'{title} by Organoid', fontsize=13, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)
            
            # Global mean line
            all_vals = [v for vals in org_data.values() for v in vals]
            if all_vals:
                global_mean = np.mean(all_vals)
                ax.axhline(global_mean, color='#666666', linestyle='--', 
                          linewidth=1.5, alpha=0.7, zorder=1)
                ax.text(ax.get_xlim()[1] + 0.05, global_mean, f'mean={global_mean:.2f}',
                       va='center', ha='left', fontsize=9, color='#666666',
                       clip_on=False)
            
            plt.tight_layout()
            metric_path = os.path.join(by_metric_dir, f'metric_{attr.replace("_mean", "")}.png')
            plt.savefig(metric_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            results['visualizations'].append(metric_path)
            logger.info(f"Metric plot saved: {metric_path}")
            
        except Exception as e:
            logger.warning(f"Metric plot {attr} failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    
    return results



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


def run_genotype_comparison(datasets: List[DatasetMetrics], output_dir: str,
                            mutant_label: str = 'CEP41 R242H') -> dict:
    """
    Compare Control vs Mutant genotypes at two levels:

    1. **Within-day**: For each organoid day (e.g. D109), compare control
       (3_x) vs mutant (1_x) recordings from the same day. This controls
       for day-to-day variation in imaging conditions.

    2. **Global pooled**: Pool ALL control recordings vs ALL mutant recordings
       across all days. More statistical power but confounded by day effects.

    3. **Paired meta-analysis**: For each day that has BOTH genotypes, compute
       a within-day effect size (Cohen's d), then combine effect sizes across
       days with a fixed-effects meta-analysis.

    Metrics compared (all using z-scored values to remove imaging confounds):
    - Spike rate (per-neuron, from deconvolved spike trains)
    - Spike amplitude (per-neuron, from denoised trace transients)
    - Pairwise correlation (per-recording, from denoised traces)
    - Synchrony index (per-recording, from denoised traces)

    Statistical tests:
    - Mann-Whitney U (non-parametric, two-sided) with Bonferroni correction
    - Cohen's d effect sizes
    - Both raw and z-scored comparisons reported side-by-side

    Parameters
    ----------
    datasets : list of DatasetMetrics
        Quality-gated datasets.
    output_dir : str
        Base output directory; figures saved under figures/2 - Genotype Comparison/

    Returns
    -------
    dict : Full results including per-day breakdowns, pooled tests, and
           meta-analysis across days.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats as sp_stats
    from collections import OrderedDict

    results = {'level': 'genotype', 'tests': {}}

    geno_dir = os.path.join(output_dir, 'figures', '2 - Genotype Comparison')
    os.makedirs(geno_dir, exist_ok=True)

    # ── Parse genotypes ──────────────────────────────────────────────────
    genotypes = [_extract_genotype(ds.name) for ds in datasets]
    organoid_ids = [_extract_organoid_id(ds.name) for ds in datasets]
    line_ids = [_extract_line_id(ds.name) for ds in datasets]

    unique_geno = sorted(set(genotypes))
    n_ctrl = sum(1 for g in genotypes if g == 'Control')
    n_mut = sum(1 for g in genotypes if g == 'Mutant')
    n_unk = sum(1 for g in genotypes if g == 'Unknown')

    logger.info(f"Genotype comparison: {n_ctrl} Control, {n_mut} Mutant, "
                f"{n_unk} Unknown recordings")

    results['n_control'] = n_ctrl
    results['n_mutant'] = n_mut
    results['n_unknown'] = n_unk
    results['dataset_genotypes'] = {
        ds.name: {'genotype': g, 'organoid': o, 'line': l}
        for ds, g, o, l in zip(datasets, genotypes, organoid_ids, line_ids)
    }

    if n_ctrl < 1 or n_mut < 1:
        logger.warning("Need at least 1 Control and 1 Mutant for genotype comparison")
        results['skipped'] = True
        results['reason'] = f"Only {n_ctrl} Control and {n_mut} Mutant recordings"
        return results

    # Exclude unknowns from comparison
    geno_datasets = [(ds, g, o) for ds, g, o in zip(datasets, genotypes, organoid_ids)
                     if g in ('Control', 'Mutant')]

    # ── Helper: extract per-neuron spike rates from a dataset ────────────
    def _ds_spike_rates(ds):
        """Return spike rates for ACTIVE neurons only (rate > 0)."""
        if ds.neuron_spike_rates is not None:
            rates = ds.neuron_spike_rates
            return rates[rates > 0]
        if ds.selected_spikes is None:
            return np.array([])
        dur_s = ds.duration_seconds if ds.duration_seconds > 0 else 1.0
        rates = np.array([np.sum(ds.selected_spikes[j] > 0) / dur_s * 10.0
                         for j in range(ds.selected_spikes.shape[0])])
        return rates[rates > 0]

    def _ds_spike_amplitudes(ds):
        """Return amplitudes for ACTIVE neurons only (amplitude > 0)."""
        if ds.neuron_spike_amplitudes is not None:
            amps = ds.neuron_spike_amplitudes
            return amps[amps > 0]
        if ds.selected_traces is None or ds.selected_spikes is None:
            return np.array([])
        amps = _measure_transient_amplitudes(
            ds.selected_traces, ds.selected_spikes, ds.frame_rate)
        amps = np.array(amps) if amps else np.array([])
        return amps[amps > 0] if len(amps) > 0 else amps

    # ── Colour scheme ────────────────────────────────────────────────────
    CTRL_COLOR = '#4472C4'   # blue
    MUT_COLOR = '#ED7D31'    # orange

    rng = np.random.default_rng(42)

    # =====================================================================
    # SECTION 1: GLOBAL COMPARISON (Control vs Mutant, all days)
    # =====================================================================
    # PRIMARY: per-recording averages (each recording = one data point)
    # This prevents single recordings with many neurons from dominating.
    # SUPPLEMENTARY: per-neuron pooled data kept for distribution plots.
    # =====================================================================
    logger.info("=== Global genotype comparison (per-recording averaging) ===")

    # Pool per-neuron metrics by genotype (for supplementary)
    ctrl_ds = [ds for ds, g, _ in geno_datasets if g == 'Control']
    mut_ds = [ds for ds, g, _ in geno_datasets if g == 'Mutant']

    # Per-neuron pooled (supplementary)
    ctrl_rates_raw = np.concatenate([_ds_spike_rates(ds) for ds in ctrl_ds]) if ctrl_ds else np.array([])
    mut_rates_raw = np.concatenate([_ds_spike_rates(ds) for ds in mut_ds]) if mut_ds else np.array([])
    ctrl_amps_raw = np.concatenate([_ds_spike_amplitudes(ds) for ds in ctrl_ds
                                    if len(_ds_spike_amplitudes(ds)) > 0]) if ctrl_ds else np.array([])
    mut_amps_raw = np.concatenate([_ds_spike_amplitudes(ds) for ds in mut_ds
                                   if len(_ds_spike_amplitudes(ds)) > 0]) if mut_ds else np.array([])

    # Per-recording averages (PRIMARY — each dot = one recording)
    def _recording_means(ds_list):
        """Compute per-recording mean rate and amplitude for active neurons."""
        rec_rates = []
        rec_amps = []
        for ds in ds_list:
            rates = _ds_spike_rates(ds)
            amps = _ds_spike_amplitudes(ds)
            if len(rates) > 0:
                rec_rates.append(float(np.mean(rates)))
            if len(amps) > 0:
                rec_amps.append(float(np.mean(amps)))
        return np.array(rec_rates), np.array(rec_amps)

    ctrl_rec_rates, ctrl_rec_amps = _recording_means(ctrl_ds)
    mut_rec_rates, mut_rec_amps = _recording_means(mut_ds)

    # Per-recording metrics (already one value per recording)
    # NaN = recording excluded from corr/sync due to n_selected < 5
    ctrl_corr = np.array([ds.pairwise_correlation_mean for ds in ctrl_ds])
    mut_corr  = np.array([ds.pairwise_correlation_mean for ds in mut_ds])
    ctrl_sync = np.array([ds.synchrony_index for ds in ctrl_ds])
    mut_sync  = np.array([ds.synchrony_index for ds in mut_ds])
    ctrl_af   = np.array([ds.active_fraction for ds in ctrl_ds])
    mut_af    = np.array([ds.active_fraction for ds in mut_ds])

    n_ctrl_corr_excl = int(np.sum(~np.isfinite(ctrl_corr)))
    n_mut_corr_excl  = int(np.sum(~np.isfinite(mut_corr)))
    if n_ctrl_corr_excl + n_mut_corr_excl > 0:
        logger.info(f"  Corr/sync: excluded {n_ctrl_corr_excl} Control and "
                    f"{n_mut_corr_excl} Mutant recordings (n_selected < 5)")

    logger.info(f"  Control: {len(ctrl_rec_rates)} recordings with active neurons "
                f"({len(ctrl_rates_raw)} total neurons)")
    logger.info(f"  Mutant:  {len(mut_rec_rates)} recordings with active neurons "
                f"({len(mut_rates_raw)} total neurons)")

    def _run_mw_test(ctrl, mut, label, use_zscore=False):
        """Run Mann-Whitney U and compute Cohen's d. NaN values are excluded."""
        result = {'metric': label, 'z_scored': use_zscore}
        ctrl = ctrl[np.isfinite(ctrl)]
        mut  = mut[np.isfinite(mut)]
        if len(ctrl) < 2 or len(mut) < 2:
            result['skipped'] = True
            result['reason'] = f"n_ctrl={len(ctrl)}, n_mut={len(mut)} (after NaN removal)"
            return result
        U, p = sp_stats.mannwhitneyu(ctrl, mut, alternative='two-sided')
        # Cohen's d
        pooled_std = np.sqrt(((len(ctrl)-1)*np.var(ctrl, ddof=1) +
                              (len(mut)-1)*np.var(mut, ddof=1)) /
                             (len(ctrl) + len(mut) - 2))
        d = (np.mean(ctrl) - np.mean(mut)) / pooled_std if pooled_std > 1e-10 else 0.0
        result.update({
            'n_ctrl': len(ctrl), 'n_mut': len(mut),
            'ctrl_mean': float(np.mean(ctrl)), 'ctrl_median': float(np.median(ctrl)),
            'ctrl_sd': float(np.std(ctrl, ddof=1)),
            'mut_mean': float(np.mean(mut)), 'mut_median': float(np.median(mut)),
            'mut_sd': float(np.std(mut, ddof=1)),
            'U': float(U), 'p': float(p), 'cohens_d': float(d),
        })
        return result

    # Run tests on per-recording averages (PRIMARY)
    global_tests = {}
    global_tests['spike_rate_raw'] = _run_mw_test(ctrl_rec_rates, mut_rec_rates,
                                                   'Event rate (events/10s)')
    global_tests['spike_amplitude_raw'] = _run_mw_test(ctrl_rec_amps, mut_rec_amps,
                                                        'Transient amplitude (ΔF/F₀)')
    global_tests['pairwise_correlation'] = _run_mw_test(ctrl_corr, mut_corr,
                                                         'Pairwise Correlation (r)')
    global_tests['synchrony_index'] = _run_mw_test(ctrl_sync, mut_sync,
                                                    'Synchrony Index')
    global_tests['active_fraction'] = _run_mw_test(ctrl_af, mut_af,
                                                    'Active Fraction')

    results['tests']['global_pooled'] = global_tests

    for k, t in global_tests.items():
        if 'p' in t:
            logger.info(f"  Global {k}: p={t['p']:.4f}, d={t['cohens_d']:.3f}, "
                        f"ctrl={t['ctrl_mean']:.3f}+/-{t['ctrl_sd']:.3f}, "
                        f"mut={t['mut_mean']:.3f}+/-{t['mut_sd']:.3f}")

    # =====================================================================
    # SECTION 2: WITHIN-DAY COMPARISON (paired by organoid day)
    # =====================================================================
    logger.info("=== Within-day genotype comparison ===")

    unique_days = sorted(set(organoid_ids))
    within_day_results = OrderedDict()
    day_effect_sizes = []  # for meta-analysis

    for day in unique_days:
        day_ctrl = [ds for ds, g, o in geno_datasets if o == day and g == 'Control']
        day_mut = [ds for ds, g, o in geno_datasets if o == day and g == 'Mutant']

        if not day_ctrl or not day_mut:
            logger.info(f"  {day}: skipped (ctrl={len(day_ctrl)}, mut={len(day_mut)})")
            within_day_results[day] = {
                'n_ctrl_recordings': len(day_ctrl),
                'n_mut_recordings': len(day_mut),
                'skipped': True,
            }
            continue

        # Per-recording mean spike rates (active neurons only)
        day_ctrl_rates = np.array([float(np.mean(r)) for ds in day_ctrl
                                   for r in [_ds_spike_rates(ds)] if len(r) > 0])
        day_mut_rates = np.array([float(np.mean(r)) for ds in day_mut
                                  for r in [_ds_spike_rates(ds)] if len(r) > 0])

        day_result = {
            'n_ctrl_recordings': len(day_ctrl),
            'n_mut_recordings': len(day_mut),
            'n_ctrl_with_active': len(day_ctrl_rates),
            'n_mut_with_active': len(day_mut_rates),
        }

        # Raw test (per-recording means)
        day_result['spike_rate_raw'] = _run_mw_test(
            day_ctrl_rates, day_mut_rates, f'{day} Event Rate')

        # Per-recording metrics for this day
        day_ctrl_corr = np.array([ds.pairwise_correlation_mean for ds in day_ctrl])
        day_mut_corr  = np.array([ds.pairwise_correlation_mean for ds in day_mut])
        day_ctrl_sync = np.array([ds.synchrony_index for ds in day_ctrl])
        day_mut_sync  = np.array([ds.synchrony_index for ds in day_mut])
        # Filter NaN (n_selected < 5) before within-day tests
        day_ctrl_corr = day_ctrl_corr[np.isfinite(day_ctrl_corr)]
        day_mut_corr  = day_mut_corr[np.isfinite(day_mut_corr)]
        day_ctrl_sync = day_ctrl_sync[np.isfinite(day_ctrl_sync)]
        day_mut_sync  = day_mut_sync[np.isfinite(day_mut_sync)]

        day_result['correlation'] = _run_mw_test(
            day_ctrl_corr, day_mut_corr, f'{day} Correlation')
        day_result['synchrony'] = _run_mw_test(
            day_ctrl_sync, day_mut_sync, f'{day} Synchrony')

        within_day_results[day] = day_result

        # Store effect size for meta-analysis (per-recording spike rate)
        t = day_result['spike_rate_raw']
        if 'cohens_d' in t:
            n_c, n_m = t['n_ctrl'], t['n_mut']
            se_d = np.sqrt((n_c + n_m) / (n_c * n_m) + t['cohens_d']**2 / (2 * (n_c + n_m)))
            day_effect_sizes.append({
                'day': day, 'd': t['cohens_d'], 'se': se_d,
                'n_ctrl': n_c, 'n_mut': n_m,
            })

        logger.info(f"  {day}: ctrl={len(day_ctrl_rates)}rec/{len(day_ctrl)}total, "
                     f"mut={len(day_mut_rates)}rec/{len(day_mut)}total, "
                     f"raw {_fmt_p(day_result['spike_rate_raw']['p']) if 'p' in day_result.get('spike_rate_raw', {}) else 'skipped'}")

    results['tests']['within_day'] = within_day_results

    # =====================================================================
    # SECTION 3: PAIRED META-ANALYSIS ACROSS DAYS
    # =====================================================================
    meta_result = {'n_paired_days': len(day_effect_sizes)}
    if len(day_effect_sizes) >= 2:
        ds_arr = np.array([e['d'] for e in day_effect_sizes])
        se_arr = np.array([e['se'] for e in day_effect_sizes])
        weights = 1.0 / (se_arr**2 + 1e-10)
        pooled_d = np.sum(weights * ds_arr) / np.sum(weights)
        pooled_se = 1.0 / np.sqrt(np.sum(weights))
        z_meta = pooled_d / pooled_se if pooled_se > 1e-10 else 0.0
        p_meta = 2 * (1 - sp_stats.norm.cdf(abs(z_meta)))

        meta_result.update({
            'pooled_cohens_d': float(pooled_d),
            'pooled_se': float(pooled_se),
            'z': float(z_meta),
            'p': float(p_meta),
            'per_day': day_effect_sizes,
            'interpretation': (
                'Positive d = Control > Mutant spike rate. '
                'This meta-analysis weights each day by inverse variance, '
                'controlling for day-to-day variation in imaging conditions.'
            ),
        })
        logger.info(f"  Meta-analysis: pooled d={pooled_d:.3f}, p={p_meta:.4f} "
                     f"({len(day_effect_sizes)} paired days)")
    else:
        meta_result['skipped'] = True
        meta_result['reason'] = f"Need >= 2 days with both genotypes, got {len(day_effect_sizes)}"

    results['tests']['meta_analysis'] = meta_result

    # =====================================================================
    # FIGURES — clean white-bg individual PNGs for genotype comparison
    # =====================================================================
    CTRL_COL = CTRL_COLOR   # '#4472C4' blue
    MUT_COL  = MUT_COLOR    # '#ED7D31' orange

    def _clean_boxplot(ax, ctrl_vals, mut_vals, ylabel, test_key):
        """White-background boxplot panel for genotype comparison."""
        data = [ctrl_vals, mut_vals]
        colors = [CTRL_COL, MUT_COL]

        if all(len(d) > 0 for d in data):
            bp = ax.boxplot(data, positions=[0, 1], widths=0.5,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='#E53935', linewidth=2.5),
                            whiskerprops=dict(color='#666', linewidth=1.0),
                            capprops=dict(color='#666', linewidth=1.0))
            for patch, col in zip(bp['boxes'], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.35)
                patch.set_edgecolor(col)
                patch.set_linewidth(1.5)

        for i, (vals, col) in enumerate(zip(data, colors)):
            if len(vals) > 0:
                jitter_x = rng.uniform(-0.12, 0.12, len(vals))
                ax.scatter(i + jitter_x, vals, c=col, s=60, alpha=0.75,
                           zorder=5, edgecolors='white', linewidth=0.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control', mutant_label], fontsize=16, fontweight='bold')
        for tick_label, col in zip(ax.get_xticklabels(), colors):
            tick_label.set_color(col)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(labelsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)

        test = global_tests.get(test_key, {})
        if 'p' in test:
            stars = _sig_stars(test['p'])
            _cv = ctrl_vals[np.isfinite(ctrl_vals)] if len(ctrl_vals) > 0 else ctrl_vals
            _mv = mut_vals[np.isfinite(mut_vals)]   if len(mut_vals)  > 0 else mut_vals
            y_max = max(float(np.max(_cv)) if len(_cv) > 0 else 0,
                        float(np.max(_mv)) if len(_mv) > 0 else 0)
            if y_max > 0:
                _draw_sig_bracket(ax, 0, 1, y_max * 1.05, y_max * 0.06,
                                  f'{stars}\n{_fmt_p(test["p"])}', fontsize=13,
                                  color='#333')
                ax.set_ylim(top=y_max * 1.25)

    # ── Individual metric comparison figures ─────────────────────────────
    individual_panels = [
        ('active_fraction',        ctrl_af,         mut_af,       'Active fraction',                 'active_fraction'),
        ('event_rate',             ctrl_rec_rates,  mut_rec_rates,'Event rate (events/10s)',          'spike_rate_raw'),
        ('event_amplitude',        ctrl_rec_amps,   mut_rec_amps, 'Mean transient amplitude (ΔF/F₀)','spike_amplitude_raw'),
        ('pairwise_correlation',   ctrl_corr[np.isfinite(ctrl_corr)],
                                   mut_corr[np.isfinite(mut_corr)],
                                                                  'Mean pairwise correlation (r)',   'pairwise_correlation'),
        ('synchrony_index',        ctrl_sync[np.isfinite(ctrl_sync)],
                                   mut_sync[np.isfinite(mut_sync)],
                                                                  'Synchrony index',                 'synchrony_index'),
    ]

    for fname, cv, mv, ylabel, test_key in individual_panels:
        fig, ax = plt.subplots(figsize=(5.5, 6.5))
        _clean_boxplot(ax, cv, mv, ylabel, test_key)
        n_c = len(cv[np.isfinite(cv)]) if len(cv) > 0 else 0
        n_m = len(mv[np.isfinite(mv)]) if len(mv) > 0 else 0
        fig.text(0.5, 0.01,
                 f'Control: {n_c}  |  Mutant: {n_m} recordings  |  Mann-Whitney U',
                 ha='center', fontsize=11, color='#888', style='italic')
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(os.path.join(geno_dir, f'genotype_{fname}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    # ── Combined activity figure (1×3: rate, amplitude, active fraction) ──
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 7))
    combined_panels = [
        (ctrl_rec_rates, mut_rec_rates, 'Event rate (events/10s)', 'spike_rate_raw'),
        (ctrl_rec_amps,  mut_rec_amps,  'Mean transient amplitude (ΔF/F₀)', 'spike_amplitude_raw'),
        (ctrl_af,        mut_af,        'Active fraction',                   'active_fraction'),
    ]
    for ax, (cv, mv, ylabel, key) in zip(axes1, combined_panels):
        _clean_boxplot(ax, cv, mv, ylabel, key)
    fig1.suptitle('Genotype comparison: activity metrics', fontsize=18, fontweight='bold')
    fig1.text(0.5, 0.01,
              f'Control: {n_ctrl}  |  Mutant: {n_mut} recordings  |  Mann-Whitney U',
              ha='center', fontsize=11, color='#888', style='italic')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig1.savefig(os.path.join(geno_dir, 'genotype_activity_combined.png'),
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)

    # ── Active fraction vs pairwise correlation scatter ──────────────────
    fig_scatter, ax_sc = plt.subplots(figsize=(8, 7))
    ctrl_corr_fin = ctrl_corr[np.isfinite(ctrl_corr)]
    mut_corr_fin  = mut_corr[np.isfinite(mut_corr)]
    # Match active fractions to recordings with valid correlation
    ctrl_af_fin = np.array([ds.active_fraction for ds in ctrl_ds
                            if np.isfinite(ds.pairwise_correlation_mean)])
    mut_af_fin  = np.array([ds.active_fraction for ds in mut_ds
                            if np.isfinite(ds.pairwise_correlation_mean)])
    ctrl_names_fin = [ds.name for ds in ctrl_ds
                      if np.isfinite(ds.pairwise_correlation_mean)]
    mut_names_fin  = [ds.name for ds in mut_ds
                      if np.isfinite(ds.pairwise_correlation_mean)]

    ax_sc.scatter(ctrl_af_fin, ctrl_corr_fin, c=CTRL_COL, s=50, alpha=0.7,
                  edgecolors='white', linewidth=0.5, label='Control', zorder=5)
    ax_sc.scatter(mut_af_fin, mut_corr_fin, c=MUT_COL, s=50, alpha=0.7,
                  edgecolors='white', linewidth=0.5, label=mutant_label, zorder=5)

    # Label outliers (IQR method: > Q3 + 1.5*IQR in either dimension)
    all_af = np.concatenate([ctrl_af_fin, mut_af_fin])
    all_corr = np.concatenate([ctrl_corr_fin, mut_corr_fin])

    def _iqr_upper_fence(values):
        """Compute upper outlier fence: Q3 + 1.5 * IQR."""
        q1, q3 = np.percentile(values, [25, 75])
        return q3 + 1.5 * (q3 - q1)

    af_thresh = _iqr_upper_fence(all_af) if len(all_af) > 5 else np.inf
    corr_thresh = _iqr_upper_fence(all_corr) if len(all_corr) > 5 else np.inf

    # Collect outlier details for reporting
    outlier_records = []

    for af_arr, corr_arr, names, col, geno_label in [
        (ctrl_af_fin, ctrl_corr_fin, ctrl_names_fin, CTRL_COL, 'Control'),
        (mut_af_fin,  mut_corr_fin,  mut_names_fin,  MUT_COL, mutant_label),
    ]:
        for i in range(len(af_arr)):
            af_out = af_arr[i] > af_thresh
            corr_out = corr_arr[i] > corr_thresh
            if af_out or corr_out:
                # Abbreviate name for label
                short = names[i].split('_')[0] + '_' + names[i].split('_')[1] if '_' in names[i] else names[i][:15]
                ax_sc.annotate(short, (af_arr[i], corr_arr[i]),
                               fontsize=6, color=col, alpha=0.8,
                               xytext=(5, 5), textcoords='offset points')
                outlier_records.append({
                    'recording': names[i],
                    'genotype': geno_label,
                    'active_fraction': float(af_arr[i]),
                    'pairwise_corr': float(corr_arr[i]),
                    'outlier_af': af_out,
                    'outlier_corr': corr_out,
                })

    # Log outlier results
    af_q1, af_q3 = np.percentile(all_af, [25, 75])
    corr_q1, corr_q3 = np.percentile(all_corr, [25, 75])
    n_ctrl_out = sum(1 for o in outlier_records if o['genotype'] == 'Control')
    n_mut_out = sum(1 for o in outlier_records if o['genotype'] != 'Control')

    logger.info(f"  Outlier detection (IQR method, Q3 + 1.5*IQR):")
    logger.info(f"    Active fraction:      Q1={af_q1:.4f}, Q3={af_q3:.4f}, "
                f"IQR={af_q3-af_q1:.4f}, upper fence={af_thresh:.4f}")
    logger.info(f"    Pairwise correlation: Q1={corr_q1:.4f}, Q3={corr_q3:.4f}, "
                f"IQR={corr_q3-corr_q1:.4f}, upper fence={corr_thresh:.4f}")
    logger.info(f"    Outliers: {len(outlier_records)} total "
                f"(Control: {n_ctrl_out}, {mutant_label}: {n_mut_out})")
    for o in outlier_records:
        flags = []
        if o['outlier_af']:
            flags.append(f"AF={o['active_fraction']:.3f}")
        if o['outlier_corr']:
            flags.append(f"corr={o['pairwise_corr']:.3f}")
        logger.info(f"      {o['recording']} ({o['genotype']}): {', '.join(flags)}")

    # Write outlier report to file
    outlier_report_path = os.path.join(geno_dir, 'outlier_report.txt')
    with open(outlier_report_path, 'w') as f_out:
        f_out.write("Outlier Detection Report: Active Fraction vs Pairwise Correlation\n")
        f_out.write("=" * 70 + "\n\n")
        f_out.write("Method: IQR (Interquartile Range) upper fence\n")
        f_out.write("Criterion: value > Q3 + 1.5 * IQR in either metric\n")
        f_out.write(f"N recordings: {len(all_af)} "
                    f"(Control: {len(ctrl_af_fin)}, {mutant_label}: {len(mut_af_fin)})\n\n")
        f_out.write("Thresholds:\n")
        f_out.write(f"  Active fraction:      Q1={af_q1:.4f}, Q3={af_q3:.4f}, "
                    f"IQR={af_q3-af_q1:.4f}, upper fence={af_thresh:.4f}\n")
        f_out.write(f"  Pairwise correlation: Q1={corr_q1:.4f}, Q3={corr_q3:.4f}, "
                    f"IQR={corr_q3-corr_q1:.4f}, upper fence={corr_thresh:.4f}\n\n")
        f_out.write(f"Outliers identified: {len(outlier_records)} "
                    f"(Control: {n_ctrl_out}, {mutant_label}: {n_mut_out})\n")
        f_out.write("-" * 70 + "\n")
        f_out.write(f"{'Recording':<35} {'Genotype':<15} {'AF':>8} {'Corr':>8} {'Flag':>15}\n")
        f_out.write("-" * 70 + "\n")
        for o in outlier_records:
            flags = []
            if o['outlier_af']:
                flags.append('AF')
            if o['outlier_corr']:
                flags.append('Corr')
            f_out.write(f"{o['recording']:<35} {o['genotype']:<15} "
                        f"{o['active_fraction']:>8.4f} {o['pairwise_corr']:>8.4f} "
                        f"{'  '.join(flags):>15}\n")
        f_out.write("-" * 70 + "\n")
    logger.info(f"  Outlier report saved to {outlier_report_path}")

    ax_sc.set_xlabel('Active fraction', fontsize=16)
    ax_sc.set_ylabel('Mean pairwise correlation (r)', fontsize=16)
    ax_sc.spines['top'].set_visible(False)
    ax_sc.spines['right'].set_visible(False)
    ax_sc.grid(alpha=0.15)
    ax_sc.legend(fontsize=14)
    ax_sc.tick_params(labelsize=13)
    plt.tight_layout()
    fig_scatter.savefig(os.path.join(geno_dir, 'genotype_af_vs_correlation.png'),
                        dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_scatter)

    # ── Figure: Event rate + amplitude: Event rate + amplitude (1×2 panel) ───────
    fig_r3, (ax_r3a, ax_r3b) = plt.subplots(1, 2, figsize=(12, 7))
    _clean_boxplot(ax_r3a, ctrl_rec_rates, mut_rec_rates,
                   'Event rate (events/10s)', 'spike_rate_raw')
    _clean_boxplot(ax_r3b, ctrl_rec_amps, mut_rec_amps,
                   'Mean transient amplitude (ΔF/F₀)', 'spike_amplitude_raw')
    ax_r3a.text(-0.08, 1.05, 'A', transform=ax_r3a.transAxes,
                fontsize=22, fontweight='bold', va='top')
    ax_r3b.text(-0.08, 1.05, 'B', transform=ax_r3b.transAxes,
                fontsize=22, fontweight='bold', va='top')
    fig_r3.text(0.5, 0.01,
                f'Control: {n_ctrl}  |  Mutant: {n_mut} recordings  |  '
                f'Mann-Whitney U, two-sided',
                ha='center', fontsize=11, color='#888', style='italic')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig_r3.savefig(os.path.join(geno_dir, 'genotype_rate_amplitude.png'),
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_r3)

    # ── Figure: Developmental trajectory: Developmental trajectory (2×2 panel) ─────
    try:
        import re as _re_r7
        from scipy import stats as _stats_r7

        def _day_sort_key_r7(d):
            nums = _re_r7.findall(r'\d+', d)
            return int(nums[0]) if nums else 0

        all_days_r7 = OrderedDict()
        for ds, g, o in geno_datasets:
            all_days_r7.setdefault(o, OrderedDict())
            all_days_r7[o].setdefault(g, []).append(ds)
        sorted_days_r7 = sorted(all_days_r7.keys(), key=_day_sort_key_r7)
        day_x_r7 = {d: i for i, d in enumerate(sorted_days_r7)}

        geno_display_r7 = {'Control': 'Control', 'Mutant': mutant_label}

        r7_metrics = [
            ('active_fraction', 'Active fraction'),
            ('mean_spike_rate', 'Event rate (events/10s)'),
            ('mean_spike_amplitude', 'Mean transient amplitude (ΔF/F₀)'),
            ('total_events', 'Total events per recording'),
        ]

        fig_r7, axes_r7 = plt.subplots(2, 2, figsize=(16, 12))
        axes_r7 = axes_r7.ravel()
        panel_labels = ['A', 'B', 'C', 'D']

        for ax, (attr, ylabel), plabel in zip(axes_r7, r7_metrics, panel_labels):
            ax.set_facecolor('white')
            ax.text(-0.08, 1.05, plabel, transform=ax.transAxes,
                    fontsize=22, fontweight='bold', va='top')

            for geno, color in [('Control', CTRL_COLOR), ('Mutant', MUT_COLOR)]:
                # Collect per-day mean and SEM
                day_means, day_sems, day_xs = [], [], []
                for day in sorted_days_r7:
                    gds = all_days_r7[day].get(geno, [])
                    if gds:
                        day_vals = [v for v in (_recording_metric(d, attr) for d in gds)
                                    if v is not None and np.isfinite(v)]
                        if day_vals:
                            day_xs.append(day_x_r7[day])
                            day_means.append(np.mean(day_vals))
                            sem = np.std(day_vals, ddof=1) / np.sqrt(len(day_vals)) if len(day_vals) > 1 else 0.0
                            day_sems.append(sem)

                if not day_xs:
                    continue

                day_xs = np.array(day_xs, dtype=float)
                day_means = np.array(day_means)
                day_sems = np.array(day_sems)

                # Slight x-offset to separate Control and Mutant
                x_offset = -0.1 if geno == 'Control' else 0.1

                ax.errorbar(day_xs + x_offset, day_means, yerr=day_sems,
                            fmt='o-', color=color, markersize=7, linewidth=2,
                            capsize=4, capthick=1.5, elinewidth=1.5,
                            alpha=0.85, zorder=6, label=geno_display_r7[geno])

                # Individual data points (small, behind error bars)
                for day in sorted_days_r7:
                    gds = all_days_r7[day].get(geno, [])
                    day_vals = [v for v in (_recording_metric(d, attr) for d in gds)
                                if v is not None and np.isfinite(v)]
                    if day_vals:
                        jitter = rng.uniform(-0.06, 0.06, len(day_vals))
                        ax.scatter(np.full(len(day_vals), day_x_r7[day]) + x_offset + jitter,
                                   day_vals, c=color, s=16, alpha=0.35,
                                   edgecolor='none', zorder=4)

            ax.set_xticks(range(len(sorted_days_r7)))
            ax.set_xticklabels(sorted_days_r7, fontsize=13, rotation=45, ha='right')
            ax.set_ylabel(ylabel, fontsize=15)
            ax.set_xlabel('Developmental day', fontsize=15)
            ax.tick_params(labelsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.15)
            ax.legend(fontsize=13)

            # Robust y-limits: set bottom to 0, clip top to 97th percentile
            all_y = []
            for line in ax.lines:
                ydata = line.get_ydata()
                if len(ydata) > 0:
                    all_y.extend(ydata.tolist())
            if len(all_y) > 3:
                y_hi = np.percentile(all_y, 97) * 1.15
                ax.set_ylim(bottom=0, top=y_hi)

        fig_r7.suptitle(f'Developmental trajectory: Control vs {mutant_label}',
                        fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_r7.savefig(os.path.join(geno_dir, 'genotype_longitudinal.png'),
                       dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_r7)
    except Exception as e:
        logger.warning(f"Longitudinal figure failed: {e}")

    global_path = os.path.join(geno_dir, 'genotype_activity_combined.png')

    # ── Figure 3: Within-day breakdown ───────────────────────────────────
    paired_days = [d for d in unique_days
                   if not within_day_results.get(d, {}).get('skipped', True)]

    if paired_days:
        n_days = len(paired_days)
        fig, axes = plt.subplots(1, n_days, figsize=(max(6, n_days * 4.5), 6),
                                 squeeze=False)
        fig.patch.set_facecolor('white')

        for i, day in enumerate(paired_days):
            ax = axes[0, i]
            ax.set_facecolor('white')

            day_ctrl = [ds for ds, g, o in geno_datasets if o == day and g == 'Control']
            day_mut = [ds for ds, g, o in geno_datasets if o == day and g == 'Mutant']

            # Per-recording mean spike rates (active neurons only)
            c_rates = np.array([float(np.mean(r)) for ds in day_ctrl
                                for r in [_ds_spike_rates(ds)] if len(r) > 0])
            m_rates = np.array([float(np.mean(r)) for ds in day_mut
                                for r in [_ds_spike_rates(ds)] if len(r) > 0])

            if len(c_rates) == 0 or len(m_rates) == 0:
                continue

            bp = ax.boxplot([c_rates, m_rates], positions=[0, 1], widths=0.5,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='white', linewidth=1.5),
                            whiskerprops=dict(color='#555', linewidth=0.8),
                            capprops=dict(color='#555', linewidth=0.8))
            for patch, col_c in zip(bp['boxes'], [CTRL_COLOR, MUT_COLOR]):
                patch.set_facecolor(col_c)
                patch.set_alpha(0.35)
                patch.set_edgecolor(col_c)

            for j, (vals, col_c) in enumerate(zip([c_rates, m_rates],
                                                   [CTRL_COLOR, MUT_COLOR])):
                jitter = rng.uniform(-0.15, 0.15, len(vals))
                ax.scatter(j + jitter, vals, c=col_c, s=40, alpha=0.7,
                           zorder=5, edgecolors='white', linewidth=0.5)

            ax.set_xticks([0, 1])
            ax.set_xticklabels([f'Ctrl\n(n={len(c_rates)} rec)',
                                f'Mut\n(n={len(m_rates)} rec)'],
                               fontsize=9)
            for tick, col_c in zip(ax.get_xticklabels(), [CTRL_COLOR, MUT_COLOR]):
                tick.set_color(col_c)

            # Significance
            wd = within_day_results[day]
            t_raw = wd.get('spike_rate_raw', {})
            if 'p' in t_raw:
                stars = _sig_stars(t_raw['p'])
                _cr = c_rates[np.isfinite(c_rates)]
                _mr = m_rates[np.isfinite(m_rates)]
                y_max = max(float(np.max(_cr)) if len(_cr) > 0 else 0,
                            float(np.max(_mr)) if len(_mr) > 0 else 0)
                _draw_sig_bracket(ax, 0, 1, y_max * 1.05, y_max * 0.06,
                                  f"{stars} {_fmt_p(t_raw['p'])}", fontsize=8)
                ax.set_ylim(top=y_max * 1.3)

            ax.set_title(f'{day}\n({len(day_ctrl)} ctrl, {len(day_mut)} mut rec)',
                         fontsize=10, fontweight='bold')
            ax.set_ylabel('Event rate (events/10s)' if i == 0 else '', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.15)

        fig.suptitle(f'Within-Day Genotype Comparison: Control vs {mutant_label}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        within_path = os.path.join(geno_dir, 'genotype_within_day.png')
        plt.savefig(within_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

    # ── Figure 4: Meta-analysis forest plot ──────────────────────────────
    if len(day_effect_sizes) >= 2:
        fig, ax = plt.subplots(figsize=(10, max(4, len(day_effect_sizes) * 0.8 + 2)))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        y_positions = list(range(len(day_effect_sizes)))
        day_labels = []

        for y, es in zip(y_positions, day_effect_sizes):
            d_val = es['d']
            se = es['se']
            ci_lo = d_val - 1.96 * se
            ci_hi = d_val + 1.96 * se

            ax.plot([ci_lo, ci_hi], [y, y], color='#555', linewidth=1.5)
            ax.scatter(d_val, y, color=CTRL_COLOR if d_val > 0 else MUT_COLOR,
                       s=80, zorder=5, edgecolor='white', linewidth=0.5)
            day_labels.append(f"{es['day']} (n={es['n_ctrl']}+{es['n_mut']})")

        # Pooled estimate
        pooled = meta_result.get('pooled_cohens_d', 0)
        pooled_se = meta_result.get('pooled_se', 0)
        y_pooled = len(day_effect_sizes) + 0.5
        ax.axhline(y_pooled - 0.3, color='#ccc', linewidth=0.5)
        ax.plot([pooled - 1.96*pooled_se, pooled + 1.96*pooled_se],
                [y_pooled, y_pooled], color='#333', linewidth=2.5)
        ax.scatter(pooled, y_pooled, color='#333', s=120, marker='D',
                   zorder=5, edgecolor='white')
        day_labels.append(f"POOLED (p={meta_result.get('p', 1):.3f})")

        ax.axvline(0, color='#999', linestyle='--', linewidth=1, zorder=1)

        ax.set_yticks(y_positions + [y_pooled])
        ax.set_yticklabels(day_labels, fontsize=9)
        ax.set_xlabel("Cohen's d (positive = Control > Mutant)", fontsize=10)
        ax.set_title('Meta-Analysis: Z-Scored Event Rate Effect Sizes by Day',
                     fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()

        plt.tight_layout()
        forest_path = os.path.join(geno_dir, 'genotype_meta_analysis.png')
        plt.savefig(forest_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

    # ── Figure 5: Longitudinal developmental trajectory ──────────────────
    # Shows how each metric evolves over day-age, split by genotype,
    # addressing the secondary objective of tracking the relationship over time
    try:
        import re as _re

        def _day_sort_key(d):
            nums = _re.findall(r'\d+', d)
            return int(nums[0]) if nums else 0

        # Build day -> genotype -> [datasets] mapping from geno_datasets
        all_days = OrderedDict()
        for ds, g, o in geno_datasets:
            day = o  # organoid_id is the day-age (e.g. D109)
            all_days.setdefault(day, OrderedDict())
            all_days[day].setdefault(g, []).append(ds)
        sorted_all_days = sorted(all_days.keys(), key=_day_sort_key)
        day_x = {d: i for i, d in enumerate(sorted_all_days)}

        longit_metrics = [
            ('mean_spike_rate', 'Event rate (events/10s)'),
            ('pairwise_correlation_mean', 'Mean pairwise correlation (r)'),
            ('synchrony_index', 'Synchrony index'),
            ('mean_spike_amplitude', 'Mean transient amplitude (ΔF/F₀)'),
            ('active_fraction', 'Active fraction'),
        ]

        # Helper: get per-recording metric value consistent with global
        # comparison (active-neuron-only means for rate/amplitude).
        def _longit_value(ds, attr):
            return _recording_metric(ds, attr)

        n_metrics = len(longit_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig_long, axes_long = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 6))
        fig_long.patch.set_facecolor('white')
        axes_long = axes_long.ravel()
        # Hide unused axes
        for ax_i in range(n_metrics, len(axes_long)):
            axes_long[ax_i].set_visible(False)

        geno_display = {'Control': 'Control', 'Mutant': mutant_label}
        for ax, (attr, ylabel) in zip(axes_long, longit_metrics):
            ax.set_facecolor('white')
            for geno, color in [('Control', CTRL_COLOR), ('Mutant', MUT_COLOR)]:
                x_vals, y_vals = [], []
                for day in sorted_all_days:
                    for ds in all_days[day].get(geno, []):
                        val = _longit_value(ds, attr)
                        # Skip None and NaN (corr/sync excluded due to n < 5)
                        if val is not None and np.isfinite(val):
                            x_vals.append(day_x[day])
                            y_vals.append(val)

                if not x_vals:
                    continue

                jitter = rng.uniform(-0.12, 0.12, len(x_vals))
                ax.scatter(np.array(x_vals) + jitter, y_vals, c=color,
                           s=40, alpha=0.6, edgecolor='white', linewidth=0.5,
                           zorder=5, label=geno_display[geno])

                # Mean trend line connecting days
                dmx, dmy = [], []
                for day in sorted_all_days:
                    gds = all_days[day].get(geno, [])
                    if gds:
                        day_vals = [v for v in (_longit_value(d, attr) for d in gds)
                                    if v is not None and np.isfinite(v)]
                        if day_vals:
                            dmx.append(day_x[day])
                            dmy.append(np.mean(day_vals))
                if len(dmx) > 1:
                    ax.plot(dmx, dmy, color=color, linewidth=2, alpha=0.7,
                            marker='o', markersize=6, zorder=6)

            ax.set_xticks(range(len(sorted_all_days)))
            ax.set_xticklabels(sorted_all_days, fontsize=9, rotation=45, ha='right')
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlabel('Day Age', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.2)
            ax.legend(fontsize=9)

            # Robust y-limits: clip to 95th percentile to prevent outlier stretching
            all_y = [v for v in ax.collections[0].get_offsets()[:, 1]] if ax.collections else []
            for coll in ax.collections:
                offs = coll.get_offsets()
                if len(offs) > 0:
                    all_y.extend(offs[:, 1].tolist())
            if len(all_y) > 5:
                y_lo = max(0, np.percentile(all_y, 1) - 0.01)
                y_hi = np.percentile(all_y, 97) * 1.15
                ax.set_ylim(bottom=y_lo, top=y_hi)

        fig_long.suptitle('Developmental trajectory: mutant vs control',
                          fontsize=14, fontweight='bold')
        plt.tight_layout()
        longit_path = os.path.join(geno_dir, 'genotype_longitudinal.png')
        plt.savefig(longit_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    except Exception as e:
        logger.warning(f"Longitudinal figure failed: {e}")

    # ── Figure 5b: Per-cell-line across days ─────────────────────────────
    # For each metric, show side-by-side boxplots of every cell line at each
    # day-age so the user can see how individual lines behave over time
    try:
        import re as _re2

        # Build line -> day -> [metric values] mapping
        line_day_data = {}  # {line_id: {day: [ds, ...]}}
        for ds in datasets:
            line_id = _extract_line_id(ds.name)
            day = _extract_organoid_id(ds.name)
            line_day_data.setdefault(line_id, {}).setdefault(day, []).append(ds)

        all_lines = sorted(line_day_data.keys())
        all_days_set = set()
        for ld in line_day_data.values():
            all_days_set.update(ld.keys())

        def _day_num(d):
            nums = _re2.findall(r'\d+', d)
            return int(nums[0]) if nums else 0

        sorted_days = sorted(all_days_set, key=_day_num)
        n_lines = len(all_lines)
        n_days = len(sorted_days)

        if n_lines > 1 and n_days > 1:
            line_metrics = [
                ('mean_spike_rate', 'Event rate (events/10s)'),
                ('pairwise_correlation_mean', 'Pairwise Correlation (r)'),
                ('synchrony_index', 'Synchrony Index'),
                ('mean_spike_amplitude', 'Transient amplitude (ΔF/F₀)'),
                ('active_fraction', 'Active Fraction (selected/total)'),
            ]

            # Assign each line a colour: control lines in blue shades, mutant in orange
            import matplotlib.cm as _cm
            ctrl_lines = [l for l in all_lines if l.startswith('3')]
            mut_lines = [l for l in all_lines if not l.startswith('3')]
            blues = _cm.Blues(np.linspace(0.35, 0.85, max(len(ctrl_lines), 1)))
            oranges = _cm.Oranges(np.linspace(0.35, 0.85, max(len(mut_lines), 1)))
            line_colors = {}
            for i, l in enumerate(ctrl_lines):
                line_colors[l] = blues[i]
            for i, l in enumerate(mut_lines):
                line_colors[l] = oranges[i]

            fig_lines, axes_lines = plt.subplots(len(line_metrics), 1,
                                                  figsize=(max(14, n_days * n_lines * 0.6), 4 * len(line_metrics)))
            fig_lines.patch.set_facecolor('white')
            if len(line_metrics) == 1:
                axes_lines = [axes_lines]

            for ax, (attr, ylabel) in zip(axes_lines, line_metrics):
                ax.set_facecolor('white')

                # For each day, draw grouped boxplots — one per line
                positions = []
                box_data = []
                box_colors = []
                tick_positions = []
                tick_labels = []
                group_width = n_lines * 0.7 + 0.5
                x = 0

                for day in sorted_days:
                    day_start = x
                    has_data = False
                    for li, line_id in enumerate(all_lines):
                        ds_list = line_day_data.get(line_id, {}).get(day, [])
                        vals = [v for v in (_longit_value(ds, attr) for ds in ds_list) if v is not None]
                        if vals:
                            has_data = True
                            positions.append(x)
                            box_data.append(vals)
                            box_colors.append(line_colors.get(line_id, '#999'))
                            x += 0.7
                        else:
                            x += 0.7

                    if has_data:
                        tick_positions.append((day_start + x - 0.7) / 2)
                        tick_labels.append(day)
                    x += 1.0  # gap between days

                if box_data:
                    bp = ax.boxplot(box_data, positions=positions, widths=0.55,
                                    patch_artist=True, showfliers=True,
                                    flierprops=dict(marker='.', markersize=3, alpha=0.5))
                    for patch, color in zip(bp['boxes'], box_colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    for median in bp['medians']:
                        median.set_color('white')
                        median.set_linewidth(1.5)

                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, fontsize=9, rotation=45, ha='right')
                ax.set_ylabel(ylabel, fontsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', alpha=0.2)

            # Legend
            from matplotlib.patches import Patch as _Patch
            legend_handles = []
            for line_id in all_lines:
                geno = 'Ctrl' if line_id.startswith('3') else 'Mut'
                legend_handles.append(_Patch(
                    facecolor=line_colors.get(line_id, '#999'),
                    alpha=0.7, label=f'{line_id} ({geno})'))
            axes_lines[0].legend(handles=legend_handles, fontsize=8,
                                  ncol=min(n_lines, 6), loc='upper right',
                                  framealpha=0.8)

            fig_lines.suptitle('Per-Cell-Line Metrics Across Days',
                               fontsize=14, fontweight='bold')
            plt.tight_layout()
            lines_path = os.path.join(geno_dir, 'genotype_per_line_longitudinal.png')
            plt.savefig(lines_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Per-line longitudinal figure saved: {lines_path}")
        else:
            logger.info("Skipping per-line figure: need >1 line and >1 day")

    except Exception as e:
        logger.warning(f"Per-line longitudinal figure failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
    # ── Figure 6: Summary statistics table ───────────────────────────────
    fig_table = plt.figure(figsize=(18, 10))
    fig_table.patch.set_facecolor('white')

    fig_table.text(0.5, 0.97, 'Genotype Comparison — Statistical Summary',
                   ha='center', fontsize=14, fontweight='bold')

    # Table A: Global comparison
    ax_t = fig_table.add_axes([0.05, 0.55, 0.90, 0.35])
    ax_t.axis('off')
    fig_table.text(0.5, 0.92, 'Global Pooled Comparison (all days)',
                   ha='center', fontsize=11, fontweight='bold')

    headers = ['Metric', 'Analysis', 'Ctrl mean +/- SD', 'Mut mean +/- SD',
               'U', 'p-value', "Cohen's d", 'Sig']
    rows = []
    for key in ['spike_rate_raw', 'spike_amplitude_raw',
                'pairwise_correlation', 'synchrony_index', 'active_fraction']:
        t = global_tests.get(key, {})
        if t.get('skipped') or 'metric' not in t:
            continue
        mode = 'Z-scored' if t.get('z_scored') else 'Raw'
        rows.append([
            t['metric'], mode,
            f"{t['ctrl_mean']:.3f} +/- {t['ctrl_sd']:.3f}",
            f"{t['mut_mean']:.3f} +/- {t['mut_sd']:.3f}",
            f"{t.get('U', 0):.0f}", _fmt_p(t['p']),
            f"{t['cohens_d']:.3f}", _sig_stars(t['p']),
        ])

    if rows:
        tbl = ax_t.table(cellText=rows, colLabels=headers,
                         loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor('#E6F1FB')
                cell.set_text_props(color='white', fontweight='bold')
            elif col == 7:  # Sig column
                txt = cell.get_text().get_text()
                if '***' in txt:
                    cell.set_facecolor('#C6EFCE')
                elif '**' in txt:
                    cell.set_facecolor('#D9F2D9')
                elif '*' == txt.strip():
                    cell.set_facecolor('#FFEB9C')
                else:
                    cell.set_facecolor('white')
            else:
                cell.set_facecolor('#F2F7FC' if row % 2 == 0 else 'white')
            cell.set_edgecolor('#DDD')

    # Table B: Within-day summary
    ax_t2 = fig_table.add_axes([0.05, 0.05, 0.90, 0.40])
    ax_t2.axis('off')
    fig_table.text(0.5, 0.48, 'Within-Day Breakdown (spike rate)',
                   ha='center', fontsize=11, fontweight='bold')

    headers2 = ['Day', 'Ctrl rec', 'Mut rec', 'Ctrl rate', 'Mut rate',
                'p-value', "Cohen's d"]
    rows2 = []
    for day in unique_days:
        wd = within_day_results.get(day, {})
        if wd.get('skipped'):
            rows2.append([day, str(wd.get('n_ctrl_recordings', 0)),
                         str(wd.get('n_mut_recordings', 0)),
                         '-', '-', '-', '-'])
            continue
        t_raw = wd.get('spike_rate_raw', {})
        rows2.append([
            day,
            str(wd.get('n_ctrl_with_active', wd.get('n_ctrl_recordings', 0))),
            str(wd.get('n_mut_with_active', wd.get('n_mut_recordings', 0))),
            f"{t_raw.get('ctrl_mean', 0):.2f}",
            f"{t_raw.get('mut_mean', 0):.2f}",
            _fmt_p(t_raw['p']) if 'p' in t_raw else '-',
            f"{t_raw.get('cohens_d', 0):.3f}" if 'cohens_d' in t_raw else '-',
        ])

    if rows2:
        tbl2 = ax_t2.table(cellText=rows2, colLabels=headers2,
                           loc='center', cellLoc='center')
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(8)
        tbl2.scale(1, 1.5)
        for (row, col), cell in tbl2.get_celld().items():
            if row == 0:
                cell.set_facecolor('#E6F1FB')
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#F2F7FC' if row % 2 == 0 else 'white')
            cell.set_edgecolor('#DDD')

    table_path = os.path.join(geno_dir, 'genotype_statistics_table.png')
    plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Genotype comparison figures saved to {geno_dir}")

    # ── Figure 7: Mixed-effects model analysis ───────────────────────────
    # Addresses the z-score limitation: z-scoring removes ALL between-recording
    # variance including genuine biological shifts. Mixed models separate
    # biological variance (genotype) from technical variance (recording).
    try:
        _run_mixed_model_analysis(
            datasets, genotypes, line_ids, organoid_ids,
            geno_dir, results)
    except Exception as e:
        logger.warning(f"Mixed-effects model analysis failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())

    return results


def _run_mixed_model_analysis(
    datasets: List[DatasetMetrics],
    genotypes: List[str],
    line_ids: List[str],
    organoid_ids: List[str],
    geno_dir: str,
    results: dict,
) -> None:
    """
    Mixed-effects model analysis of genotype effects.

    Addresses the limitation of z-scoring (which removes genuine global shifts)
    by using hierarchical models that separate technical variance (between
    recordings) from biological variance (between genotypes).

    Models:
    1. Recording-level LMM: metric ~ genotype + (1|organoid_day)
       Uses organoid day as random effect to absorb session-level confounds
       (same day = same imaging conditions). Conservative: one obs per recording.

    2. Neuron-level LMM: metric ~ genotype + (1|recording)
       Uses recording as random effect. More power but needs correct nesting.

    3. Permutation test: model-free, shuffles genotype labels across recording means.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    logger.info("Running mixed-effects model analysis...")

    # ── Build dataset-level data ─────────────────────────────────────────
    ds_records = []
    for ds, geno, line, org_id in zip(datasets, genotypes, line_ids, organoid_ids):
        if geno not in ('Control', 'Mutant'):
            continue
        # Use active-neuron-only means for rate/amplitude, consistent with
        # global comparison and longitudinal figures.
        active_rates = _get_neuron_rates(ds)
        active_amps = _get_neuron_amplitudes(ds)
        rec_rate = float(np.mean(active_rates)) if len(active_rates) > 0 else None
        rec_amp = float(np.mean(active_amps)) if len(active_amps) > 0 else None
        # Skip recordings with no active neurons for rate/amplitude
        # (correlation and synchrony are always available)
        ds_records.append({
            'name': ds.name,
            'genotype': geno,
            'genotype_code': 0 if geno == 'Control' else 1,
            'line_id': line,
            'organoid_day': org_id,
            'spike_rate': rec_rate,
            'spike_amplitude': rec_amp,
            'correlation': ds.pairwise_correlation_mean,
            'synchrony': ds.synchrony_index,
            'n_neurons': ds.n_selected,
            'mean_quality': ds.mean_quality_score,
        })

    if len(ds_records) < 6:
        logger.warning("Too few datasets for mixed model analysis")
        return

    # ── Build neuron-level data ──────────────────────────────────────────
    neuron_records = []
    for ds, geno, line, org_id in zip(datasets, genotypes, line_ids, organoid_ids):
        if geno not in ('Control', 'Mutant'):
            continue
        rates = _get_neuron_rates(ds)
        amps = _get_neuron_amplitudes(ds)
        for j in range(len(rates)):
            neuron_records.append({
                'genotype': geno,
                'genotype_code': 0 if geno == 'Control' else 1,
                'line_id': line,
                'recording': ds.name,
                'organoid_day': org_id,
                'spike_rate': rates[j],
                'spike_amplitude': amps[j] if j < len(amps) else 0.0,
            })

    # ── Try statsmodels ──────────────────────────────────────────────────
    has_statsmodels = False
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        has_statsmodels = True
    except ImportError:
        logger.warning("statsmodels not available — permutation tests only")

    metrics_recording = [
        ('spike_rate', 'Event rate (events/10s)'),
        ('spike_amplitude', 'Transient amplitude (ΔF/F₀)'),
        ('correlation', 'Pairwise Correlation (r)'),
        ('synchrony', 'Synchrony Index'),
    ]
    metrics_neuron = [
        ('spike_rate', 'Event rate (events/10s)'),
        ('spike_amplitude', 'Transient amplitude (ΔF/F₀)'),
    ]

    mixed_results = {}

    if has_statsmodels:
        ds_df = pd.DataFrame(ds_records)
        neuron_df = pd.DataFrame(neuron_records)

        n_days = ds_df['organoid_day'].nunique()
        n_recs = ds_df['name'].nunique()
        logger.info(f"  {len(ds_df)} recordings, {n_days} unique days, "
                    f"{len(neuron_df)} neurons")

        for metric, label in metrics_recording:
            mr = {'metric': metric, 'label': label}

            # ── Recording-level: metric ~ genotype + (1|organoid_day) ────
            try:
                # Drop rows with missing values for this metric (e.g. no active neurons)
                metric_df = ds_df.dropna(subset=[metric])
                n_days_metric = metric_df['organoid_day'].nunique()
                if n_days_metric >= 3:
                    model = smf.mixedlm(
                        f"{metric} ~ genotype_code",
                        data=metric_df,
                        groups=metric_df['organoid_day'],
                    )
                    fit = model.fit(reml=True, method='lbfgs')

                    coef = float(fit.params.get('genotype_code', 0))
                    se = float(fit.bse.get('genotype_code', 0))
                    p = float(fit.pvalues.get('genotype_code', 1))
                    ci = fit.conf_int()
                    ci_low = float(ci.loc['genotype_code', 0]) if 'genotype_code' in ci.index else coef - 1.96*se
                    ci_high = float(ci.loc['genotype_code', 1]) if 'genotype_code' in ci.index else coef + 1.96*se

                    mr['recording_lmm'] = {
                        'coef': coef, 'se': se, 'p': p,
                        'ci_low': ci_low, 'ci_high': ci_high,
                        'n_obs': len(metric_df), 'n_groups': n_days_metric,
                        'formula': f'{metric} ~ genotype + (1|organoid_day)',
                    }
                    logger.info(f"  {metric} rec-LMM: coef={coef:.4f}, p={p:.4f}")
                else:
                    mr['recording_lmm'] = {'skipped': f'need ≥3 days, have {n_days_metric}'}
            except Exception as e:
                mr['recording_lmm'] = {'error': str(e)}
                logger.warning(f"  {metric} rec-LMM failed: {e}")

            # ── Neuron-level: metric ~ genotype + (1|recording) ──────────
            if metric in ('spike_rate', 'spike_amplitude'):
                try:
                    n_recs_neuron = neuron_df['recording'].nunique()
                    if n_recs_neuron >= 5:
                        model = smf.mixedlm(
                            f"{metric} ~ genotype_code",
                            data=neuron_df,
                            groups=neuron_df['recording'],
                        )
                        fit = model.fit(reml=True, method='lbfgs')

                        coef = float(fit.params.get('genotype_code', 0))
                        se = float(fit.bse.get('genotype_code', 0))
                        p = float(fit.pvalues.get('genotype_code', 1))
                        ci = fit.conf_int()
                        ci_low = float(ci.loc['genotype_code', 0]) if 'genotype_code' in ci.index else coef - 1.96*se
                        ci_high = float(ci.loc['genotype_code', 1]) if 'genotype_code' in ci.index else coef + 1.96*se

                        mr['neuron_lmm'] = {
                            'coef': coef, 'se': se, 'p': p,
                            'ci_low': ci_low, 'ci_high': ci_high,
                            'n_obs': len(neuron_df), 'n_groups': n_recs_neuron,
                            'formula': f'{metric} ~ genotype + (1|recording)',
                        }
                        logger.info(f"  {metric} neuron-LMM: coef={coef:.4f}, p={p:.4f}")
                    else:
                        mr['neuron_lmm'] = {'skipped': f'need ≥5 recordings, have {n_recs_neuron}'}
                except Exception as e:
                    mr['neuron_lmm'] = {'error': str(e)}
                    logger.warning(f"  {metric} neuron-LMM failed: {e}")

            mixed_results[metric] = mr

    # ── Permutation test (always runs, model-free) ───────────────────────
    n_perms = 5000
    rng = np.random.default_rng(42)

    for metric, label in metrics_recording:
        if metric not in mixed_results:
            mixed_results[metric] = {'metric': metric, 'label': label}

        ctrl_vals = np.array([r[metric] for r in ds_records if r['genotype'] == 'Control' and r[metric] is not None and np.isfinite(r[metric])])
        mut_vals = np.array([r[metric] for r in ds_records if r['genotype'] == 'Mutant' and r[metric] is not None and np.isfinite(r[metric])])

        if len(ctrl_vals) < 2 or len(mut_vals) < 2:
            mixed_results[metric]['permutation'] = {'skipped': True}
            continue

        observed_diff = float(np.mean(mut_vals) - np.mean(ctrl_vals))
        all_vals = np.concatenate([ctrl_vals, mut_vals])
        nc = len(ctrl_vals)

        null_diffs = np.zeros(n_perms)
        for p_idx in range(n_perms):
            perm = rng.permutation(all_vals)
            null_diffs[p_idx] = np.mean(perm[nc:]) - np.mean(perm[:nc])

        p_perm = float(np.mean(np.abs(null_diffs) >= np.abs(observed_diff)))
        pooled_sd = float(np.std(all_vals, ddof=1))
        d_perm = observed_diff / pooled_sd if pooled_sd > 0 else 0.0

        mixed_results[metric]['permutation'] = {
            'observed_diff': observed_diff,
            'p': p_perm,
            'cohens_d': d_perm,
            'n_ctrl': len(ctrl_vals),
            'n_mut': len(mut_vals),
            'ctrl_mean': float(np.mean(ctrl_vals)),
            'ctrl_sd': float(np.std(ctrl_vals, ddof=1)),
            'mut_mean': float(np.mean(mut_vals)),
            'mut_sd': float(np.std(mut_vals, ddof=1)),
        }
        logger.info(f"  {metric} perm: diff={observed_diff:.4f}, p={p_perm:.4f}, d={d_perm:.3f}")

    results['mixed_model'] = mixed_results

    # ── Generate figures ─────────────────────────────────────────────────
    _fig_mixed_model_panel(
        mixed_results, metrics_recording, ds_records, geno_dir, results)


def _fig_mixed_model_panel(
    mixed_results: dict,
    metrics: list,
    ds_records: list,
    geno_dir: str,
    results: dict,
) -> None:
    """
    Two-panel figure per metric:
      Left:  recording-level dot plot (one dot per recording)
      Right: coefficient comparison across methods (raw units, not normalised)
    Plus a summary table.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    CTRL_COLOR = '#4472C4'
    MUT_COLOR = '#ED7D31'
    rng = np.random.default_rng(42)

    n_metrics = len(metrics)

    # ── Main figure: paired dot-plot + method comparison ─────────────────
    fig, axes = plt.subplots(n_metrics, 2, figsize=(14, 4 * n_metrics),
                             gridspec_kw={'width_ratios': [1.2, 1]})
    fig.patch.set_facecolor('white')
    if n_metrics == 1:
        axes = axes.reshape(1, 2)

    for row, (metric, label) in enumerate(metrics):
        mr = mixed_results.get(metric, {})
        perm = mr.get('permutation', {})

        ctrl = np.array([r[metric] for r in ds_records if r['genotype'] == 'Control' and r[metric] is not None and np.isfinite(r[metric])])
        mut = np.array([r[metric] for r in ds_records if r['genotype'] == 'Mutant' and r[metric] is not None and np.isfinite(r[metric])])

        # ── Left panel: recording-level dot plot ─────────────────────────
        ax_dot = axes[row, 0]
        ax_dot.set_facecolor('white')

        # Box plots (transparent)
        bp = ax_dot.boxplot(
            [ctrl, mut], positions=[0, 1], widths=0.4,
            patch_artist=True, showfliers=False,
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(color='#888'),
            capprops=dict(color='#888'))
        bp['boxes'][0].set(facecolor=CTRL_COLOR, alpha=0.2)
        bp['boxes'][1].set(facecolor=MUT_COLOR, alpha=0.2)

        # Individual dots (one per recording)
        jc = rng.uniform(-0.1, 0.1, len(ctrl))
        jm = rng.uniform(-0.1, 0.1, len(mut))
        ax_dot.scatter(jc, ctrl, c=CTRL_COLOR, s=35, alpha=0.7,
                       edgecolors='white', linewidths=0.5, zorder=5)
        ax_dot.scatter(1 + jm, mut, c=MUT_COLOR, s=35, alpha=0.7,
                       edgecolors='white', linewidths=0.5, zorder=5)

        # Means with error bars
        ax_dot.errorbar(0, np.mean(ctrl), yerr=np.std(ctrl, ddof=1)/np.sqrt(len(ctrl)),
                        fmt='D', color=CTRL_COLOR, markersize=8, zorder=10,
                        capsize=4, linewidth=2, markeredgecolor='white')
        ax_dot.errorbar(1, np.mean(mut), yerr=np.std(mut, ddof=1)/np.sqrt(len(mut)),
                        fmt='D', color=MUT_COLOR, markersize=8, zorder=10,
                        capsize=4, linewidth=2, markeredgecolor='white')

        ax_dot.set_xticks([0, 1])
        ax_dot.set_xticklabels([f'Control\n(n={len(ctrl)})',
                                f'Mutant\n(n={len(mut)})'], fontsize=9)
        ax_dot.set_ylabel(label, fontsize=9)
        ax_dot.spines['top'].set_visible(False)
        ax_dot.spines['right'].set_visible(False)
        ax_dot.grid(axis='y', alpha=0.15)

        # Significance bracket
        p_best = perm.get('p', 1.0)
        rec_lmm = mr.get('recording_lmm', {})
        if isinstance(rec_lmm.get('p'), float) and not np.isnan(rec_lmm['p']):
            p_best = rec_lmm['p']
        stars = '***' if p_best < 0.001 else '**' if p_best < 0.01 else '*' if p_best < 0.05 else 'ns'
        ctrl_fin = ctrl[np.isfinite(ctrl)]
        mut_fin  = mut[np.isfinite(mut)]
        if len(ctrl_fin) == 0 or len(mut_fin) == 0:
            ax_dot.set_ylim(top=1.0)
        else:
            ymax = max(float(np.max(ctrl_fin)), float(np.max(mut_fin)))
            bracket_y = ymax * 1.05
            ax_dot.plot([0, 0, 1, 1], [bracket_y * 0.98, bracket_y, bracket_y, bracket_y * 0.98],
                        color='#333', linewidth=1)
            ax_dot.text(0.5, bracket_y * 1.02, f'{stars} p={p_best:.3f}',
                        ha='center', fontsize=9, fontweight='bold' if p_best < 0.05 else 'normal')
            ax_dot.set_ylim(top=bracket_y * 1.12)

        if row == 0:
            ax_dot.set_title('Recording-Level Comparison\n(each dot = one recording)',
                             fontsize=10, fontweight='bold')

        # ── Right panel: method comparison (coefficients in raw units) ───
        ax_meth = axes[row, 1]
        ax_meth.set_facecolor('white')

        methods = []
        # 1. Permutation
        if perm and not perm.get('skipped'):
            methods.append({
                'name': 'Permutation\n(recording means)',
                'coef': perm['observed_diff'],
                'ci_low': None, 'ci_high': None,
                'p': perm['p'],
                'color': '#2E86AB',
            })

        # 2. Recording-level LMM
        rec_lmm = mr.get('recording_lmm', {})
        if isinstance(rec_lmm.get('coef'), (int, float)):
            methods.append({
                'name': f'LMM recording\n(1|day, n={rec_lmm.get("n_groups","")})',
                'coef': rec_lmm['coef'],
                'ci_low': rec_lmm.get('ci_low'),
                'ci_high': rec_lmm.get('ci_high'),
                'p': rec_lmm.get('p', 1),
                'color': '#A23B72',
            })

        # 3. Neuron-level LMM
        neu_lmm = mr.get('neuron_lmm', {})
        if isinstance(neu_lmm.get('coef'), (int, float)):
            methods.append({
                'name': f'LMM neuron\n(1|recording)',
                'coef': neu_lmm['coef'],
                'ci_low': neu_lmm.get('ci_low'),
                'ci_high': neu_lmm.get('ci_high'),
                'p': neu_lmm.get('p', 1),
                'color': '#F18F01',
            })

        y_positions = list(range(len(methods)))
        for y, m in zip(y_positions, methods):
            coef = m['coef']
            p_val = m['p']
            color = m['color']

            ax_meth.plot(coef, y, 'o', color=color, markersize=9, zorder=5)

            if m['ci_low'] is not None and m['ci_high'] is not None:
                ci_l, ci_h = m['ci_low'], m['ci_high']
                # Clip extreme CIs for readability
                plot_range = max(abs(coef) * 5, 0.01)
                ci_l_clip = max(ci_l, -plot_range)
                ci_h_clip = min(ci_h, plot_range)
                ax_meth.plot([ci_l_clip, ci_h_clip], [y, y], '-',
                             color=color, linewidth=2.5, alpha=0.6, zorder=4)

            stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            p_str = f'p<.001' if p_val < 0.001 else f'p={p_val:.3f}'
            ax_meth.text(coef, y + 0.25, f'{stars} {p_str}',
                         ha='center', fontsize=8, color=color,
                         fontweight='bold' if p_val < 0.05 else 'normal')

        ax_meth.axvline(0, color='#999', linewidth=1, linestyle='--', zorder=1)
        ax_meth.set_yticks(y_positions)
        ax_meth.set_yticklabels([m['name'] for m in methods], fontsize=8)
        ax_meth.set_xlabel(f'Genotype effect ({label.split("(")[0].strip()} units)',
                           fontsize=8)
        ax_meth.spines['top'].set_visible(False)
        ax_meth.spines['right'].set_visible(False)
        ax_meth.grid(axis='x', alpha=0.15)
        ax_meth.invert_yaxis()

        if row == 0:
            ax_meth.set_title('Effect Estimates by Method\n(Mutant − Control, raw units)',
                              fontsize=10, fontweight='bold')

    fig.suptitle('Genotype Effect — Multi-Method Analysis',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.text(0.5, -0.01,
             'Left: each dot is one recording (avoids pseudoreplication)  |  '
             'Right: coefficient = Mutant − Control difference  |  '
             'LMM = linear mixed model controlling for session effects',
             ha='center', fontsize=7, color='#777', style='italic')
    plt.tight_layout()
    path = os.path.join(geno_dir, 'genotype_mixed_model_forest.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Mixed model panel figure saved: {path}")

    # ── Summary table figure ─────────────────────────────────────────────
    fig_t = plt.figure(figsize=(16, max(4, len(metrics) * 1.2 + 2)))
    fig_t.patch.set_facecolor('white')
    ax_t = fig_t.add_subplot(111)
    ax_t.axis('off')

    headers = ['Metric',
               'Control\nmean ± SD', 'Mutant\nmean ± SD',
               'Permutation\np (d)',
               'LMM recording\np (coef)',
               'LMM neuron\np (coef)']

    table_data = []
    for metric, label in metrics:
        mr = mixed_results.get(metric, {})
        perm = mr.get('permutation', {})
        rec = mr.get('recording_lmm', {})
        neu = mr.get('neuron_lmm', {})

        ctrl_mean = perm.get('ctrl_mean', 0)
        ctrl_sd = perm.get('ctrl_sd', 0)
        mut_mean = perm.get('mut_mean', 0)
        mut_sd = perm.get('mut_sd', 0)

        row = [
            label,
            f'{ctrl_mean:.3f} ± {ctrl_sd:.3f}',
            f'{mut_mean:.3f} ± {mut_sd:.3f}',
        ]

        # Permutation
        if perm and not perm.get('skipped'):
            row.append(f'p={perm["p"]:.3f} (d={perm["cohens_d"]:.2f})')
        else:
            row.append('—')

        # Recording LMM
        if isinstance(rec.get('p'), float) and not np.isnan(rec['p']):
            row.append(f'p={rec["p"]:.3f} ({rec["coef"]:.4f})')
        elif rec.get('error'):
            row.append(f'failed')
        elif rec.get('skipped'):
            row.append(str(rec['skipped'])[:20])
        else:
            row.append('—')

        # Neuron LMM
        if isinstance(neu.get('p'), float) and not np.isnan(neu['p']):
            row.append(f'p={neu["p"]:.3f} ({neu["coef"]:.4f})')
        elif neu.get('error'):
            row.append(f'failed')
        else:
            row.append('—')

        table_data.append(row)

    table = ax_t.table(cellText=table_data, colLabels=headers,
                       cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    for col in range(len(headers)):
        cell = table[0, col]
        cell.set_facecolor('#E6F1FB')
        cell.set_text_props(color='white', fontweight='bold', fontsize=7)

    for row_idx, row in enumerate(table_data, start=1):
        for col_idx in range(len(row)):
            cell = table[row_idx, col_idx]
            cell.set_edgecolor('#DDD')
            cell.set_facecolor('#F8F8F8' if row_idx % 2 == 0 else 'white')
            # Highlight significant p-values
            try:
                text = row[col_idx]
                if 'p=' in text:
                    p_str = text.split('p=')[1].split(' ')[0].split(')')[0]
                    if float(p_str) < 0.05:
                        cell.set_facecolor('#D4EDDA')
                        cell.set_text_props(fontweight='bold')
            except (ValueError, IndexError):
                pass

    fig_t.suptitle('Genotype Effect — Statistical Summary',
                   fontsize=13, fontweight='bold', y=0.98)
    fig_t.text(0.5, 0.02,
               'Permutation: shuffles genotype labels across recording means (5000 iterations)  |  '
               'LMM recording: genotype fixed + (1|organoid_day)  |  '
               'LMM neuron: genotype fixed + (1|recording)',
               ha='center', fontsize=7, color='#777', style='italic')

    tpath = os.path.join(geno_dir, 'genotype_methods_comparison.png')
    plt.savefig(tpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Methods comparison table saved: {tpath}")


def run_between_organoid_tests(datasets: List[DatasetMetrics], output_dir: str) -> dict:
    """
    Statistical comparison BETWEEN organoids.

    Groups recordings by organoid ID, pools neurons across recordings
    within each organoid, then compares between organoids.

    Per-neuron metrics (spike rate):
        - Kruskal-Wallis across organoids, pairwise Mann-Whitney U

    Per-recording metrics (correlation, synchrony):
        - Each organoid has multiple recordings → can run Mann-Whitney
          between organoids on these distributions

    Only runs if there are 2+ distinct organoids. Returns empty dict
    and skips figure generation if only one organoid is present.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats as sp_stats
    from collections import OrderedDict

    results = {'level': 'between-organoid', 'tests': {}}

    # ── Group datasets by organoid ───────────────────────────────────────
    organoid_map = OrderedDict()  # organoid_id → list of DatasetMetrics
    for ds in datasets:
        org_id = _extract_organoid_id(ds.name)
        if org_id not in organoid_map:
            organoid_map[org_id] = []
        organoid_map[org_id].append(ds)

    org_ids = list(organoid_map.keys())
    # Sort by numeric day value (e.g. D109 → 109) so age groups are ascending
    def _day_sort_key(oid):
        import re
        m = re.search(r'\d+', oid)
        return int(m.group()) if m else 0
    org_ids.sort(key=_day_sort_key)
    # Rebuild map in sorted order
    organoid_map = OrderedDict((oid, organoid_map[oid]) for oid in org_ids)
    n_org = len(org_ids)
    results['n_organoids'] = n_org
    results['organoids'] = {
        oid: [ds.name for ds in dsets] for oid, dsets in organoid_map.items()
    }

    if n_org < 2:
        logger.info(f"Between-organoid tests: only {n_org} organoid(s) found, skipping.")
        results['skipped'] = True
        results['reason'] = f'Only {n_org} organoid(s) — need at least 2 for comparison.'
        return results

    logger.info(f"Between-organoid comparison: {n_org} organoids "
                f"({', '.join(f'{oid} ({len(dsets)} recs)' for oid, dsets in organoid_map.items())})")

    # ── Pool per-neuron spike rates and transient amplitudes by organoid ─────
    # Descriptive view: shows full distribution of individual neuron activity
    # across organoid days. Not used for statistical testing.
    org_rates = OrderedDict()
    org_amplitudes = OrderedDict()
    for oid, dsets in organoid_map.items():
        pooled_rates = []
        pooled_amps = []
        for ds in dsets:
            rates = _get_neuron_rates(ds)
            amps = _get_neuron_amplitudes(ds)
            pooled_rates.extend(rates)
            pooled_amps.extend(amps)
        
        # Only include organoids that have at least one active neuron
        if len(pooled_rates) > 0:
            org_rates[oid] = np.array(pooled_rates)
        if len(pooled_amps) > 0:
            org_amplitudes[oid] = np.array(pooled_amps)

    # Update org_ids to only include organoids with data
    org_ids = [oid for oid in organoid_map.keys() if oid in org_rates]
    n_org = len(org_ids)
    
    if n_org < 2:
        logger.warning(f"Only {n_org} organoids with active neurons — skipping between-organoid tests")
        return results

    # ── Pool per-recording metrics by organoid (active organoids only) ──
    org_corr = OrderedDict()
    org_sync = OrderedDict()
    for oid in org_ids:
        dsets = organoid_map[oid]
        _oc = np.array([ds.pairwise_correlation_mean for ds in dsets])
        _os = np.array([ds.synchrony_index for ds in dsets])
        org_corr[oid] = _oc[np.isfinite(_oc)]
        org_sync[oid] = _os[np.isfinite(_os)]

    # ── Statistical tests ────────────────────────────────────────────────
    def _run_tests(data_dict, metric_name, level):
        """Run KW + pairwise MW on a dict of {group: array}."""
        ids = list(data_dict.keys())
        arrays = [data_dict[k] for k in ids]
        test_result = {
            'metric': metric_name, 'level': level,
            'per_organoid': {},
        }

        for oid, arr in zip(ids, arrays):
            test_result['per_organoid'][oid] = {
                'n': len(arr),
                'mean': float(np.mean(arr)) if len(arr) > 0 else 0,
                'median': float(np.median(arr)) if len(arr) > 0 else 0,
                'sd': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0,
            }

        # Kruskal-Wallis (store in JSON but don't display prominently)
        valid = [a for a in arrays if len(a) >= 2]
        if len(valid) >= 2:
            H, p_kw = sp_stats.kruskal(*valid)
            test_result['kruskal_wallis'] = {'H': float(H), 'p': float(p_kw)}

        # Pairwise Mann-Whitney
        n_pairs = len(ids) * (len(ids) - 1) // 2
        pairwise = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if len(arrays[i]) < 2 or len(arrays[j]) < 2:
                    continue
                U, p_mw = sp_stats.mannwhitneyu(
                    arrays[i], arrays[j], alternative='two-sided')
                p_corr = min(p_mw * max(n_pairs, 1), 1.0)
                pairwise.append({
                    'i': i, 'j': j,
                    'a': ids[i], 'b': ids[j],
                    'U': float(U), 'p_raw': float(p_mw), 'p_bonf': float(p_corr),
                })
        test_result['pairwise'] = pairwise
        return test_result

    rate_tests = _run_tests(org_rates, 'Event rate (events/10s)', 'per-neuron, pooled')
    amp_tests = _run_tests(org_amplitudes, 'Transient amplitude (ΔF/F₀)', 'per-neuron, pooled')
    corr_tests = _run_tests(org_corr, 'Pairwise Correlation (r)', 'per-recording')
    sync_tests = _run_tests(org_sync, 'Synchrony Index', 'per-recording')

    results['tests']['spike_rate'] = rate_tests
    results['tests']['spike_amplitude'] = amp_tests
    results['tests']['pairwise_correlation'] = corr_tests
    results['tests']['synchrony_index'] = sync_tests

    # =====================================================================
    # FIGURE: Between-organoid comparison (spike rate only)
    # =====================================================================
    BG      = 'white'
    TEXT    = '#333333'
    GRID    = '#DDDDDD'
    MEDIAN  = '#E53935'

    fig_width = max(14, n_org * 1.8)
    fig = plt.figure(figsize=(fig_width, 9))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(1, 1, left=0.08, right=0.97, top=0.92, bottom=0.10)

    # Colour palette — slightly brighter for dark background
    palette = ['#5B8FD4', '#F0923B', '#7DC460', '#FFD04A', '#74B8E8',
               '#B0B0B0', '#4A7ABF', '#C04050']
    org_colors = {oid: palette[i % len(palette)] for i, oid in enumerate(org_ids)}

    rng = np.random.default_rng(42)

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(BG)

    rate_arrays = [org_rates[oid] for oid in org_ids]
    bp = ax.boxplot(
        rate_arrays, positions=range(n_org), widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops=dict(color=MEDIAN, linewidth=2.0),
        whiskerprops=dict(color=TEXT, linewidth=0.8, alpha=0.5),
        capprops=dict(color=TEXT, linewidth=0.8, alpha=0.5),
    )
    for i, patch in enumerate(bp['boxes']):
        col = org_colors[org_ids[i]]
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.2)

    # Compute 3-SD outlier threshold across all organoids (used for y-axis limits
    # and scatter display only — boxplot statistics use the full unclipped data)
    all_rates_flat = np.concatenate([org_rates[oid] for oid in org_ids if len(org_rates[oid]) > 0])
    _global_mean = float(np.mean(all_rates_flat)) if len(all_rates_flat) > 1 else 0.0
    _global_sd   = float(np.std(all_rates_flat, ddof=1)) if len(all_rates_flat) > 1 else 1.0
    _y_clip      = _global_mean + 3.0 * _global_sd  # upper clip boundary

    n_zero_total     = int(np.sum(all_rates_flat == 0))
    n_excluded_total = int(np.sum(all_rates_flat > _y_clip))

    # Overlay individual neuron dots — zeros and >3 SD outliers excluded from
    # display (inactive neurons add no trend information; outliers compress scale)
    all_rates_display = all_rates_flat[(all_rates_flat > 0) & (all_rates_flat <= _y_clip)]
    y_range = np.ptp(all_rates_display) if len(all_rates_display) > 1 else 1.0
    y_jitter_scale = y_range * 0.015

    for i, (oid, rates) in enumerate(org_rates.items()):
        mask = (rates > 0) & (rates <= _y_clip)
        rates_plot = rates[mask]
        n_pts = len(rates_plot)
        if n_pts == 0:
            continue
        if n_pts == 1:
            x_pts = np.array([i])
            y_pts = rates_plot
        else:
            base = np.linspace(-0.22, 0.22, n_pts)
            rng.shuffle(base)
            noise = rng.uniform(-0.02, 0.02, n_pts)
            x_pts = i + base + noise
            y_pts = rates_plot + rng.uniform(-y_jitter_scale, y_jitter_scale, n_pts)
        ax.scatter(x_pts, y_pts, color=org_colors[oid], s=10,
                   alpha=0.45, zorder=5, linewidths=0, edgecolors='none')

    ax.set_xticks(range(n_org))
    ax.set_xticklabels(org_ids, fontsize=11, fontweight='bold',
                       color=TEXT,
                       rotation=45 if n_org > 6 else 0,
                       ha='right' if n_org > 6 else 'center')
    ax.set_ylabel('Event rate (events/10s)', fontsize=12, color=TEXT)
    # Linear y-axis; clip view to 3 SD above global mean, with a small top margin
    ax.set_ylim(bottom=0, top=_y_clip * 1.08)
    ax.tick_params(colors=TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.6, color=GRID)

    _excl_parts = []
    if n_zero_total > 0:
        _excl_parts.append(f'{n_zero_total} inactive (zero-rate) neuron(s) not shown')
    if n_excluded_total > 0:
        _excl_parts.append(f'{n_excluded_total} point(s) >3 SD above mean not shown')
    _excl_note = ('  ·  ' + '  ·  '.join(_excl_parts)) if _excl_parts else ''
    fig.text(0.5, 0.02,
             f'Descriptive overview: Active neurons only (zero-rate excluded){_excl_note}',
             ha='center', fontsize=8, color='#5A7A8A', style='italic')
    fig.suptitle('Pooled event rates by age',
                 fontsize=14, fontweight='bold', color=TEXT, y=0.97)

    organoid_dir = os.path.join(output_dir, 'figures', '1 - Main Results')
    os.makedirs(organoid_dir, exist_ok=True)
    path = os.path.join(organoid_dir, 'between_organoid_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # =====================================================================
    # FIGURE: Transient amplitude by organoid (same style as panel A)
    # =====================================================================
    fig_amp, ax_amp = plt.subplots(figsize=(max(8, n_org * 1.2), 6))
    fig_amp.patch.set_facecolor('white')
    ax_amp.set_facecolor('white')

    amp_arrays = [org_amplitudes[oid] for oid in org_ids]
    bp_amp = ax_amp.boxplot(
        amp_arrays, positions=range(n_org), widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops=dict(color='#CC3333', linewidth=1.5),
        whiskerprops=dict(color='#555555', linewidth=0.8),
        capprops=dict(color='#555555', linewidth=0.8),
    )
    for i, patch in enumerate(bp_amp['boxes']):
        col = org_colors[org_ids[i]]
        patch.set_facecolor(col)
        patch.set_alpha(0.25)
        patch.set_edgecolor(col)
        patch.set_linewidth(0.8)

    # Overlay individual neurons
    for i, (oid, amps) in enumerate(org_amplitudes.items()):
        if len(amps) > 0:
            jitter = rng.uniform(-0.18, 0.18, len(amps))
            ax_amp.scatter(i + jitter, amps, color=org_colors[oid], s=8,
                          alpha=0.3, zorder=5, linewidths=0, edgecolors='none')

    ax_amp.set_xticks(range(n_org))
    ax_amp.set_xticklabels(org_ids, fontsize=10, fontweight='bold',
                           rotation=45 if n_org > 6 else 0, 
                           ha='right' if n_org > 6 else 'center')
    # Color x-tick labels
    for i, (tick_label, oid) in enumerate(zip(ax_amp.get_xticklabels(), org_ids)):
        tick_label.set_color(org_colors[oid])
    
    ax_amp.set_ylabel('Transient amplitude (ΔF/F₀)', fontsize=11)
    ax_amp.set_xlabel('Organoid', fontsize=10)
    ax_amp.grid(axis='y', alpha=0.15)
    ax_amp.spines['top'].set_visible(False)
    ax_amp.spines['right'].set_visible(False)

    ax_amp.set_title('Transient amplitude by organoid', fontsize=13, fontweight='bold')
    
    fig_amp.text(0.5, 0.01,
                 'Descriptive overview — per-neuron spike amplitudes  ·  Each dot = one neuron',
                 ha='center', fontsize=7, color='#777777', style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    amp_path = os.path.join(organoid_dir, 'spike_amplitude_by_organoid.png')
    plt.savefig(amp_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Spike amplitude figure saved: {amp_path}")

    # =====================================================================
    # TABLE: Between-organoid pairwise p-values
    # =====================================================================
    # Scale figure height with number of organoids
    table_height = max(12, 10 + n_org * 0.8)
    fig2 = plt.figure(figsize=(16, table_height))
    fig2.patch.set_facecolor('white')

    # Use fig.text() with carefully calculated y-coordinates for proper spacing
    # Calculate vertical spacing based on number of organoids
    summary_table_height = 0.08 + n_org * 0.018  # Height needed for summary table
    pvalue_table_height = 0.06 + n_org * 0.022   # Height for each p-value matrix
    
    # Position from top to bottom with explicit spacing
    y_title = 0.97
    y_summary_title = 0.93
    y_summary_table_top = 0.90
    summary_table_bottom = y_summary_table_top - summary_table_height
    
    # Calculate spacing for the three p-value tables
    remaining_space = summary_table_bottom - 0.04  # Leave margin at bottom
    table_spacing = 0.04  # Gap between tables
    single_table_block = (remaining_space - 2 * table_spacing) / 3  # Space for each table + title

    fig2.text(0.5, y_title, 'Between-Organoid Statistical Tests',
              ha='center', fontsize=14, fontweight='bold')

    # --- Table A: Descriptive stats per organoid ---
    ax_t1 = fig2.add_axes([0.05, summary_table_bottom, 0.90, summary_table_height])
    ax_t1.axis('off')

    fig2.text(0.5, y_summary_title, 'Per-Organoid Summary',
              ha='center', fontsize=11, fontweight='bold')

    headers1 = ['Organoid', 'Recordings', 'Neurons', 'Rate mean±SD',
                'Corr mean±SD', 'Sync mean±SD']
    rows1 = []
    for oid in org_ids:
        rates = org_rates[oid]
        corrs = org_corr[oid]
        syncs = org_sync[oid]
        rows1.append([
            oid, str(len(organoid_map[oid])), str(len(rates)),
            f'{np.mean(rates):.3f} ± {np.std(rates, ddof=1):.3f}' if len(rates) > 1 else (f'{np.mean(rates):.3f}' if len(rates) == 1 else '—'),
            f'{np.mean(corrs):.4f} ± {np.std(corrs, ddof=1):.4f}' if len(corrs) > 1 else (f'{np.mean(corrs):.4f}' if len(corrs) == 1 else '—'),
            f'{np.mean(syncs):.4f} ± {np.std(syncs, ddof=1):.4f}' if len(syncs) > 1 else (f'{np.mean(syncs):.4f}' if len(syncs) == 1 else '—'),
        ])

    t1 = ax_t1.table(cellText=rows1, colLabels=headers1,
                      loc='center', cellLoc='center')
    t1.auto_set_font_size(False)
    t1.set_fontsize(8)
    t1.scale(1, 1.5)
    for (row, col), cell in t1.get_celld().items():
        if row == 0:
            cell.set_facecolor('#E6F1FB')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            i = row - 1
            cell.set_facecolor('#F2F7FC' if row % 2 == 0 else 'white')
        cell.set_edgecolor('#DDDDDD')

    # --- Tables B/C/D: Pairwise p-values for each metric ---
    metrics_for_table = [
        ('spike_rate', 'Spike Rate — Pairwise p-values (Mann-Whitney U, Bonferroni)', rate_tests),
        ('pairwise_correlation', 'Pairwise Correlation — Pairwise p-values', corr_tests),
        ('synchrony_index', 'Synchrony Index — Pairwise p-values', sync_tests),
    ]

    # Calculate positions for three p-value tables, evenly spaced below summary
    # Each table block = title + table content
    title_offset = 0.03  # Space between title and table
    for idx_m, (mkey, mtitle, mtest) in enumerate(metrics_for_table):
        # Calculate this table's vertical position
        block_top = summary_table_bottom - table_spacing - idx_m * (single_table_block + table_spacing)
        table_title_y = block_top
        table_bottom = block_top - title_offset - pvalue_table_height
        
        ax_m = fig2.add_axes([0.10, table_bottom, 0.80, pvalue_table_height])
        ax_m.axis('off')
        fig2.text(0.5, table_title_y, mtitle,
                  ha='center', fontsize=10, fontweight='bold')

        # Build pairwise matrix
        p_mat = np.ones((n_org, n_org))
        for pair in mtest.get('pairwise', []):
            p_mat[pair['i'], pair['j']] = pair['p_bonf']
            p_mat[pair['j'], pair['i']] = pair['p_bonf']

        col_h = [''] + org_ids
        rows_m = []
        for i in range(n_org):
            row_data = [org_ids[i]]
            for j in range(n_org):
                if i == j:
                    row_data.append('—')
                else:
                    p = p_mat[i, j]
                    row_data.append(f'{_fmt_p(p)} {_sig_stars(p)}')
            rows_m.append(row_data)

        tm = ax_m.table(cellText=rows_m, colLabels=col_h,
                         loc='center', cellLoc='center')
        tm.auto_set_font_size(False)
        tm.set_fontsize(8)
        tm.scale(1, 1.5)

        for (row, col), cell in tm.get_celld().items():
            if row == 0:
                cell.set_facecolor('#E6F1FB')
                cell.set_text_props(color='white', fontweight='bold')
            elif col == 0 and row > 0:
                cell.set_facecolor('#E8EFF7')
                cell.set_text_props(fontweight='bold')
            else:
                r, c = row - 1, col - 1
                if r == c:
                    cell.set_facecolor('#E0E0E0')
                else:
                    p = p_mat[r, c] if 0 <= r < n_org and 0 <= c < n_org else 1.0
                    if p < 0.001:
                        cell.set_facecolor('#C6EFCE')
                        cell.set_text_props(color='#006100', fontweight='bold')
                    elif p < 0.01:
                        cell.set_facecolor('#D9F2D9')
                        cell.set_text_props(color='#006100')
                    elif p < 0.05:
                        cell.set_facecolor('#FFEB9C')
                        cell.set_text_props(color='#9C6500')
                    else:
                        cell.set_facecolor('white')
                        cell.set_text_props(color='#999999')
            cell.set_edgecolor('#CCCCCC')

    table_path = os.path.join(organoid_dir, 'between_organoid_table.png')
    plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Between-organoid figures saved: {path}, {table_path}")
    return results



def generate_figures(
    datasets: List[DatasetMetrics],
    X: np.ndarray,
    feat_labels: List[str],
    names: List[str],
    output_dir: str,
) -> List[str]:
    """Generate all analysis figures with organised directory structure.

    Output layout:
        figures/
        ├── overview/               Combined multi-dataset views
        ├── per_dataset/            Individual dataset detail
        │   ├── D109_R1/
        │   ...
        └── per_metric/             Individual bar chart metrics
    
    New structure (v1.3):
        figures/
        ├── overview/
        │   ├── summary/            High-level summaries (UMAP, heatmap)
        │   ├── statistics/         Statistical test results
        │   └── quality/            Quality gating, neuron selection
        ├── by_organoid/            Between-organoid comparisons
        ├── by_metric/              Individual metric plots
        ├── correlations/           Correlation matrices (one per dataset)
        ├── rasters/                Raster plots (one per dataset)
        ├── traces/                 Trace comparisons
        └── per_dataset/            All per-dataset outputs
            ├── D109_R1/
            ...
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Create new directory structure matching user preferences
    base_dir = output_dir
    dirs = {
        'main_results': os.path.join(base_dir, '1 - Main Results'),
        'metrics': os.path.join(base_dir, '1b - Metrics'),
        'correlations': os.path.join(base_dir, 'Correlation Graphs'),
        'overview': os.path.join(base_dir, 'Full Overview'),
        'per_dataset': os.path.join(base_dir, 'Results by Dataset'),
        'temporal': os.path.join(base_dir, 'Temporal Visualisations'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    paths = []
    n_ds = len(datasets)
    default_color = '#5B8DBE'
    labels = np.ones(n_ds, dtype=int)
    colors = {1: default_color}

    # ── Main Results figures ───────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(X)
    paths.append(_fig_feature_heatmap(X_std, names, feat_labels, labels, dirs['main_results']))
    paths.append(_fig_neuron_distributions(datasets, labels, colors, dirs['overview']))
    
    # ── Per-dataset figures (split into individual files) ─────────────────
    paths.extend(_fig_raster_plots_split(datasets, labels, colors, dirs['temporal'], dirs['per_dataset']))
    paths.extend(_fig_correlation_matrices_split(datasets, labels, colors, dirs['correlations'], dirs['per_dataset']))
    paths.append(_fig_population_activity(datasets, labels, colors, dirs['overview']))
    
    # ── Per-metric figures ────────────────────────────────────────────────
    paths.extend(_fig_bar_charts(datasets, labels, colors, dirs['overview'], dirs['metrics']))

    logger.info(f"Generated {len(paths)} figures in {output_dir}")
    return paths


def _fig_feature_heatmap(X_std, names, feat_labels, labels, output_dir):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage

    abbrevs = [_abbrev(n) for n in names]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.5), 8))

    # Reorder by cluster then by linkage within cluster
    order = np.argsort(labels)
    X_ordered = X_std[order]
    names_ordered = [abbrevs[i] for i in order]
    labels_ordered = labels[order]

    im = ax.imshow(X_ordered.T, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(names_ordered)))
    ax.set_xticklabels(names_ordered, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels, fontsize=8)
    ax.set_title('Standardised Feature Heatmap', fontweight='bold')

    # Dataset separators (cosmetic)
    prev = labels_ordered[0]
    for i in range(1, len(labels_ordered)):
        if labels_ordered[i] != prev:
            ax.axvline(i - 0.5, color='white', linewidth=2)
            prev = labels_ordered[i]

    plt.colorbar(im, ax=ax, label='Z-score', shrink=0.6)
    plt.tight_layout()
    path = os.path.join(output_dir, 'feature_heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def _fig_neuron_distributions(datasets, labels, colors, output_dir):
    """Per-neuron distributions from selected neurons, pooled across all datasets."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ('spike_rate', 'Event rate (events/10s)'),
        ('quality', 'Quality Score'),
        ('spike_amplitude', 'Mean Transient amplitude (dF/F)'),
    ]

    for ax, (metric_key, title) in zip(axes, metrics):
        pooled = []
        for d in datasets:
            if metric_key == 'spike_rate':
                pooled.extend(_get_neuron_rates(d))
            elif metric_key == 'quality':
                if d.selected_quality is not None:
                    pooled.extend(d.selected_quality)
            elif metric_key == 'spike_amplitude':
                pooled.extend(_get_neuron_amplitudes(d))

        if len(pooled) == 0:
            continue
        sorted_vals = np.sort(pooled)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, color='#5B8DBE', linewidth=2,
                label=f'All datasets (n={len(pooled)})')

        ax.set_xlabel(title)
        ax.set_ylabel('Cumulative Fraction')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('Per-Neuron Distributions (Selected Neurons)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'neuron_distributions.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def _fig_raster_plots_split(datasets, labels, colors, raster_dir, per_dataset_dir):
    """
    Generate individual raster plot figures for each dataset.
    
    Outputs:
    - rasters/<dataset_name>.png — individual raster plots
    - per_dataset/<n>/raster.png — detailed raster in per-dataset folder
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    os.makedirs(raster_dir, exist_ok=True)
    paths = []
    
    for i, ds in enumerate(datasets):
        result_path = Path(ds.filepath)
        spikes_path = result_path / 'spike_trains.npy'
        denoised_path = result_path / 'traces_denoised.npy'
        conf_path = result_path / 'confidence_scores.npy'
        
        if not spikes_path.exists():
            continue
        
        S_all = np.load(spikes_path)
        
        # Exclude edge ROIs
        valid = _load_valid_mask(result_path)
        if valid is not None and len(valid) == S_all.shape[0]:
            valid_idx = np.where(valid)[0]
        else:
            conf = np.load(conf_path) if conf_path.exists() else np.ones(S_all.shape[0])
            valid_idx = np.where(conf >= 0.5)[0]
        
        S = S_all[valid_idx]
        N, T = S.shape
        t_ax = np.arange(T) / ds.frame_rate
        ds_col = "#5B8DBE"
        
        # Load denoised traces for example panel
        C_den = None
        C_raw_valid = None
        if denoised_path.exists():
            C_all = np.load(denoised_path)
            C_den = C_all[valid_idx]
        raw_path = result_path / 'temporal_traces.npy'
        if raw_path.exists():
            C_raw_valid = np.load(raw_path)[valid_idx]
        
        # Full detailed raster figure
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('white')
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 3], hspace=0.3)
        
        # 1. Raster
        ax_raster = fig.add_subplot(gs[0])
        spike_counts = np.array([np.sum(S[j] > 0) for j in range(N)])
        order = np.argsort(spike_counts)[::-1]
        
        for row_i, neuron_idx in enumerate(order):
            spike_times = np.where(S[neuron_idx] > 0)[0] / ds.frame_rate
            ax_raster.scatter(spike_times, np.full_like(spike_times, row_i),
                              s=0.5, c='black', marker='|', linewidths=0.5)
        
        ax_raster.set_ylim(-0.5, N - 0.5)
        ax_raster.set_xlim(0, t_ax[-1])
        ax_raster.set_ylabel('Neuron', fontsize=10)
        ax_raster.set_title(f'{_abbrev(ds.name)} — Spike Raster ({N} neurons)',
                            fontsize=12, fontweight='bold', color=ds_col)
        ax_raster.tick_params(labelsize=8)
        
        # 2. Population rate
        ax_pop = fig.add_subplot(gs[1], sharex=ax_raster)
        bin_width = max(0.5, t_ax[-1] / 200)
        bins = np.arange(0, t_ax[-1] + bin_width, bin_width)
        pop_counts = np.zeros(len(bins) - 1)
        for j in range(N):
            spike_times_j = np.where(S[j] > 0)[0] / ds.frame_rate
            hist, _ = np.histogram(spike_times_j, bins=bins)
            pop_counts += hist
        bin_centres = (bins[:-1] + bins[1:]) / 2
        ax_pop.fill_between(bin_centres, pop_counts / N, color=ds_col, alpha=0.5)
        ax_pop.plot(bin_centres, pop_counts / N, color=ds_col, linewidth=0.8)
        ax_pop.set_ylabel('Mean spikes\nper neuron', fontsize=9)
        ax_pop.set_xlabel('Time (s)', fontsize=10)
        ax_pop.tick_params(labelsize=8)
        
        # 3. Example traces (top 6 by robust quality)
        ax_traces = fig.add_subplot(gs[2], sharex=ax_raster)
        n_examples = min(6, N)
        if C_den is not None and C_den.shape[0] >= n_examples:
            quality = np.array([_trace_snr(C_den[j]) for j in range(C_den.shape[0])])
            top_idx = np.argsort(quality)[::-1][:n_examples]
            
            offsets = np.arange(n_examples) * 1.2
            for plot_i, idx in enumerate(top_idx):
                trace_norm = C_den[idx]
                r = np.percentile(trace_norm, 99) - np.percentile(trace_norm, 1)
                if r > 0:
                    trace_norm = (trace_norm - np.percentile(trace_norm, 1)) / r
                ax_traces.plot(t_ax[:len(trace_norm)],
                               trace_norm + offsets[plot_i],
                               color='#00e676', linewidth=0.6)
                # Spike markers
                spk = np.where(S[idx] > 0)[0]
                if len(spk) > 0:
                    ax_traces.scatter(spk / ds.frame_rate,
                                      trace_norm[spk] + offsets[plot_i],
                                      color='red', s=8, zorder=5)
                # ROI label
                orig_idx = valid_idx[idx] if valid is not None else idx
                ax_traces.text(0, offsets[plot_i] + 0.5,
                               f'ROI {orig_idx}', fontsize=7, fontweight='bold',
                               va='center', ha='right')
            
            ax_traces.set_ylabel(f'ΔF/F₀ traces (top {n_examples} by quality)', fontsize=9)
            ax_traces.set_yticks([])
        else:
            ax_traces.text(0.5, 0.5, 'No denoised traces', ha='center',
                           va='center', transform=ax_traces.transAxes)
        ax_traces.set_xlabel('Time (s)', fontsize=10)
        ax_traces.tick_params(labelsize=8)
        ax_traces.set_xlim(0, t_ax[-1])
        
        plt.tight_layout()
        
        # Save to rasters directory
        safe_name = ds.name.replace('/', '_').replace(' ', '_')[:50]
        raster_path = os.path.join(raster_dir, f'{safe_name}.png')
        plt.savefig(raster_path, dpi=200, bbox_inches='tight', facecolor='white')
        paths.append(raster_path)
        
        # Also save to per-dataset directory
        ds_dir = os.path.join(per_dataset_dir, safe_name)
        os.makedirs(ds_dir, exist_ok=True)
        plt.savefig(os.path.join(ds_dir, 'raster.png'), dpi=200, 
                   bbox_inches='tight', facecolor='white')
        
        plt.close()
    
    logger.info(f"Generated {len(paths)} individual raster plots in {raster_dir}")
    return paths


def _fig_correlation_matrices_split(datasets, labels, colors, corr_dir, per_dataset_dir):
    """
    Generate individual correlation matrix figures for each dataset.
    
    Outputs:
    - correlations/<dataset_name>.png  — individual correlation matrices
    - per_dataset/<name>/correlation.png — copy in per-dataset folder
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.cluster.hierarchy import linkage, leaves_list
    
    os.makedirs(corr_dir, exist_ok=True)
    paths = []
    
    for i, ds in enumerate(datasets):
        result_path = Path(ds.filepath)
        
        # Prefer denoised
        denoised_path = result_path / 'traces_denoised.npy'
        raw_path = result_path / 'temporal_traces.npy'
        conf_path = result_path / 'confidence_scores.npy'
        
        if denoised_path.exists():
            C_all = np.load(denoised_path)
        elif raw_path.exists():
            C_all = np.load(raw_path)
        else:
            continue
        
        # Exclude edge ROIs
        valid = _load_valid_mask(result_path)
        if valid is not None and len(valid) == C_all.shape[0]:
            valid_idx = np.where(valid)[0]
            C = C_all[valid_idx]
        else:
            conf = np.load(conf_path) if conf_path.exists() else np.ones(C_all.shape[0])
            valid_idx = np.where(conf >= 0.5)[0]
            C = C_all[valid_idx]
        N = C.shape[0]
        
        if N < 3:
            continue
        
        # Limit to top 100 neurons by robust quality
        if N > 100:
            R = np.load(raw_path)[valid_idx] if raw_path.exists() else None
            quality = np.array([_trace_snr(C[j]) for j in range(C.shape[0])])
            top = np.argsort(quality)[::-1][:100]
            C = C[top]
            N = 100
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(C)
        corr = np.nan_to_num(corr, nan=0.0)
        
        # Sort by hierarchical clustering
        try:
            from scipy.spatial.distance import squareform; dist = np.clip(1 - corr, 0, 2); np.fill_diagonal(dist, 0); Z = linkage(squareform(dist), method='ward')
            order = leaves_list(Z)
        except Exception:
            order = np.arange(N)
        
        corr_sorted = corr[np.ix_(order, order)]
        
        # Create individual figure
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor('white')
        
        im = ax.imshow(corr_sorted, cmap='RdBu_r', vmin=-0.3, vmax=0.8,
                       aspect='equal', interpolation='none')
        ax.set_title(f'{_abbrev(ds.name)}\nPairwise Correlation ({N} neurons)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Neuron #', fontsize=10)
        ax.set_ylabel('Neuron #', fontsize=10)
        ax.tick_params(labelsize=8)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Pearson r', fontsize=10)
        
        # Add mean correlation annotation
        mask = np.triu(np.ones_like(corr), k=1).astype(bool)
        mean_corr = corr[mask].mean()
        ax.text(0.02, 0.98, f'Mean r = {mean_corr:.3f}', 
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save to correlations directory
        safe_name = ds.name.replace('/', '_').replace(' ', '_')[:50]
        corr_path = os.path.join(corr_dir, f'{safe_name}.png')
        plt.savefig(corr_path, dpi=200, bbox_inches='tight', facecolor='white')
        paths.append(corr_path)
        
        # Also save to per-dataset directory
        ds_dir = os.path.join(per_dataset_dir, safe_name)
        os.makedirs(ds_dir, exist_ok=True)
        plt.savefig(os.path.join(ds_dir, 'correlation.png'), dpi=200, 
                   bbox_inches='tight', facecolor='white')
        
        plt.close()
    
    logger.info(f"Generated {len(paths)} individual correlation matrices in {corr_dir}")
    return paths


def _fig_population_activity(datasets, labels, colors, output_dir):
    """Dataset-level metrics as labelled scatter + bar summary with organoid colors."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from collections import OrderedDict

    metrics = [
        ('mean_spike_rate',          'Spike Rate\n(events/10s)'),
        ('mean_spike_amplitude',     'Transient amplitude\n(dF/F)'),
        ('pairwise_correlation_mean','Pairwise\nCorrelation (r)'),
        ('synchrony_index',          'Synchrony\nIndex'),
        ('burst_rate',               'Burst Rate\n(bursts/10s)'),
        ('mean_iei',                 'Inter-Event\nInterval (s)'),
    ]

    # Extract organoid IDs and create color map
    organoid_ids = [_extract_organoid_id(ds.name) for ds in datasets]
    unique_organoids = list(OrderedDict.fromkeys(organoid_ids))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    org_colors = {oid: palette[i % len(palette)] for i, oid in enumerate(unique_organoids)}
    ds_colors = [org_colors[oid] for oid in organoid_ids]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 2.8, 5))
    fig.patch.set_facecolor('white')

    rng = np.random.default_rng(42)

    for ax, (attr, title) in zip(axes, metrics):
        ax.set_facecolor('white')
        vals = [getattr(ds, attr, 0) for ds in datasets]

        mean_val = np.mean(vals) if vals else 0
        ax.bar(0, mean_val, width=0.5, color='#5B8DBE', alpha=0.25,
               edgecolor='#5B8DBE', linewidth=1.5)

        jitter = rng.uniform(-0.15, 0.15, len(vals))
        x_pts = np.zeros(len(vals)) + jitter
        
        # Scatter points colored by organoid
        for i, (x, y, col) in enumerate(zip(x_pts, vals, ds_colors)):
            ax.scatter(x, y, c=col, s=35, edgecolor='white', linewidth=0.5, zorder=5)

        ax.hlines(mean_val, -0.25, 0.25, color='#333333', linewidth=2, zorder=6)

        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add legend for organoids
    legend_elements = [Patch(facecolor=org_colors[oid], edgecolor='white', label=oid) 
                      for oid in unique_organoids]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.95),
              fontsize=7, title='Organoid', title_fontsize=8, framealpha=0.9,
              ncol=min(3, len(unique_organoids)))

    fig.suptitle('Population Activity Summary (selected neurons)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(output_dir, 'population_activity.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


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


def _draw_bar_panel(ax, key, ylabel, has_neurons, per_ds, ds_names, n_ds,
                    bar_width=0.65):
    """Draw a single bar chart panel. Shared by combined and individual figures."""
    from scipy import stats as sp_stats

    BASE_COLOR = '#5B8DBE'
    FLAG_COLOR_FEW = '#B8D4E8'   # light tint when few flagged
    FLAG_COLOR_MANY = '#A3C4D9'  # slightly darker when many flagged

    ax.set_facecolor('white')
    x = np.arange(n_ds)
    means_clean, sems_clean = [], []

    for d in per_ds:
        if has_neurons and len(d[key]) > 0:
            clean_mask = ~d['flagged']
            clean_vals = d[key][clean_mask]
            if len(clean_vals) > 0:
                means_clean.append(float(np.mean(clean_vals)))
                sems_clean.append(float(sp_stats.sem(clean_vals))
                                  if len(clean_vals) > 1 else 0)
            else:
                means_clean.append(float(np.mean(d[key])))
                sems_clean.append(0)
        else:
            means_clean.append(float(d[key]))
            sems_clean.append(0)

    # Bar colours: subtle tint shift for flagged datasets
    bar_colors = []
    edge_colors = []
    for i in range(n_ds):
        n_flag = per_ds[i]['n_flagged']
        if n_flag > 0:
            n_total = len(per_ds[i]['flagged'])
            frac = n_flag / n_total if n_total > 0 else 0
            bar_colors.append(FLAG_COLOR_MANY if frac >= 0.3 else FLAG_COLOR_FEW)
            edge_colors.append('#7799AA')
        else:
            bar_colors.append(BASE_COLOR)
            edge_colors.append('#3D6D8E')

    bars = ax.bar(x, means_clean, yerr=sems_clean, width=bar_width,
                  color=bar_colors, edgecolor=edge_colors, linewidth=0.8,
                  capsize=3, error_kw={'linewidth': 1.0, 'color': '#555',
                                       'capthick': 1.0},
                  zorder=3)

    # Individual neuron data points
    if has_neurons:
        rng = np.random.default_rng(42)
        for i, d in enumerate(per_ds):
            if len(d[key]) > 0:
                clean_mask = ~d['flagged']
                clean_vals = d[key][clean_mask]
                if len(clean_vals) > 0:
                    jitter = rng.uniform(-0.18, 0.18, len(clean_vals))
                    ax.scatter(i + jitter, clean_vals, color='#333333',
                               s=12, alpha=0.4, zorder=5,
                               linewidths=0, edgecolors='none')
                flag_vals = d[key][d['flagged']]
                if len(flag_vals) > 0:
                    jitter_f = rng.uniform(-0.18, 0.18, len(flag_vals))
                    ax.scatter(i + jitter_f, flag_vals, color='#DD3333',
                               s=30, alpha=0.7, zorder=6,
                               marker='x', linewidths=1.5)

    # Grand mean line
    if len(means_clean) > 0:
        grand_mean = np.mean(means_clean)
        ax.axhline(grand_mean, color='#888888', linewidth=0.8,
                   linestyle='--', alpha=0.5, zorder=2)
        ax.text(n_ds - 0.3, grand_mean, f'μ = {grand_mean:.2f}',
                fontsize=7, color='#666666', va='bottom', ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=45, ha='right', fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=9.5, fontweight='medium')
    ax.grid(axis='y', alpha=0.15, color='#999999', linewidth=0.5)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['left'].set_color('#AAAAAA')
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['bottom'].set_color('#AAAAAA')
    ax.set_ylim(bottom=0)

    return means_clean, sems_clean

def _fig_flagged_neurons(datasets, per_dataset_dir):
    """
    Inspection figures for flagged neurons.

    For each flagged neuron shows:
    - Left panels: max projection crop and baseline (mean) crop with ROI
      contour overlaid so you can assess whether the detection is a real
      neuron, neuropil, vessel, or merged cells.
    - Right panel: raw trace (gray, left axis) and denoised trace (green,
      right axis) on separate scales, with spike markers.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize

    print(f"\n=== FLAGGED NEURON INSPECTION ===")
    print(f"  Checking {len(datasets)} datasets")
    print(f"  Output dir: {per_dataset_dir}")
    logger.info(f"Flagged neuron inspection: checking {len(datasets)} datasets")

    n_total_flagged = 0

    for ds in datasets:
        if ds.selected_spikes is None or ds.selected_spikes.shape[0] < 3:
            print(f"  {ds.name}: SKIPPED (no spikes or <3 neurons)")
            continue

        dur_s = ds.duration_seconds if ds.duration_seconds > 0 else 1.0
        flags = _flag_suspicious_neurons(ds.selected_spikes, dur_s)

        status = (f"  {ds.name}: {flags['n_flagged']}/{ds.selected_spikes.shape[0]} flagged "
                  f"(amp_z max={flags['amp_z'].max():.2f}, freq_z max={flags['freq_z'].max():.2f})")
        print(status)
        logger.info(status)

        if flags['n_flagged'] == 0:
            continue

        flagged_indices = np.where(flags['flagged'])[0]
        n_flagged = len(flagged_indices)
        n_total_flagged += n_flagged

        # Pick the best clean neuron as a reference for comparison
        clean_mask = ~flags['flagged']
        clean_indices = np.where(clean_mask)[0]
        reference_idx = None
        if len(clean_indices) > 0 and ds.selected_quality is not None:
            # Highest quality among clean neurons
            clean_qualities = ds.selected_quality[clean_indices]
            reference_idx = clean_indices[np.argmax(clean_qualities)]

        # Build display list: all flagged + 1 clean reference
        display_indices = list(flagged_indices)
        is_reference = [False] * len(display_indices)
        if reference_idx is not None:
            display_indices.append(reference_idx)
            is_reference.append(True)
        n_display = len(display_indices)

        t_ax = np.arange(ds.selected_traces.shape[1]) / ds.frame_rate
        has_crops = (ds.selected_roi_crops is not None and
                     len(ds.selected_roi_crops) == ds.selected_traces.shape[0])

        # Determine layout: 2 image panels if projections available, else 1
        has_max = has_crops and any(
            c is not None and 'max_proj' in c for c in ds.selected_roi_crops)
        has_baseline = has_crops and any(
            c is not None and 'baseline' in c for c in ds.selected_roi_crops)
        n_img_cols = int(has_max) + int(has_baseline)
        if n_img_cols == 0 and has_crops:
            n_img_cols = 1  # fallback to contour-only

        fig = plt.figure(figsize=(20, n_display * 3.8))
        fig.patch.set_facecolor('white')

        if n_img_cols > 0:
            width_ratios = [1] * n_img_cols + [6]
        else:
            width_ratios = [1]

        gs = gridspec.GridSpec(n_display, len(width_ratios),
                               width_ratios=width_ratios,
                               hspace=0.5, wspace=0.3)

        for row, sel_j in enumerate(display_indices):
            is_ref = is_reference[row]
            # ROI index
            if ds.selected_roi_indices is not None and sel_j < len(ds.selected_roi_indices):
                roi_idx = int(ds.selected_roi_indices[sel_j])
            else:
                roi_idx = sel_j

            col = 0
            crop_data = None
            if has_crops and sel_j < len(ds.selected_roi_crops):
                crop_data = ds.selected_roi_crops[sel_j]

            def _draw_crop(img, contour, title, col_idx):
                """Draw a projection crop with ROI contour overlay."""
                ax_img = fig.add_subplot(gs[row, col_idx])
                # Contrast-stretch the image
                vlo, vhi = np.percentile(img, [2, 99])
                ax_img.imshow(img, cmap='gray', aspect='equal',
                              vmin=vlo, vmax=vhi, interpolation='bilinear')
                # Overlay contour edges
                if contour is not None and contour.shape == img.shape:
                    # Find contour edges (boundary pixels of the mask)
                    from scipy.ndimage import binary_dilation
                    dilated = binary_dilation(contour > 0, iterations=1)
                    edge = dilated & ~(contour > 0)
                    # Draw edges as coloured overlay
                    overlay = np.zeros((*img.shape, 4))
                    overlay[edge, :] = [1, 0.2, 0.2, 0.9]  # red edges
                    ax_img.imshow(overlay, aspect='equal')
                ax_img.set_title(f'{title}\nROI {roi_idx}', fontsize=7,
                                 fontweight='bold')
                ax_img.tick_params(labelsize=5)
                ax_img.set_xlabel('px', fontsize=6)
                if col_idx == 0:
                    ax_img.set_ylabel('px', fontsize=6)

            # Draw max projection crop
            if crop_data is not None and 'max_proj' in crop_data:
                _draw_crop(crop_data['max_proj'], crop_data.get('contour'),
                           'Max Intensity', col)
                col += 1

            # Draw baseline (mean) projection crop
            if crop_data is not None and 'baseline' in crop_data:
                _draw_crop(crop_data['baseline'], crop_data.get('contour'),
                           'Baseline', col)
                col += 1

            # Fallback: just show contour if no projections
            if col == 0 and crop_data is not None and 'contour' in crop_data:
                ax_img = fig.add_subplot(gs[row, 0])
                ax_img.imshow(crop_data['contour'], cmap='hot', aspect='equal')
                ax_img.set_title(f'Footprint\nROI {roi_idx}', fontsize=7,
                                 fontweight='bold')
                ax_img.tick_params(labelsize=5)
                col = 1

            # Trace panel (always the last column)
            trace_col = len(width_ratios) - 1
            ax_raw = fig.add_subplot(gs[row, trace_col])

            # Raw trace on primary y-axis
            if ds.selected_raw_traces is not None and sel_j < ds.selected_raw_traces.shape[0]:
                raw_trace = ds.selected_raw_traces[sel_j]
                baseline = np.percentile(raw_trace, 20)
                if baseline > 0:
                    raw_dff = (raw_trace - baseline) / baseline
                else:
                    raw_dff = raw_trace - np.median(raw_trace)
                ax_raw.plot(t_ax, raw_dff, color='#BBBBBB', linewidth=0.4,
                            alpha=0.8, label='Raw dF/F')
                ax_raw.set_ylabel('Raw dF/F', fontsize=8, color='#888888')
                ax_raw.tick_params(axis='y', labelsize=7, colors='#888888')

            # Denoised trace on secondary y-axis
            ax_den = ax_raw.twinx()
            denoised = ds.selected_traces[sel_j]
            ax_den.plot(t_ax, denoised, color='#00e676', linewidth=1.0,
                        alpha=0.9, label='Denoised')
            ax_den.set_ylabel('Denoised dF/F', fontsize=8, color='#00cc66')
            ax_den.tick_params(axis='y', labelsize=7, colors='#00cc66')

            # Spike markers
            spk_frames = np.where(ds.selected_spikes[sel_j] > 0)[0]
            if len(spk_frames) > 0:
                ax_den.scatter(spk_frames / ds.frame_rate,
                               denoised[spk_frames],
                               color='red', s=25, zorder=5, label='Events')

            # Title with flag reason or reference label
            amp = flags['amps'][sel_j]
            rate = flags['rates'][sel_j]
            amp_z = flags['amp_z'][sel_j]
            freq_z = flags['freq_z'][sel_j]
            quality = ds.selected_quality[sel_j] if ds.selected_quality is not None else 0

            if is_ref:
                title_str = (f'ROI {roi_idx}  \u2014  CLEAN REFERENCE  |  '
                             f'amp={amp:.4f}  rate={rate:.1f}/10s  quality={quality:.3f}  '
                             f'n_spikes={len(spk_frames)}')
                title_color = '#228B22'
            else:
                reasons = []
                if amp_z > 3.0:
                    reasons.append(f'amplitude Z={amp_z:.1f}')
                if freq_z > 3.0:
                    reasons.append(f'frequency Z={freq_z:.1f}')
                reason_str = ', '.join(reasons) if reasons else 'flagged'
                title_str = (f'ROI {roi_idx}  \u2014  {reason_str}  |  '
                             f'amp={amp:.4f}  rate={rate:.1f}/10s  quality={quality:.3f}  '
                             f'n_spikes={len(spk_frames)}')
                title_color = '#CC3333'

            ax_raw.set_title(title_str, fontsize=9, fontweight='bold',
                             color=title_color, loc='left')
            ax_raw.grid(alpha=0.15)
            ax_raw.spines['top'].set_visible(False)
            ax_den.spines['top'].set_visible(False)
            ax_raw.set_xlim(0, t_ax[-1])

            # Combined legend (first row only)
            if row == 0:
                lines_raw, labels_raw = ax_raw.get_legend_handles_labels()
                lines_den, labels_den = ax_den.get_legend_handles_labels()
                ax_den.legend(lines_raw + lines_den, labels_raw + labels_den,
                              fontsize=7, loc='upper right', framealpha=0.8)

            if row == n_display - 1:
                ax_raw.set_xlabel('Time (s)', fontsize=9)

        fig.suptitle(
            f'{ds.name} \u2014 Flagged Neurons ({n_flagged} of {ds.n_selected})',
            fontsize=12, fontweight='bold', color='#CC3333', y=1.01
        )

        safe_name = ds.name.replace('/', '_').replace(' ', '_')[:50]
        ds_dir = os.path.join(per_dataset_dir, safe_name)
        os.makedirs(ds_dir, exist_ok=True)
        path = os.path.join(ds_dir, 'flagged_neurons.png')
        plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  SAVED: {path}")
        logger.info(f"  {ds.name}: saved {n_flagged} flagged neuron inspections -> {path}")

    logger.info(f"  Flagged neuron inspection complete: {n_total_flagged} neurons total")


def _fig_bar_charts(datasets, labels, colors, output_dir, per_metric_dir=None):
    """
    Publication-quality bar charts with within-dataset neuron outlier detection.

    Produces a combined 4-panel figure AND individual per-metric figures.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_ds = len(datasets)
    if n_ds == 0:
        return os.path.join(output_dir, 'bar_charts.png')

    ds_names = [_abbrev(d.name) for d in datasets]

    # ── Compute per-neuron stats and flag suspicious neurons ─────────────
    per_ds = []
    for ds in datasets:
        dur_s = ds.duration_seconds if ds.duration_seconds > 0 else 1.0
        rates = _get_neuron_rates(ds)
        amps = _get_neuron_amplitudes(ds)
        n_sel = len(rates) if len(rates) > 0 else ds.n_selected

        if n_sel >= 3 and len(rates) > 0:
            # Ensure amps matches rates length (some neurons may have no spikes)
            if len(amps) != len(rates):
                amps = np.zeros(len(rates))
                raw_amps = _get_neuron_amplitudes(ds)
                amps[:len(raw_amps)] = raw_amps[:len(rates)]

            # Flag using precomputed rates/amplitudes
            from scipy.stats import zscore as _zscore_scipy
            rate_z = _zscore_scipy(rates) if len(rates) > 1 else np.zeros(len(rates))
            amp_z = _zscore_scipy(amps) if len(amps) > 1 else np.zeros(len(amps))
            rate_z = np.nan_to_num(rate_z, 0.0)
            amp_z = np.nan_to_num(amp_z, 0.0)
            flagged = (np.abs(rate_z) > 3.0) | (np.abs(amp_z) > 3.0)
            flags = {
                'flagged': flagged,
                'amp_z': amp_z,
                'freq_z': rate_z,
                'n_flagged': int(flagged.sum()),
                'rates': rates,
                'amps': amps,
            }
        else:
            flags = {
                'flagged': np.zeros(n_sel, dtype=bool),
                'amp_z': np.zeros(n_sel),
                'freq_z': np.zeros(n_sel),
                'n_flagged': 0,
                'rates': rates if len(rates) > 0 else np.zeros(n_sel),
                'amps': amps if len(amps) > 0 else np.zeros(n_sel),
            }
        per_ds.append({
            'rates': flags['rates'],
            'amps': flags['amps'],
            'flagged': flags['flagged'],
            'n_flagged': flags['n_flagged'],
            'amp_z': flags['amp_z'],
            'freq_z': flags['freq_z'],
            'corr': ds.pairwise_correlation_mean,
            'sync': ds.synchrony_index,
        })

    panels = [
        ('rates', 'Spike Rate (events / 10 s)',                      True,  'spike_rate'),
        ('amps',  'Transient amplitude (ΔF/F₀)',                         True,  'spike_amplitude'),
        ('corr',  'Pairwise Correlation (calcium traces, r)',         False, 'pairwise_correlation'),
        ('sync',  'Synchrony Index (0 = independent, 1 = synchronised)', False, 'synchrony_index'),
    ]

    # Legend elements (shared)
    typical_n = int(np.median([ds.n_selected for ds in datasets]))
    n_flagged_total = sum(d['n_flagged'] for d in per_ds)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333',
               markersize=6, label=f'Neuron (n={typical_n} per dataset)'),
        Line2D([0], [0], marker='x', color='#DD3333', markeredgewidth=1.5,
               markersize=7, linestyle='None',
               label=f'Flagged — high amp/freq ({n_flagged_total} total)'),
        Line2D([0], [0], linestyle='--', color='#888888', linewidth=0.8,
               label='Grand mean (excl. flagged)'),
    ]

    fig_width = max(14, n_ds * 0.85)

    # ── Individual panel figures ─────────────────────────────────────────
    metric_dir = per_metric_dir or os.path.join(output_dir, 'per_metric')
    os.makedirs(metric_dir, exist_ok=True)

    for key, ylabel, has_neurons, fname in panels:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(fig_width, 3.8))
        fig_single.patch.set_facecolor('white')
        _draw_bar_panel(ax_single, key, ylabel, has_neurons,
                        per_ds, ds_names, n_ds)
        ax_single.set_title(ylabel, fontsize=11, fontweight='bold',
                            color='#333333', pad=10)
        fig_single.legend(handles=legend_elements, loc='upper right',
                          fontsize=7, framealpha=0.9, edgecolor='#CCCCCC',
                          bbox_to_anchor=(0.98, 0.98))
        plt.tight_layout()
        plt.savefig(os.path.join(metric_dir, f'{fname}.png'),
                    dpi=250, bbox_inches='tight', facecolor='white')
        plt.close(fig_single)

    # ── Combined 4-panel figure ──────────────────────────────────────────
    fig, axes = plt.subplots(len(panels), 1, figsize=(fig_width, len(panels) * 3.5))
    fig.patch.set_facecolor('white')

    for ax, (key, ylabel, has_neurons, _) in zip(axes, panels):
        _draw_bar_panel(ax, key, ylabel, has_neurons,
                        per_ds, ds_names, n_ds)

    fig.legend(handles=legend_elements, loc='upper right', fontsize=7.5,
               framealpha=0.9, edgecolor='#CCCCCC', bbox_to_anchor=(0.98, 0.99),
               ncol=1)
    fig.suptitle('Dataset Comparison',
                 fontsize=13, fontweight='bold', color='#333333', y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(output_dir, 'bar_charts.png')
    plt.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def _draw_neuron_selection_row(ax_hist, ax_traces, ds):
    """Draw a single dataset's neuron selection panels (histogram + traces)."""

    # ── SNR distribution ──────────────────────────────────────────────
    if ds.selected_quality is not None and len(ds.selected_quality) > 0:
        snr_sel = ds.selected_quality  # now stores SNR values
        ax_hist.hist(snr_sel, bins=20, color='#00e676', alpha=0.8,
                     edgecolor='none', label=f'Selected ({len(snr_sel)})')
        ax_hist.set_xlabel('Trace SNR', fontsize=7)
        ax_hist.set_ylabel('Count', fontsize=7)
        ax_hist.legend(fontsize=6)
        ax_hist.set_title(f'{_abbrev(ds.name)}\n{ds.n_selected}/{ds.n_neurons} selected',
                          fontsize=8, fontweight='bold')
        ax_hist.tick_params(labelsize=6)

    # ── Selected traces ─────────────────────────────────────────────
    if ds.selected_traces is not None and ds.selected_spikes is not None:
        n_sel = ds.selected_traces.shape[0]
        T = ds.selected_traces.shape[1]
        t_ax = np.arange(T) / ds.frame_rate

        for j in range(n_sel):
            trace = ds.selected_traces[j]
            offset = j * 1.2
            t_min, t_max = trace.min(), trace.max()
            t_range = t_max - t_min if t_max > t_min else 1
            trace_norm = (trace - t_min) / t_range + offset

            ax_traces.plot(t_ax, trace_norm, color='#00e676',
                           linewidth=0.6, alpha=0.8)

            spike_frames = np.where(ds.selected_spikes[j] > 0)[0]
            if len(spike_frames) > 0:
                ax_traces.scatter(
                    spike_frames / ds.frame_rate,
                    trace_norm[spike_frames],
                    color='red', s=4, zorder=5,
                )

            q = ds.selected_quality[j] if ds.selected_quality is not None else 0
            roi_idx = ds.selected_indices[j] if ds.selected_indices is not None else j
            ax_traces.text(t_ax[-1] * 1.01, offset + 0.5,
                           f'ROI {roi_idx} (q={q:.2f})',
                           fontsize=5, va='center')

        ax_traces.set_xlim(0, t_ax[-1])
        ax_traces.set_xlabel('Time (s)', fontsize=7)
        ax_traces.set_yticks([])
        ax_traces.set_title('Selected Neuron Traces (ranked by quality)',
                            fontsize=8)
        ax_traces.tick_params(labelsize=6)


def _fig_neuron_selection(datasets, output_dir, per_dataset_dir=None):
    """
    Neuron selection transparency figure.

    Produces:
    - per_dataset/<name>/neuron_selection.png  per dataset
    - overview/neuron_selection.png            combined overview
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_ds = len(datasets)
    sel_base = per_dataset_dir or os.path.join(output_dir, 'Results by Dataset')

    # ── Individual per-dataset figures ────────────────────────────────────
    for ds in datasets:
        safe_name = ds.name.replace('/', '_').replace(' ', '_')[:50]
        ds_dir = os.path.join(sel_base, safe_name)
        os.makedirs(ds_dir, exist_ok=True)

        fig, (ax_hist, ax_traces) = plt.subplots(
            1, 2, figsize=(18, 3.5),
            gridspec_kw={'width_ratios': [1, 3]}
        )
        _draw_neuron_selection_row(ax_hist, ax_traces, ds)
        fig.suptitle(f'{ds.name} — Neuron Selection',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(ds_dir, 'neuron_selection.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

    # ── Combined overview ────────────────────────────────────────────────
    overview_dir = os.path.join(output_dir, 'Full Overview')
    os.makedirs(overview_dir, exist_ok=True)

    fig, axes = plt.subplots(n_ds, 2, figsize=(18, n_ds * 2.5),
                             gridspec_kw={'width_ratios': [1, 3]})
    if n_ds == 1:
        axes = axes.reshape(1, -1)

    for i, ds in enumerate(datasets):
        _draw_neuron_selection_row(axes[i, 0], axes[i, 1], ds)

    fig.suptitle('Neuron Selection — Quality Scoring & Selected Traces',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(overview_dir, 'neuron_selection.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def _fig_quality_gating(datasets, max_thresh, res_thresh, drift_thresh, output_dir):
    """Show all quality metrics for all datasets with exclusion thresholds."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [_abbrev(d.name) for d in datasets]
    max_shifts = [d.motion_max_shift for d in datasets]
    residuals = [d.motion_residual_std for d in datasets]
    drifts = [d.baseline_drift for d in datasets]
    excluded = [d.motion_excluded for d in datasets]

    n_ds = len(datasets)
    fig_h = max(5, n_ds * 0.4)
    fig, axes = plt.subplots(1, 3, figsize=(20, fig_h))
    fig.patch.set_facecolor('white')

    y_pos = range(n_ds)

    # Per-dataset: determine exclusion reason for colour coding
    colors_shift = []
    colors_resid = []
    colors_drift = []
    for d in datasets:
        if d.motion_max_shift > max_thresh:
            colors_shift.append('#FF5252')
        elif d.motion_excluded:
            colors_shift.append('#FFAB91')  # excluded by other criterion
        else:
            colors_shift.append('#5B8DBE')

        if d.motion_residual_std > res_thresh:
            colors_resid.append('#FF5252')
        elif d.motion_excluded:
            colors_resid.append('#FFAB91')
        else:
            colors_resid.append('#5B8DBE')

        if d.baseline_drift > drift_thresh:
            colors_drift.append('#FF5252')
        elif d.motion_excluded:
            colors_drift.append('#FFAB91')
        else:
            colors_drift.append('#5B8DBE')

    # Panel 1: Max shift
    ax = axes[0]
    ax.set_facecolor('white')
    ax.barh(y_pos, max_shifts, color=colors_shift, edgecolor='#333333',
            linewidth=0.4, height=0.7)
    ax.axvline(max_thresh, color='#CC3333', linestyle='--', linewidth=1.5,
               label=f'Threshold ({max_thresh} px)', zorder=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Max Shift (px)', fontsize=9)
    ax.set_title('Maximum Motion Shift', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: Residual jitter
    ax = axes[1]
    ax.set_facecolor('white')
    ax.barh(y_pos, residuals, color=colors_resid, edgecolor='#333333',
            linewidth=0.4, height=0.7)
    ax.axvline(res_thresh, color='#CC3333', linestyle='--', linewidth=1.5,
               label=f'Threshold ({res_thresh} px)', zorder=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Residual Jitter Std (px)', fontsize=9)
    ax.set_title('Residual Motion After Correction', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 3: Baseline drift
    ax = axes[2]
    ax.set_facecolor('white')
    ax.barh(y_pos, drifts, color=colors_drift, edgecolor='#333333',
            linewidth=0.4, height=0.7)
    ax.axvline(drift_thresh, color='#CC3333', linestyle='--', linewidth=1.5,
               label=f'Threshold ({drift_thresh})', zorder=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Drift Ratio (|Q4-Q1| / std)', fontsize=9)
    ax.set_title('Baseline Drift', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    n_ex = sum(excluded)
    fig.suptitle(f'Dataset Quality Gating \u2014 {n_ex} excluded / {n_ds} total',
                 fontsize=13, fontweight='bold', y=1.02)

    # Legend for colours
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#5B8DBE', edgecolor='#333', label='Included'),
        Patch(facecolor='#FF5252', edgecolor='#333', label='Excluded (this criterion)'),
        Patch(facecolor='#FFAB91', edgecolor='#333', label='Excluded (other criterion)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = os.path.join(output_dir, 'quality_gating.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def _fig_selected_traces(datasets: List[DatasetMetrics], output_dir: str) -> List[str]:
    """
    Generate trace figures for the SELECTED neurons used in statistical
    comparisons.

    For each recording, produces a figure with one row per selected neuron.
    Each row shows two overlaid traces:
      - Raw ΔF/F trace (before OASIS deconvolution) in grey
      - Denoised trace (after OASIS) in green/colour
      - Detected spike events marked as red dots

    This allows direct visual comparison of the deconvolution quality
    for every neuron entering the statistical analysis.

    Saves all figures to:
        {output_dir}/figures/Selected Traces/{recording_name}.png
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path

    traces_dir = os.path.join(output_dir, 'figures', 'Selected Traces')
    os.makedirs(traces_dir, exist_ok=True)
    paths = []

    for ds in datasets:
        result_path = Path(ds.filepath)
        denoised_path = result_path / 'traces_denoised.npy'
        spikes_path = result_path / 'spike_trains.npy'
        raw_path = result_path / 'temporal_traces.npy'

        if not spikes_path.exists():
            logger.warning(f"  {ds.name}: no spike_trains.npy, skipping trace figure")
            continue

        # Load full arrays
        S_all = np.load(spikes_path)
        C_all = np.load(denoised_path) if denoised_path.exists() else None
        R_all = np.load(raw_path) if raw_path.exists() else None

        if C_all is None and R_all is None:
            logger.warning(f"  {ds.name}: no trace data, skipping")
            continue

        # Get the selected neuron indices (into full array)
        roi_idx = ds.selected_roi_indices
        if roi_idx is None or len(roi_idx) == 0:
            logger.warning(f"  {ds.name}: no selected_roi_indices, skipping")
            continue

        # Bounds-check indices against loaded arrays
        max_idx = S_all.shape[0]
        valid_mask = roi_idx < max_idx
        if C_all is not None:
            valid_mask &= roi_idx < C_all.shape[0]
        if R_all is not None:
            valid_mask &= roi_idx < R_all.shape[0]
        roi_idx_valid = roi_idx[valid_mask]
        if len(roi_idx_valid) == 0:
            continue

        S_sel = S_all[roi_idx_valid]
        C_sel = C_all[roi_idx_valid] if C_all is not None else None
        R_sel = R_all[roi_idx_valid] if R_all is not None else None
        N = len(roi_idx_valid)
        T = S_sel.shape[1]
        t_ax = np.arange(T) / ds.frame_rate
        duration_s = t_ax[-1] if len(t_ax) > 0 else 0

        # Active / inactive mask
        is_active = ds.neuron_is_active
        if is_active is not None and len(is_active) == len(roi_idx):
            is_active = is_active[valid_mask]
        else:
            is_active = np.array([np.sum(S_sel[j] > 0) > 0 for j in range(N)])

        # Per-neuron spike rates for labelling
        rates = np.array([np.sum(S_sel[j] > 0) / duration_s * 10.0
                         if duration_s > 0 else 0.0 for j in range(N)])

        # Sort by spike rate (most active at top)
        order = np.argsort(rates)[::-1]

        # ── Figure: one row per neuron ───────────────────────────────────
        row_height = 1.8
        fig_height = max(6, N * row_height + 2)
        fig, axes = plt.subplots(N, 1, figsize=(16, fig_height), sharex=True)
        fig.patch.set_facecolor('white')
        if N == 1:
            axes = [axes]

        for plot_i, neuron_i in enumerate(order):
            ax = axes[plot_i]
            ax.set_facecolor('white')

            orig_roi = int(roi_idx_valid[neuron_i])
            active = is_active[neuron_i]
            n_spikes = int(np.sum(S_sel[neuron_i] > 0))
            rate = rates[neuron_i]

            # ── Raw trace (before OASIS) ─────────────────────────────
            if R_sel is not None:
                raw_trace = R_sel[neuron_i]
                ax.plot(t_ax[:len(raw_trace)], raw_trace,
                        color='#B0BEC5', linewidth=0.5, alpha=0.7,
                        label='Raw ΔF/F', zorder=2)

            # ── Denoised trace (after OASIS) ─────────────────────────
            if C_sel is not None:
                den_trace = C_sel[neuron_i]
                trace_color = '#00e676' if active else '#78909C'
                ax.plot(t_ax[:len(den_trace)], den_trace,
                        color=trace_color, linewidth=0.8,
                        alpha=0.95 if active else 0.6,
                        label='Denoised (OASIS)', zorder=3)

            # ── Spike event markers ──────────────────────────────────
            spk_frames = np.where(S_sel[neuron_i] > 0)[0]
            if len(spk_frames) > 0:
                # Place markers on the denoised trace if available, else raw
                if C_sel is not None:
                    spk_y = C_sel[neuron_i][spk_frames]
                elif R_sel is not None:
                    spk_y = R_sel[neuron_i][spk_frames]
                else:
                    spk_y = np.zeros(len(spk_frames))
                ax.scatter(spk_frames / ds.frame_rate, spk_y,
                           color='#D32F2F', s=12, zorder=5, alpha=0.8,
                           marker='v', linewidths=0, label='Calcium events')

            # ── ROI label ────────────────────────────────────────────
            status = '●' if active else '○'
            label_color = '#333' if active else '#999'
            ax.set_ylabel(f'{status} ROI {orig_roi}\n{rate:.1f}/10s\n({n_spikes} spk)',
                         fontsize=7, fontweight='bold', color=label_color,
                         rotation=0, labelpad=55, va='center')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.grid(axis='x', alpha=0.1)

            # Legend on first row only
            if plot_i == 0:
                ax.legend(fontsize=6, loc='upper right', framealpha=0.8,
                         ncol=3, handlelength=1.5)

        # X-axis label on bottom row
        axes[-1].set_xlabel('Time (s)', fontsize=10)
        axes[-1].set_xlim(0, duration_s)

        # ── Title and caption ────────────────────────────────────────
        genotype = _extract_genotype(ds.name)
        geno_str = f'  [{genotype}]' if genotype != 'Unknown' else ''
        n_active = int(np.sum(is_active))

        fig.suptitle(
            f'{ds.name}{geno_str} — {N} selected neurons '
            f'({n_active} active, {N - n_active} inactive)',
            fontsize=12, fontweight='bold', y=1.0)

        fig.text(0.5, -0.005,
                 f'Grey = raw ΔF/F  ·  Green = OASIS denoised  ·  '
                 f'Red ▼ = detected spike events  ·  '
                 f'Duration: {duration_s:.0f}s  ·  {ds.frame_rate:.1f} Hz',
                 ha='center', fontsize=7, color='#777', style='italic')

        plt.tight_layout(rect=[0.08, 0.01, 1, 0.98])

        # Save with recording name
        safe_name = ds.name.replace('/', '_').replace(' ', '_').replace('\\', '_')
        fig_path = os.path.join(traces_dir, f'{safe_name}.png')
        plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        paths.append(fig_path)

    logger.info(f"Generated {len(paths)} selected-neuron trace figures in {traces_dir}")
    return paths


# =============================================================================
# CORE ACTIVITY ANALYSIS (v1.6)
# =============================================================================

def run_activity_analysis(datasets: List[DatasetMetrics], output_dir: str,
                          mutant_label: str = 'CEP41 R242H') -> dict:
    """
    Core activity analysis: frequency, amplitude, and active fraction.

    Analyses activity across three dimensions:
    1. Genotype comparison (Control vs Mutant)
    2. Longitudinal (across organoid day/age)
    3. Combined (genotype × age)

    Only ACTIVE neurons (≥1 validated transient) contribute to frequency
    and amplitude metrics.  Active fraction is reported separately.

    Figures saved to: figures/3 - Activity Analysis/
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats as sp_stats
    from collections import OrderedDict
    import re

    results = {'level': 'activity_analysis'}

    activity_dir = os.path.join(output_dir, 'figures', '3 - Activity Analysis')
    os.makedirs(activity_dir, exist_ok=True)

    # ── Parse metadata ───────────────────────────────────────────────────
    genotypes = [_extract_genotype(ds.name) for ds in datasets]
    organoid_ids = [_extract_organoid_id(ds.name) for ds in datasets]

    # Sort organoid days by numeric age
    def _day_num(oid):
        m = re.search(r'\d+', oid)
        return int(m.group()) if m else 0

    unique_days = sorted(set(organoid_ids), key=_day_num)

    CTRL_COLOR = '#4472C4'
    MUT_COLOR = '#ED7D31'
    GENO_COLORS = {'Control': CTRL_COLOR, 'Mutant': MUT_COLOR, 'Unknown': '#999999'}

    n_ctrl = sum(1 for g in genotypes if g == 'Control')
    n_mut = sum(1 for g in genotypes if g == 'Mutant')
    logger.info(f"Activity analysis: {n_ctrl} Control, {n_mut} Mutant, "
                f"{len(unique_days)} days ({', '.join(unique_days)})")

    # ── Extract per-dataset metrics ──────────────────────────────────────
    ds_data = []
    for ds, geno, day in zip(datasets, genotypes, organoid_ids):
        if geno == 'Unknown':
            continue

        # Active neurons only for frequency/amplitude
        active_mask = ds.neuron_is_active if ds.neuron_is_active is not None else np.zeros(0, dtype=bool)
        active_rates = ds.neuron_spike_rates[active_mask] if ds.neuron_spike_rates is not None and active_mask.any() else np.array([])
        active_amps = ds.neuron_spike_amplitudes[active_mask] if ds.neuron_spike_amplitudes is not None and active_mask.any() else np.array([])
        # Filter out zero amplitudes from active neurons (neurons where amplitude measurement failed)
        active_amps = active_amps[active_amps > 0] if len(active_amps) > 0 else active_amps

        ds_data.append({
            'name': ds.name,
            'genotype': geno,
            'day': day,
            'day_num': _day_num(day),
            'n_selected': ds.n_selected,
            'n_active': ds.n_active,
            'active_fraction': ds.active_fraction,
            'active_rates': active_rates,
            'active_amps': active_amps,
            'mean_rate': float(np.mean(active_rates)) if len(active_rates) > 0 else 0.0,
            'mean_amp': float(np.mean(active_amps)) if len(active_amps) > 0 else 0.0,
        })

    if not ds_data:
        logger.warning("No datasets with genotype info for activity analysis")
        return results

    # ── Helper: Mann-Whitney with effect size ────────────────────────────
    def _mw_test(a, b, label=''):
        if len(a) < 2 or len(b) < 2:
            return {'label': label, 'skipped': True, 'n_a': len(a), 'n_b': len(b)}
        U, p = sp_stats.mannwhitneyu(a, b, alternative='two-sided')
        pooled_std = np.sqrt(((len(a)-1)*np.var(a, ddof=1) +
                              (len(b)-1)*np.var(b, ddof=1)) /
                             (len(a) + len(b) - 2))
        d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 1e-10 else 0.0
        return {'label': label, 'U': float(U), 'p': float(p), 'cohens_d': float(d),
                'n_a': len(a), 'n_b': len(b),
                'mean_a': float(np.mean(a)), 'mean_b': float(np.mean(b))}

    # =====================================================================
    # FIGURE 1: Genotype comparison (3 panels: frequency, amplitude, active fraction)
    # =====================================================================
    ctrl_data = [d for d in ds_data if d['genotype'] == 'Control']
    mut_data = [d for d in ds_data if d['genotype'] == 'Mutant']

    # Per-recording means for rate and amplitude (consistent with genotype comparison)
    ctrl_rates = np.array([d['mean_rate'] for d in ctrl_data if d['mean_rate'] > 0])
    mut_rates = np.array([d['mean_rate'] for d in mut_data if d['mean_rate'] > 0])
    ctrl_amps = np.array([d['mean_amp'] for d in ctrl_data if d['mean_amp'] > 0])
    mut_amps = np.array([d['mean_amp'] for d in mut_data if d['mean_amp'] > 0])

    # Per-recording active fractions
    ctrl_af = np.array([d['active_fraction'] for d in ctrl_data])
    mut_af = np.array([d['active_fraction'] for d in mut_data])

    # Statistical tests (per-recording)
    test_rate = _mw_test(ctrl_rates, mut_rates, 'Event Frequency')
    test_amp = _mw_test(ctrl_amps, mut_amps, 'Transient Amplitude')
    test_af = _mw_test(ctrl_af, mut_af, 'Active Fraction')

    results['genotype_tests'] = {
        'spike_frequency': test_rate,
        'spike_amplitude': test_amp,
        'active_fraction': test_af,
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor('white')

    rng = np.random.default_rng(42)

    panels = [
        ('Event frequency\n(events/10s, per recording)', ctrl_rates, mut_rates, test_rate),
        ('Transient amplitude\n(ΔF/F₀, per recording)', ctrl_amps, mut_amps, test_amp),
        ('Active Fraction\n(per recording)', ctrl_af, mut_af, test_af),
    ]

    for col, (title, ctrl_vals, mut_vals, test) in enumerate(panels):
        ax = axes[col]
        ax.set_facecolor('white')

        data = [ctrl_vals, mut_vals]
        if all(len(d) > 0 for d in data):
            bp = ax.boxplot(data, positions=[0, 1], widths=0.5,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='white', linewidth=2.0),
                            whiskerprops=dict(color='#555', linewidth=1.0),
                            capprops=dict(color='#555', linewidth=1.0))
            for patch, col_c in zip(bp['boxes'], [CTRL_COLOR, MUT_COLOR]):
                patch.set_facecolor(col_c)
                patch.set_alpha(0.35)
                patch.set_edgecolor(col_c)
                patch.set_linewidth(1.5)

        for i, (vals, col_c) in enumerate(zip(data, [CTRL_COLOR, MUT_COLOR])):
            if len(vals) > 0:
                jitter = rng.uniform(-0.15, 0.15, len(vals))
                ax.scatter(i + jitter, vals, c=col_c, s=55, alpha=0.7,
                           zorder=5, edgecolors='white', linewidth=0.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control', mutant_label], fontsize=16, fontweight='bold')
        for tick, col_c in zip(ax.get_xticklabels(), [CTRL_COLOR, MUT_COLOR]):
            tick.set_color(col_c)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.tick_params(labelsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)

        # Significance
        if 'p' in test:
            stars = _sig_stars(test['p'])
            _cv2 = ctrl_vals[np.isfinite(ctrl_vals)] if len(ctrl_vals) > 0 else ctrl_vals
            _mv2 = mut_vals[np.isfinite(mut_vals)]   if len(mut_vals)  > 0 else mut_vals
            y_max = max(float(np.max(_cv2)) if len(_cv2) > 0 else 0,
                        float(np.max(_mv2)) if len(_mv2) > 0 else 0)
            if y_max > 0:
                _draw_sig_bracket(ax, 0, 1, y_max * 1.05, y_max * 0.06,
                                  f'{stars}\n{_fmt_p(test["p"])}', fontsize=12)
                ax.set_ylim(top=y_max * 1.25)

    fig.suptitle(f'Genotype Comparison: Control vs {mutant_label} (per-recording averages)',
                 fontsize=18, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02,
             f'Control: {len(ctrl_rates)} recordings | '
             f'{mutant_label}: {len(mut_rates)} recordings | '
             f'Mann-Whitney U, two-sided  ·  Each dot = one recording',
             ha='center', fontsize=11, color='#777', style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(activity_dir, 'genotype_comparison.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # =====================================================================
    # FIGURE 2: Longitudinal — metrics across organoid day/age
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(max(18, len(unique_days) * 2.5), 6))
    fig.patch.set_facecolor('white')

    for col, (metric_name, metric_key, ylabel) in enumerate([
        ('Event Frequency', 'mean_rate', 'events/10s'),
        ('Transient Amplitude', 'mean_amp', 'ΔF/F₀'),
        ('Active Fraction', 'active_fraction', 'fraction'),
    ]):
        ax = axes[col]
        ax.set_facecolor('white')

        positions = []
        tick_labels = []
        for day_idx, day in enumerate(unique_days):
            day_ds = [d for d in ds_data if d['day'] == day]
            if not day_ds:
                continue

            for geno, color, offset in [('Control', CTRL_COLOR, -0.15), ('Mutant', MUT_COLOR, 0.15)]:
                geno_ds = [d for d in day_ds if d['genotype'] == geno]
                if not geno_ds:
                    continue
                vals = [d[metric_key] for d in geno_ds]
                x = day_idx + offset
                ax.scatter([x] * len(vals), vals, c=color, s=40, alpha=0.7,
                           zorder=5, edgecolors='white', linewidth=0.5)
                if len(vals) > 1:
                    ax.plot([x, x], [np.mean(vals) - np.std(vals), np.mean(vals) + np.std(vals)],
                            color=color, linewidth=1.5, alpha=0.5)
                    ax.plot(x, np.mean(vals), 's', color=color, markersize=8,
                            zorder=6, markeredgecolor='white', markeredgewidth=0.5)

            positions.append(day_idx)
            tick_labels.append(day)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=13, fontweight='bold', rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(f'{metric_name} by Age', fontsize=15, fontweight='bold')
        ax.tick_params(labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=CTRL_COLOR, markersize=8, label='Control'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=MUT_COLOR, markersize=8, label=mutant_label)]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=13)

    fig.suptitle('Longitudinal Activity by Organoid Age',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(activity_dir, 'longitudinal_by_age.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # =====================================================================
    # FIGURE 3: Combined — genotype × age heatmap
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(unique_days) * 2), 5))
    fig.patch.set_facecolor('white')

    for col, (metric_name, metric_key) in enumerate([
        ('Event Frequency', 'mean_rate'),
        ('Transient Amplitude', 'mean_amp'),
        ('Active Fraction', 'active_fraction'),
    ]):
        ax = axes[col]

        for geno_idx, (geno, color, marker) in enumerate([
            ('Control', CTRL_COLOR, 'o'), ('Mutant', MUT_COLOR, 's')
        ]):
            day_means = []
            day_sems = []
            day_positions = []

            for day_idx, day in enumerate(unique_days):
                geno_ds = [d for d in ds_data if d['day'] == day and d['genotype'] == geno]
                if not geno_ds:
                    continue
                vals = [d[metric_key] for d in geno_ds]
                day_means.append(np.mean(vals))
                day_sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                day_positions.append(day_idx)

            if day_means:
                display_label = mutant_label if geno == 'Mutant' else geno
                ax.errorbar(day_positions, day_means, yerr=day_sems,
                            color=color, marker=marker, markersize=7,
                            capsize=3, linewidth=1.5, label=display_label, alpha=0.8)

        ax.set_xticks(range(len(unique_days)))
        ax.set_xticklabels(unique_days, fontsize=9, rotation=45, ha='right')
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)
        if col == 0:
            ax.legend(fontsize=9)

    fig.suptitle('Genotype × Age: Activity Metrics',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(activity_dir, 'genotype_x_age.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # ── Store results ────────────────────────────────────────────────────
    results['summary'] = {
        'n_datasets': len(ds_data),
        'n_control': len(ctrl_data),
        'n_mutant': len(mut_data),
        'n_days': len(unique_days),
        'days': unique_days,
        'total_active_ctrl': int(len(ctrl_rates)),
        'total_active_mut': int(len(mut_rates)),
        'mean_active_fraction_ctrl': float(np.mean(ctrl_af)) if len(ctrl_af) > 0 else 0,
        'mean_active_fraction_mut': float(np.mean(mut_af)) if len(mut_af) > 0 else 0,
    }

    logger.info(f"  Activity analysis complete:")
    logger.info(f"    Control: {len(ctrl_rates)} active neurons across {len(ctrl_data)} recordings "
                f"(mean active fraction: {np.mean(ctrl_af):.1%})" if len(ctrl_af) > 0 else "")
    logger.info(f"    Mutant: {len(mut_rates)} active neurons across {len(mut_data)} recordings "
                f"(mean active fraction: {np.mean(mut_af):.1%})" if len(mut_af) > 0 else "")

    return results


def _generate_roi_peak_figures(datasets: List, output_dir: str) -> None:
    """Generate one PNG per selected ROI showing the peak transient frame,
    a pre-transient reference frame, and the full trace underneath.

    Layout (per file):
        Top row: [reference frame | peak frame]  — side by side, equal size,
                 ROI contour overlaid in cyan, scale bar bottom-right.
        Bottom:  Full denoised + raw trace, spike markers, time cursor at peak.

    ROIs are ranked by peak_SNR = max(denoised) / OASIS_sn and the rank
    number is embedded in the filename so they sort naturally.

    Reference frame selection: scan backwards from the peak frame with a
    minimum margin of 1 frame before the peak, find the lowest-intensity
    frame within the 10 seconds preceding the spike for maximum contrast.
    fluorescence in the ROI bounding box is closest to the rolling baseline
    of the raw trace.  This avoids picking a frame mid-transient.

    Output: figures/ROI Peak Frames/{recording_name}/rank{N:03d}_roi{M:04d}.png
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from pathlib import Path
    from scipy.sparse import load_npz
    from scipy.ndimage import percentile_filter

    out_root = os.path.join(output_dir, 'figures', 'ROI Peak Frames')
    os.makedirs(out_root, exist_ok=True)

    ref_margin_frames = 1      # minimum frames before peak to start ref search
    crop_radius    = 45    # pixels around centroid
    fig_w, fig_h   = 10, 5  # inches

    for ds in datasets:
        result_path = Path(ds.filepath)

        # ── Load required arrays ──────────────────────────────────────────
        denoised_path  = result_path / 'traces_denoised.npy'
        raw_path       = result_path / 'temporal_traces.npy'
        spikes_path    = result_path / 'spike_trains.npy'
        noise_path     = result_path / 'deconv_noise.npy'
        footprint_path = result_path / 'spatial_footprints.npz'
        info_path      = result_path / 'run_info.json'

        if not denoised_path.exists() or not spikes_path.exists():
            logger.warning(f"  {ds.name}: missing denoised/spikes, skipping peak figures")
            continue
        if ds.selected_roi_indices is None or len(ds.selected_roi_indices) == 0:
            continue

        C_all  = np.load(denoised_path)
        S_all  = np.load(spikes_path)
        R_all  = np.load(raw_path) if raw_path.exists() else None
        noise  = np.load(noise_path) if noise_path.exists() else None

        # ── Load movie ────────────────────────────────────────────────────
        movie = None
        dims  = None
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            movie_path = info.get('config', {}).get('movie') or info.get('movie')
            if 'dims' in info:
                dims = tuple(info['dims'])
            elif 'd1' in info and 'd2' in info:
                dims = (int(info['d1']), int(info['d2']))

            if movie_path and os.path.exists(movie_path):
                try:
                    ext = os.path.splitext(movie_path)[1].lower()
                    if ext == '.nd2':
                        import nd2
                        _m = nd2.imread(movie_path)
                        movie = (_m[:, 0] if _m.ndim == 4 else _m).astype(np.float32)
                    elif ext in ('.tif', '.tiff'):
                        from tifffile import imread as _tifread
                        movie = _tifread(movie_path).astype(np.float32)
                    elif ext == '.npy':
                        movie = np.load(movie_path).astype(np.float32)
                    logger.info(f"  {ds.name}: loaded movie {movie.shape}")
                except Exception as me:
                    logger.warning(f"  {ds.name}: movie load failed ({me}), using projections only")
                    movie = None

        # ── Load spatial footprints + reconcile dims ──────────────────────
        A_sparse = None
        if footprint_path.exists():
            try:
                A_sparse = load_npz(footprint_path)
                n_pix = A_sparse.shape[0]

                # dims from run_info may reflect the original uncropped movie
                # while footprints use the motion-corrected (cropped) size.
                # Work through several sources in order of reliability:
                dims_ok = dims is not None and dims[0] * dims[1] == n_pix

                if not dims_ok:
                    # 1. Try max_projection / mean_projection saved alongside results
                    for proj_name in ('max_projection.npy', 'mean_projection.npy'):
                        proj_p = result_path / proj_name
                        if proj_p.exists():
                            try:
                                mp = np.load(proj_p)
                                if mp.shape[0] * mp.shape[1] == n_pix:
                                    dims = mp.shape[:2]
                                    dims_ok = True
                                    logger.info(f"  {ds.name}: dims corrected to "
                                                f"{dims} from {proj_name}")
                                    break
                            except Exception:
                                pass

                if not dims_ok and dims is not None:
                    # 2. Motion correction trims symmetrically — try small trims
                    d1_orig, d2_orig = dims
                    for trim in range(1, 25):
                        d1t = d1_orig - 2 * trim
                        d2t = d2_orig - 2 * trim
                        if d1t > 0 and d2t > 0 and d1t * d2t == n_pix:
                            dims = (d1t, d2t)
                            dims_ok = True
                            logger.info(f"  {ds.name}: dims corrected to "
                                        f"{dims} (trimmed {trim}px per side)")
                            break

                if not dims_ok:
                    # 3. Try square
                    side = int(np.sqrt(n_pix))
                    if side * side == n_pix:
                        dims = (side, side)
                        dims_ok = True
                        logger.info(f"  {ds.name}: dims inferred as square {dims}")

                if not dims_ok:
                    logger.warning(f"  {ds.name}: cannot reconcile dims with "
                                   f"footprint ({n_pix} pixels) — frames disabled")
                    A_sparse = None
                else:
                    d1, d2 = dims

            except Exception as ae:
                logger.warning(f"  {ds.name}: footprint load failed: {ae}")

        # ── Global contrast for movie frames ─────────────────────────────
        if movie is not None:
            sample = movie[::max(1, len(movie)//200)]
            vmin = float(np.percentile(sample, 1))
            vmax = float(np.percentile(sample, 99.5))
            # Align movie spatial dims to footprint dims if needed
            _, mh, mw = movie.shape
            if dims is not None:
                fh, fw = dims
                if mh != fh or mw != fw:
                    yo = (mh - fh) // 2
                    xo = (mw - fw) // 2
                    if yo >= 0 and xo >= 0:
                        movie = movie[:, yo:yo+fh, xo:xo+fw]
        else:
            vmin = vmax = None

        # ── Per-ROI SNR ranking ───────────────────────────────────────────
        roi_indices = ds.selected_roi_indices
        N_sel = len(roi_indices)
        snr_scores = np.zeros(N_sel)
        for j, orig_roi in enumerate(roi_indices):
            if orig_roi >= C_all.shape[0]:
                continue
            den = C_all[orig_roi]
            if noise is not None and orig_roi < len(noise) and noise[orig_roi] > 0:
                snr_scores[j] = float(np.max(den)) / float(noise[orig_roi])
            else:
                diff = np.diff(den)
                mad = 1.4826 * float(np.median(np.abs(diff - np.median(diff)))) / np.sqrt(2)
                if mad > 0:
                    snr_scores[j] = float(np.percentile(den, 95) - np.percentile(den, 50)) / mad

        rank_order = np.argsort(snr_scores)[::-1]  # highest SNR = rank 1

        frame_rate  = ds.frame_rate
        T           = C_all.shape[1]
        t_ax        = np.arange(T) / frame_rate
        ref_margin  = ref_margin_frames
        bl_window   = max(10, int(5.0 * frame_rate))  # 5s rolling baseline window

        rec_dir = os.path.join(out_root, ds.name)
        os.makedirs(rec_dir, exist_ok=True)

        for rank_pos, sel_j in enumerate(rank_order):
            orig_roi = int(roi_indices[sel_j])
            rank_num = rank_pos + 1

            if orig_roi >= C_all.shape[0]:
                continue

            den   = C_all[orig_roi]
            raw   = R_all[orig_roi] if R_all is not None and orig_roi < R_all.shape[0] else None
            spikes = S_all[orig_roi] if orig_roi < S_all.shape[0] else np.zeros(T)

            # ── Find peak transient frame ─────────────────────────────────
            peak_frame = int(np.argmax(den))

            # ── Find reference frame ──────────────────────────────────────
            # Lowest mean ROI fluorescence in the 10s window before the peak.
            # Using raw trace minimum for maximum contrast with the transient.
            search_window_frames = max(1, int(10.0 * frame_rate))
            search_start = max(0, peak_frame - search_window_frames)
            search_end   = max(0, peak_frame - ref_margin_frames)
            if raw is not None and search_end > search_start:
                window_vals = raw[search_start:search_end]
                ref_frame = search_start + int(np.argmin(window_vals))
            else:
                ref_frame = max(0, peak_frame - ref_margin_frames)

            # ── Get spatial crop bounds ───────────────────────────────────
            cy, cx = None, None
            fp_mask = None
            if A_sparse is not None and dims is not None and orig_roi < A_sparse.shape[1]:
                fp = A_sparse[:, orig_roi].toarray().ravel()
                if len(fp) == d1 * d2:
                    fp_2d = fp.reshape(d1, d2)
                    ys, xs = np.where(fp_2d > 0)
                    if len(ys) > 0:
                        cy, cx = int(np.mean(ys)), int(np.mean(xs))
                        y0 = max(0, cy - crop_radius)
                        y1 = min(d1, cy + crop_radius)
                        x0 = max(0, cx - crop_radius)
                        x1 = min(d2, cx + crop_radius)
                        fp_mask = (fp_2d[y0:y1, x0:x1] > 0).astype(np.uint8)

            # ── Extract frame crops ───────────────────────────────────────
            def _frame_to_rgb(frame_idx):
                """Return normalised RGB uint8 crop with contour overlay."""
                if movie is None or cy is None:
                    return None
                raw_crop = movie[frame_idx, y0:y1, x0:x1].astype(np.float64)
                norm = np.clip((raw_crop - vmin) / (vmax - vmin + 1e-10), 0, 1)
                rgb = np.stack([norm * 0.25, norm * 0.9, norm * 0.25], axis=-1)
                # Cyan contour overlay
                if fp_mask is not None:
                    from scipy.ndimage import binary_dilation
                    edge = binary_dilation(fp_mask > 0) & ~(fp_mask > 0)
                    mh, mw = min(fp_mask.shape[0], rgb.shape[0]), min(fp_mask.shape[1], rgb.shape[1])
                    rgb[:mh, :mw, 0][edge[:mh, :mw]] = 0.0
                    rgb[:mh, :mw, 1][edge[:mh, :mw]] = 1.0
                    rgb[:mh, :mw, 2][edge[:mh, :mw]] = 1.0
                return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

            ref_img  = _frame_to_rgb(ref_frame)
            peak_img = _frame_to_rgb(peak_frame)

            # ── Figure ────────────────────────────────────────────────────
            has_frames = ref_img is not None and peak_img is not None
            fig = plt.figure(figsize=(fig_w, fig_h), facecolor='black')

            if has_frames:
                gs = gridspec.GridSpec(2, 2, figure=fig,
                                       height_ratios=[1.6, 1],
                                       hspace=0.12, wspace=0.06,
                                       left=0.06, right=0.97,
                                       top=0.93, bottom=0.10)
                ax_ref  = fig.add_subplot(gs[0, 0])
                ax_peak = fig.add_subplot(gs[0, 1])
                ax_tr   = fig.add_subplot(gs[1, :])
            else:
                gs = gridspec.GridSpec(1, 1, figure=fig,
                                       left=0.06, right=0.97,
                                       top=0.93, bottom=0.10)
                ax_tr = fig.add_subplot(gs[0, 0])

            # ── Top: reference frame ──────────────────────────────────────
            if has_frames:
                ax_ref.imshow(ref_img, interpolation='nearest', aspect='equal')
                ax_ref.set_title(
                    f'Reference  t={ref_frame/frame_rate:.1f}s',
                    color='#8BA4B8', fontsize=8, pad=3)
                ax_ref.axis('off')

                ax_peak.imshow(peak_img, interpolation='nearest', aspect='equal')
                ax_peak.set_title(
                    f'Peak  t={peak_frame/frame_rate:.1f}s',
                    color='#00e676', fontsize=8, pad=3)
                ax_peak.axis('off')

            # ── Bottom: full trace ────────────────────────────────────────
            ax_tr.set_facecolor('black')

            # Raw trace
            if raw is not None:
                ax_tr.plot(t_ax, raw, color='#606878', linewidth=0.6,
                           alpha=0.7, zorder=2)

            # Denoised trace
            ax_tr.plot(t_ax, den, color='#00e676', linewidth=1.1,
                       zorder=3, label='Denoised')

            # Spike markers — all accepted spikes (≥2.5σ + s_min=0.1)
            spk_frames = np.where(spikes > 0)[0]
            if len(spk_frames) > 0:
                ax_tr.scatter(t_ax[spk_frames], den[spk_frames],
                              color='#D32F2F', s=18, zorder=5, marker='v')

            # Vertical line at peak and at reference frame
            ax_tr.axvline(peak_frame / frame_rate,
                          color='#00e676', linewidth=0.9, alpha=0.5,
                          linestyle='--', zorder=4)
            ax_tr.axvline(ref_frame / frame_rate,
                          color='#8BA4B8', linewidth=0.9, alpha=0.5,
                          linestyle='--', zorder=4)

            ax_tr.set_xlim(0, t_ax[-1])
            ax_tr.set_xlabel('Time (s)', color='white', fontsize=8)
            ax_tr.set_ylabel('ΔF/F₀', color='white', fontsize=8)
            ax_tr.tick_params(colors='white', labelsize=7)
            for sp in ax_tr.spines.values():
                sp.set_color('#333344')
            ax_tr.set_facecolor('#08090f')

            # ── Title ─────────────────────────────────────────────────────
            snr_val = snr_scores[sel_j]
            n_spk   = int(np.sum(spikes > 0))
            fig.suptitle(
                f'Rank {rank_num}  ·  ROI {orig_roi}  ·  '
                f'SNR {snr_val:.1f}  ·  {n_spk} spikes  ·  {ds.name}',
                color='white', fontsize=9, fontweight='bold', y=0.98)

            fname = os.path.join(rec_dir,
                                 f'rank{rank_num:03d}_roi{orig_roi:04d}.png')
            fig.savefig(fname, dpi=150, bbox_inches='tight',
                        facecolor='black')
            plt.close(fig)

        n_saved = len(rank_order)
        logger.info(f"  {ds.name}: saved {n_saved} ROI peak figures → {rec_dir}/")

    logger.info(f"ROI peak figures complete → {out_root}/")


def _fig_n_selected_distribution(datasets: List, output_dir: str,
                                 mutant_label: str = 'CEP41 R242H') -> None:
    """Stacked bar chart: selected vs unselected neurons per recording by day.

    Each recording is one stacked bar. Grey segment = unselected detections
    (n_neurons - n_selected). Coloured segment on top = selected neurons
    (blue = Control, orange = Mutant). Bars are grouped by organoid day,
    sorted oldest to newest, with recordings within each day sorted by
    total detection count descending.

    Saved to figures/Full Overview/n_selected_distribution.png at 300 DPI.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import re

    overview_dir = os.path.join(output_dir, 'figures', 'Full Overview')
    os.makedirs(overview_dir, exist_ok=True)

    CTRL_COLOR   = '#4472C4'
    MUT_COLOR    = '#ED7D31'
    UNSEL_COLOR  = '#CCCCCC'
    BG_COLOR     = 'white'
    TEXT_COLOR   = '#333333'
    GRID_COLOR   = '#DDDDDD'
    SEP_COLOR    = '#CCCCCC'
    MIN_N        = 5

    def _day_num(d):
        m = re.search(r'\d+', d)
        return int(m.group()) if m else 0

    # Group recordings by organoid day
    day_map: dict = {}
    for ds in datasets:
        day = _extract_organoid_id(ds.name)
        day_map.setdefault(day, []).append(ds)

    sorted_days = sorted(day_map.keys(), key=_day_num)

    # Figure width scales with number of recordings
    n_recs = len(datasets)
    fig_w  = max(12, n_recs * 0.35 + len(sorted_days) * 0.25)
    fig_h = 10
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')
    ax.set_facecolor(BG_COLOR)

    bar_w   = 0.10
    gap_rec = 0.02    # gap between recordings within a day
    gap_day = 0.10    # extra gap between days
    x       = 0.0
    x_mids  = []      # centre of each day group for x-tick
    x_labs  = []

    for day in sorted_days:
        recs = sorted(day_map[day], key=lambda d: d.n_neurons, reverse=True)
        group_xs = []

        for ds in recs:
            geno  = _extract_genotype(ds.name)
            color = CTRL_COLOR if geno == 'Control' else MUT_COLOR
            n_sel = ds.n_selected
            n_uns = max(0, ds.n_neurons - n_sel)

            # Grey unselected base
            ax.bar(x, n_uns, width=bar_w, bottom=0,
                   color=UNSEL_COLOR, edgecolor=BG_COLOR, linewidth=0.3, zorder=2)
            # Coloured selected on top
            ax.bar(x, n_sel, width=bar_w, bottom=n_uns,
                   color=color, edgecolor=BG_COLOR, linewidth=0.3, zorder=3)

            # Label selected count above the coloured segment
            if n_sel >= 2:
                ax.text(x, n_uns + n_sel + 4, str(n_sel),
                        ha='center', va='bottom', fontsize=14,
                        color=color, fontweight='bold', zorder=4, rotation=45)

            group_xs.append(x)
            x += bar_w + gap_rec

        # Day mid-point for x-tick label
        if group_xs:
            day_start = group_xs[0] - bar_w / 2
            day_end   = group_xs[-1] + bar_w / 2
            x_mids.append((day_start + day_end) / 2)
            x_labs.append(day)

        # Separator between days
        sep_x = x - gap_rec / 2 + gap_day / 2
        if day != sorted_days[-1]:
            ax.axvline(sep_x, color=SEP_COLOR, linewidth=0.7,
                       linestyle='--', alpha=0.8, zorder=1)
        x += gap_day

    ax.set_xticks(x_mids)
    ax.set_xticklabels(x_labs, fontsize=24, fontweight='bold', color=TEXT_COLOR,
                       rotation=45, ha='center')
    ax.set_xlim(-bar_w * 0.8, x - gap_day)
    ax.set_ylabel('Number of detections (ROIs)', fontsize=24, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=20)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    # Maximum height of stacked bars
    max_bar = 500  # max bar height 
    space_above = 150  # space to leave for legend
    ax.set_ylim(0, max_bar + space_above)

    # Legend
    n_ctrl  = sum(1 for ds in datasets if _extract_genotype(ds.name) == 'Control')
    n_mut   = sum(1 for ds in datasets if _extract_genotype(ds.name) == 'Mutant')
    handles = [
        mpatches.Patch(facecolor=CTRL_COLOR,  label=f'Control — active  ({n_ctrl} recordings)'),
        mpatches.Patch(facecolor=MUT_COLOR,   label=f'{mutant_label} — active   ({n_mut} recordings)'),
        mpatches.Patch(facecolor=UNSEL_COLOR, label='No detectable activity'),
    ]
    legend = ax.legend(handles=handles, fontsize=20,
                   loc='upper right',
                   bbox_to_anchor=(1, 1),
                   framealpha=0.25, edgecolor=SEP_COLOR,
                   labelcolor=TEXT_COLOR,
                   facecolor='white')

    n_sel_vals = [ds.n_selected for ds in datasets]
    median_n   = int(np.median(n_sel_vals))
    total_sel  = sum(n_sel_vals)
    total_det  = sum(ds.n_neurons for ds in datasets)
    n_below    = sum(1 for n in n_sel_vals if n < MIN_N)

    ax.set_title(
        'Active Detections by Day Age\n'
        f'median active = {median_n}  ·  total active = {total_sel}  ·  '
        f'total detections = {total_det}',
        fontsize=28, fontweight='bold', pad=16, color=TEXT_COLOR,
    )

    fig.tight_layout()
    path = os.path.join(overview_dir, 'n_selected_distribution.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved n_selected distribution figure: {path}")
    logger.info(f"  median n_selected={median_n}, total selected={total_sel}, "
                f"total detections={total_det}, "
                f"{n_below} recordings below n={MIN_N}")

# =============================================================================
# MAIN
# =============================================================================

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
    _fig_quality_gating(all_datasets, motion_max_threshold,
                        motion_residual_threshold, drift_threshold, overview_dir)

    if len(datasets) < 3:
        logger.error(f"Need at least 3 datasets after motion exclusion, "
                     f"got {len(datasets)}")
        return {}

    # ── Generate selected-neuron trace figures ──────────────────────────
    # Unified folder: figures/Selected Traces/{recording_name}.png
    # Shows exactly the quality-selected neurons used in statistical tests
    _fig_selected_traces(datasets, output_dir)

    # ── Generate per-ROI peak transient figures ───────────────────────────
    # figures/ROI Peak Frames/{recording_name}/rank{N}_roi{M}.png
    # One PNG per selected ROI: peak frame | reference frame + full trace
    if roi_peak_figures:
        try:
            _generate_roi_peak_figures(datasets, output_dir)
        except Exception as _rpf_err:
            logger.warning(f"ROI peak figures failed: {_rpf_err}")
            import traceback; traceback.print_exc()
    else:
        logger.info("ROI peak frame figures disabled (set quality.roi_peak_figures: true to enable)")

    # ── Generate neuron selection transparency figure ────────────────────
    _fig_neuron_selection(datasets, fig_dir, per_dataset_dir=per_dataset_dir)

    # ── Generate n_selected distribution figure ───────────────────────────
    # Shows histogram, ranked bar, and n vs correlation scatter — flags
    # recordings where low n makes correlation/synchrony unreliable (n < 5)
    try:
        _fig_n_selected_distribution(datasets, output_dir,
                                     mutant_label=mutant_label)
    except Exception as _nsd_err:
        logger.warning(f"n_selected distribution figure failed: {_nsd_err}")

    # ── Build feature matrix ─────────────────────────────────────────────
    X, names = build_feature_matrix(datasets)
    feat_labels = [fl for _, fl in FEATURE_NAMES]

    # ── Save feature CSV to data/ ──────────────────────────────────────────
    csv_path = os.path.join(data_dir, 'dataset_features.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset'] + feat_labels)
        for i, name in enumerate(names):
            writer.writerow([name] + [f'{X[i, j]:.4f}' for j in range(X.shape[1])])
    logger.info(f"Saved feature matrix: {csv_path}")

    # ── Save per-ROI listing to data/ ─────────────────────────────────────
    # Lists every ROI used in the analysis with its key metrics for verification
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
    logger.info(f"Saved selected ROI listing: {roi_csv_path} "
                f"({sum(d.n_selected for d in datasets)} ROIs across {len(datasets)} datasets)")

    # ── Save results JSON to data/ ─────────────────────────────────────────
    results = {
        'n_datasets': len(datasets),
        'n_features': len(feat_labels),
        'motion_excluded': [d.name for d in excluded],
        'n_excluded': len(excluded),
        'neuron_selection': {
            'mode': 'deconv_pass + distance_dedup',
            'note': 'All ROIs with deconvolved events included; duplicates within 15px removed',
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
        'amplitude_tracking': {
            d.name: d.amplitude_tracking
            for d in datasets
            if d.amplitude_tracking is not None
        },
    }

    json_path = os.path.join(data_dir, 'analysis_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # ── Statistical tests ────────────────────────────────────────────────
    try:
        stat_results = run_statistical_tests(datasets, output_dir)
        results['statistical_tests'] = stat_results['tests']
        # Re-save JSON with test results included
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Statistical tests failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # ── Dataset overview visualizations (UMAP, heatmap, summary) ──────────
    try:
        overview_results = run_dataset_overview(datasets, output_dir)
        results['dataset_overview'] = overview_results
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Dataset overview visualization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # ── Between-organoid comparison ──────────────────────────────────────
    try:
        between_results = run_between_organoid_tests(datasets, output_dir)
        results['between_organoid'] = between_results
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Between-organoid tests failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # ── Genotype comparison (v2.0) ───────────────────────────────────────
    try:
        genotype_results = run_genotype_comparison(datasets, output_dir,
                                                    mutant_label=mutant_label)
        results['genotype_comparison'] = genotype_results
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Genotype comparison failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # ── Core results: activity analysis (v1.6) ──────────────────────────
    try:
        activity_results = run_activity_analysis(datasets, output_dir,
                                                  mutant_label=mutant_label)
        results['activity_analysis'] = activity_results
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Activity analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # ── Figures ──────────────────────────────────────────────────────────
    generate_figures(datasets, X, feat_labels, names, fig_dir)

    # ── Flagged neuron inspection (independent of other figures) ─────────
    try:
        flagged_dir = os.path.join(fig_dir, 'per_dataset')
        os.makedirs(flagged_dir, exist_ok=True)
        _fig_flagged_neurons(datasets, flagged_dir)
    except Exception as e:
        print(f"\n!!! FLAGGED NEURON INSPECTION FAILED: {e}")
        logger.error(f"Flagged neuron inspection failed: {e}")
        import traceback
        tb = traceback.format_exc()
        print(tb)
        logger.error(tb)

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
