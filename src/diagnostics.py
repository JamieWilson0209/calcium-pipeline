"""
Diagnostics & Confidence Scoring
=================================

Post-detection diagnostic analysis and normalised confidence scoring.

This module provides two things:

1. **Temporal diagnostics** — transient count, activity fraction, baseline
   drift, and indicator-aware decay validation. These are the quality
   signals that the detection stage does not compute.

2. **Confidence scoring** — combines signals from two independent sources
   (detection quality and temporal diagnostics) into a single normalised
   0–1 score per neuron.

Nothing in this module filters or drops neurons. Confidence scores are
informational — the interactive gallery exposes a slider so the user
decides the threshold.
"""

import numpy as np
from scipy.signal import find_peaks
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT CONTAINER
# =============================================================================

@dataclass
class DiagnosticResult:
    """
    Container for all diagnostic outputs.

    Attributes
    ----------
    confidence : ndarray, shape (N,)
        Combined normalised confidence score (0–1).
    transient_count : ndarray, shape (N,)
        Number of detected calcium transients per component.
    activity_fraction : ndarray, shape (N,)
        Fraction of frames each component is active (0–1).
    baseline_drift : ndarray, shape (N,)
        Baseline instability as fraction of signal range (0 = stable).
    dynamics_validity : ndarray, shape (N,)
        Fraction of transients with indicator-consistent decay (0–1).
    detection_confidence : ndarray or None
        Seed-level confidence from contour detection stage.
    confidence_source : str
        Weighting scheme used (always ``"detection_temporal"``).
    """
    confidence: np.ndarray
    transient_count: np.ndarray
    activity_fraction: np.ndarray
    baseline_drift: np.ndarray
    dynamics_validity: np.ndarray
    detection_confidence: Optional[np.ndarray] = None
    confidence_source: str = "detection_temporal"

    @property
    def n_components(self) -> int:
        return len(self.confidence)

    def get_summary(self) -> Dict[str, object]:
        """Summary statistics for logging and reporting."""
        n = self.n_components
        if n == 0:
            return {'n_components': 0}

        def _stats(arr):
            valid = arr[np.isfinite(arr)]
            if len(valid) == 0:
                return {}
            return {
                'median': float(np.median(valid)),
                'mean': float(np.mean(valid)),
                'p10': float(np.percentile(valid, 10)),
                'p90': float(np.percentile(valid, 90)),
            }

        summary = {
            'n_components': n,
            'confidence': _stats(self.confidence),
            'transient_count': _stats(self.transient_count),
            'activity_fraction': _stats(self.activity_fraction),
            'baseline_drift': _stats(self.baseline_drift),
            'dynamics_validity': _stats(self.dynamics_validity),
            'confidence_source': self.confidence_source,
        }
        return summary

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        """Flat dict suitable for ``np.savez``."""
        d = {
            'confidence': self.confidence,
            'transient_count': self.transient_count,
            'activity_fraction': self.activity_fraction,
            'baseline_drift': self.baseline_drift,
            'dynamics_validity': self.dynamics_validity,
        }
        if self.detection_confidence is not None:
            d['detection_confidence'] = self.detection_confidence
        return d


# =============================================================================
# TEMPORAL DIAGNOSTICS
# =============================================================================

def _estimate_noise(trace: np.ndarray) -> float:
    """
    Robust noise estimate using MAD of temporal differences.

    The first-order temporal difference (frame-to-frame change) removes
    slow drift, bleach residuals, and low-frequency neuropil contamination.
    The MAD of these differences, scaled by 1/sqrt(2), gives a noise
    floor estimate that is robust to both slow artifacts AND calcium
    transients (which are sparse in the difference domain).

    This is a standard approach also used by CaImAn, Suite2p, and other pipelines.
    """
    T = len(trace)
    if T < 4:
        return max(np.std(trace), 1e-10)

    diff = np.diff(trace)
    mad = np.median(np.abs(diff - np.median(diff)))
    # MAD → σ conversion (1.4826) and diff → original scaling (1/√2)
    noise = 1.4826 * mad / np.sqrt(2)
    return max(noise, 1e-10)


def _detect_calcium_events(
    trace: np.ndarray,
    frame_rate: float,
    decay_time: float = 1.0,
    min_snr: float = 5.0,
    min_prominence_snr: float = 3.5,
) -> np.ndarray:
    """
    Detect genuine calcium transient events in a single trace.

    Designed for non-deconvolved traces from the weighted-average trace
    extraction path. Uses the MAD-of-differences noise estimator (robust to
    slow drift and neuropil) with conservative peak thresholds.

    A valid event must:

    1. Exceed ``min_snr`` × noise above the 20th-percentile baseline
       (default 5.0 — handles structured noise in raw traces)
    2. Have prominence ≥ ``min_prominence_snr`` × noise (default 3.5 —
       rejects shoulders on larger events)
    3. Be separated from the previous event by at least
       ``2 × decay_time`` seconds (indicator refractory period)

    No additional shape-based filtering (rise/decay checks) is applied,
    as these are unreliable at low frame rates (≤10 Hz) where transient
    dynamics span only 2-4 frames.

    Parameters
    ----------
    trace : 1D array
    frame_rate : float
    decay_time : float
    min_snr : float
        Minimum peak height in noise units (default 5.0).
    min_prominence_snr : float
        Minimum prominence in noise units (default 3.5).

    Returns
    -------
    peaks : 1D int array — indices of detected events
    """
    baseline = np.percentile(trace, 20)
    noise = _estimate_noise(trace)
    normed = (trace - baseline) / noise

    # Refractory: 2× decay ensures previous transient has decayed to ~14%
    refractory_frames = max(int(2.0 * decay_time * frame_rate), 3)

    try:
        peaks, props = find_peaks(
            normed,
            height=min_snr,
            distance=refractory_frames,
            prominence=min_prominence_snr,
        )
    except Exception:
        return np.array([], dtype=int)

    return peaks


def compute_transient_count(
    C: np.ndarray,
    frame_rate: float,
    min_snr: float = 5.0,
    decay_time: float = 1.0,
) -> np.ndarray:
    """
    Count calcium transients per component.

    Uses ``_detect_calcium_events`` with conservative thresholds
    (min_snr=5.0, prominence=3.5) to avoid counting noise as events.
    Critical for the weighted-average trace extraction path where traces contain
    structured noise from neuropil and bleach correction residuals.

    Parameters
    ----------
    C : array, shape (N, T)
    frame_rate : float
    min_snr : float
        Minimum peak height in noise units (default 5.0).
    decay_time : float
        Expected indicator decay time in seconds (default 1.0).

    Returns
    -------
    array, shape (N,), dtype int
    """
    n = C.shape[0]
    counts = np.zeros(n, dtype=int)

    for i in range(n):
        peaks = _detect_calcium_events(
            C[i, :], frame_rate, decay_time=decay_time, min_snr=min_snr,
        )
        counts[i] = len(peaks)

    return counts


def compute_activity_fraction(
    C: np.ndarray,
    threshold_sigma: float = 3.0,
) -> np.ndarray:
    """
    Fraction of frames each component is above baseline + threshold.

    Uses the PSD-based noise estimator. Threshold of 3.0σ (raised
    from 2.0) to reduce false activity from structured noise.

    Parameters
    ----------
    C : array, shape (N, T)
    threshold_sigma : float
        Activation threshold in noise units (default 3.0).

    Returns
    -------
    array, shape (N,), in [0, 1]
    """
    n, T = C.shape
    frac = np.zeros(n)

    for i in range(n):
        trace = C[i, :]
        baseline = np.percentile(trace, 20)
        noise = _estimate_noise(trace)
        frac[i] = (trace > baseline + threshold_sigma * noise).sum() / T

    return frac


def compute_baseline_drift(
    C: np.ndarray,
    window_size: int = 100,
) -> np.ndarray:
    """
    Baseline instability as fraction of signal range (0 = stable).

    Parameters
    ----------
    C : array, shape (N, T)
    window_size : int

    Returns
    -------
    array, shape (N,)
    """
    n, T = C.shape
    drift = np.zeros(n)

    if T < window_size * 2:
        return drift

    n_win = T // window_size

    for i in range(n):
        trace = C[i, :]
        sig_range = trace.max() - trace.min()
        if sig_range < 1e-10:
            continue

        baselines = [
            np.percentile(trace[w * window_size:(w + 1) * window_size], 20)
            for w in range(n_win)
        ]
        drift[i] = (max(baselines) - min(baselines)) / sig_range

    return drift


def compute_dynamics_validity(
    C: np.ndarray,
    frame_rate: float,
    decay_time: float,
    tolerance: float = 2.0,
) -> np.ndarray:
    """
    Fraction of transients with indicator-consistent decay dynamics.

    For each peak, measures time to decay to 37 % (1/e) and checks
    whether it falls within ``[decay_time / tolerance, decay_time × tolerance]``.

    Parameters
    ----------
    C : array, shape (N, T)
    frame_rate : float
    decay_time : float
        Expected 1/e decay time in seconds.
    tolerance : float
        Multiplicative band (default 2.0 → 0.5× to 2×).

    Returns
    -------
    array, shape (N,), in [0, 1]
    """
    n, T = C.shape
    validity = np.zeros(n)

    min_frames = int(decay_time / tolerance * frame_rate)
    max_frames = int(decay_time * tolerance * frame_rate)

    for i in range(n):
        trace = C[i, :]
        baseline = np.percentile(trace, 20)
        normed = trace - baseline
        noise = _estimate_noise(trace)

        peaks = _detect_calcium_events(
            trace, frame_rate, decay_time=decay_time, min_snr=3.5,
        )

        if len(peaks) == 0:
            validity[i] = 0.5
            continue

        n_valid = 0.0
        for pk in peaks:
            target = normed[pk] * 0.37
            decay_idx = None
            for j in range(pk + 1, min(pk + max_frames + 10, T)):
                if normed[j] <= target:
                    decay_idx = j
                    break

            if decay_idx is not None:
                df = decay_idx - pk
                if min_frames <= df <= max_frames:
                    n_valid += 1.0
            else:
                n_valid += 0.5  # slow decay — partial credit

        validity[i] = n_valid / len(peaks)

    return validity


def compute_temporal_diagnostics(
    C: np.ndarray,
    frame_rate: float,
    decay_time: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all four temporal diagnostic arrays.

    Parameters
    ----------
    C : array, shape (N, T)
    frame_rate : float
    decay_time : float
        Expected indicator decay time in seconds.

    Returns
    -------
    transient_count : array (N,)
    activity_fraction : array (N,)
    baseline_drift : array (N,)
    dynamics_validity : array (N,)
    """
    tc = compute_transient_count(C, frame_rate, decay_time=decay_time)
    af = compute_activity_fraction(C)
    bd = compute_baseline_drift(C, window_size=max(int(frame_rate * 10), 20))
    dv = compute_dynamics_validity(C, frame_rate, decay_time)
    return tc, af, bd, dv


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================

def _temporal_score(
    transient_count: np.ndarray,
    activity_fraction: np.ndarray,
    baseline_drift: np.ndarray,
    dynamics_validity: np.ndarray,
) -> np.ndarray:
    """Combine temporal diagnostics into a single 0–1 score."""
    has_activity = np.clip(transient_count / 5.0, 0.0, 1.0)

    activity_ok = np.where(
        (activity_fraction >= 0.01) & (activity_fraction <= 0.6),
        1.0, 0.5,
    )

    baseline_ok = np.clip(1.0 - baseline_drift / 0.3, 0.0, 1.0)

    return (
        0.35 * has_activity
        + 0.15 * activity_ok
        + 0.20 * baseline_ok
        + 0.30 * dynamics_validity
    )


def compute_confidence(
    n_components: int,
    *,
    detection_confidence: Optional[np.ndarray] = None,
    transient_count: Optional[np.ndarray] = None,
    activity_fraction: Optional[np.ndarray] = None,
    baseline_drift: Optional[np.ndarray] = None,
    dynamics_validity: Optional[np.ndarray] = None,
    boundary_touching: Optional[np.ndarray] = None,
    contour_success: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str]:
    """
    Combine all available quality signals into a normalised 0–1 confidence.

    Weights: 0.45 detection + 0.55 temporal.

    Parameters
    ----------
    n_components : int
        Number of neurons.
    detection_confidence : array (N,), optional
        Seed-level confidence from contour extraction (0–1).
    transient_count, activity_fraction, baseline_drift, dynamics_validity
        Temporal diagnostics (from ``compute_temporal_diagnostics``).
    boundary_touching : bool array (N,), optional
        Whether each neuron's contour touches the FOV edge.
    contour_success : bool array (N,), optional
        Whether contour extraction succeeded for each seed.

    Returns
    -------
    confidence : array (N,), in [0, 1]
    source : str
        ``"detection_temporal"``
    """
    if n_components == 0:
        return np.array([]), "detection_temporal"

    # --- Detection score ---
    if detection_confidence is not None:
        det_score = np.asarray(detection_confidence, dtype=np.float64)
    else:
        det_score = np.full(n_components, 0.5)

    # --- Temporal score ---
    if transient_count is not None:
        temp_score = _temporal_score(
            transient_count, activity_fraction,
            baseline_drift, dynamics_validity,
        )
    else:
        temp_score = np.full(n_components, 0.5)

    # --- Combine ---
    confidence = (
        0.45 * det_score
        + 0.55 * temp_score
    )
    source = "detection_temporal"

    # --- Penalties ---
    if boundary_touching is not None:
        confidence = np.where(boundary_touching, confidence - 0.10, confidence)
    if contour_success is not None:
        confidence = np.where(~contour_success, confidence - 0.05, confidence)

    confidence = np.clip(confidence, 0.0, 1.0)
    return confidence, source


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_diagnostics(
    C: np.ndarray,
    frame_rate: float,
    decay_time: float,
    *,
    detection_confidence: Optional[np.ndarray] = None,
    boundary_touching: Optional[np.ndarray] = None,
    contour_success: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> DiagnosticResult:
    """
    Run full diagnostic analysis and confidence scoring.

    This is the primary entry point. Call after trace extraction
    with whatever signals are available.

    Parameters
    ----------
    C : array, shape (N, T)
        Temporal traces.
    frame_rate : float
        Acquisition rate in Hz.
    decay_time : float
        Expected indicator decay time in seconds.
    detection_confidence : array (N,), optional
        From ``ContourSeedResult.confidence``.
    boundary_touching : bool array (N,), optional
    contour_success : bool array (N,), optional
    verbose : bool

    Returns
    -------
    DiagnosticResult
    """
    n = C.shape[0] if C is not None and C.ndim == 2 else 0

    if n == 0:
        empty = np.array([])
        return DiagnosticResult(
            confidence=empty, transient_count=empty,
            activity_fraction=empty, baseline_drift=empty,
            dynamics_validity=empty,
        )

    if verbose:
        logger.info(f"Running diagnostics for {n} components "
                    f"(decay_time={decay_time}s, frame_rate={frame_rate} Hz)")

    tc, af, bd, dv = compute_temporal_diagnostics(C, frame_rate, decay_time)

    confidence, source = compute_confidence(
        n,
        detection_confidence=detection_confidence,
        transient_count=tc,
        activity_fraction=af,
        baseline_drift=bd,
        dynamics_validity=dv,
        boundary_touching=boundary_touching,
        contour_success=contour_success,
    )

    result = DiagnosticResult(
        confidence=confidence,
        transient_count=tc,
        activity_fraction=af,
        baseline_drift=bd,
        dynamics_validity=dv,
        detection_confidence=detection_confidence,
        confidence_source=source,
    )

    if verbose:
        s = result.get_summary()
        c = s.get('confidence', {})
        logger.info(
            f"  Confidence ({source}): "
            f"median={c.get('median', 0):.2f}, "
            f"p10={c.get('p10', 0):.2f}, "
            f"p90={c.get('p90', 0):.2f}"
        )
        t = s.get('transient_count', {})
        logger.info(
            f"  Transients: median={t.get('median', 0):.0f}, "
            f"dynamics validity: median={s.get('dynamics_validity', {}).get('median', 0):.2f}"
        )

    return result


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_diagnostic_figures(
    result: DiagnosticResult,
    output_dir: str,
    *,
    A=None,
    dims: Optional[tuple] = None,
    C: Optional[np.ndarray] = None,
    frame_rate: Optional[float] = None,
    decay_time: float = 1.0,
    max_projection: Optional[np.ndarray] = None,
    fmt: str = "png",
    dpi: int = 150,
) -> List[str]:
    """
    Generate diagnostic summary figures.

    Produces up to three figures:

    1. **neuron_quality_summary.{fmt}** — adaptive layout depending on
       whether detection metrics are available. Shows confidence distribution,
       temporal diagnostics, and quality breakdown.

    2. **quality_detail.{fmt}** — baseline stability, event dynamics,
       confidence breakdown, and ranked confidence curve.

    3. **spatial_activity_map.{fmt}** — (if A and dims provided) spatial
       map showing neuron locations colour-coded by event frequency, with
       an optional temporal activity raster.

    Parameters
    ----------
    result : DiagnosticResult
    output_dir : str
    A : sparse matrix, optional
        Spatial footprints (n_pixels, N) for spatial map.
    dims : (H, W), optional
        Image dimensions for spatial map.
    C : array (N, T), optional
        Temporal traces for activity raster.
    frame_rate : float, optional
        Frame rate for time axis labelling.
    fmt, dpi : str, int
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import os

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    n = result.n_components

    if n == 0:
        logger.warning("No components — skipping diagnostic figures")
        return paths

    # ── Colour palette ────────────────────────────────────────────────
    C_BLUE   = '#3b82f6'
    C_GREEN  = '#10b981'
    C_AMBER  = '#f59e0b'
    C_RED    = '#ef4444'
    C_GREY   = '#94a3b8'
    C_LGREY  = '#f1f5f9'

    def _style(ax, title, xlabel=None, ylabel=None):
        ax.set_title(title, fontsize=11, fontweight='600', pad=8)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=9, color='#475569')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9, color='#475569')
        ax.tick_params(labelsize=8, colors='#64748b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')

    def _annotate_median(ax, values, color='#1e293b'):
        med = np.median(values)
        ax.axvline(med, color=color, linestyle='--', linewidth=1.2, alpha=0.8)
        ax.text(med, ax.get_ylim()[1] * 0.92, f'median {med:.2f}',
                ha='center', fontsize=7.5, color=color,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

    # ── Figure 1: Neuron Quality Summary ─────────────────────────────

    fig1 = plt.figure(figsize=(15, 8.5))
    gs = GridSpec(2, 3, figure=fig1, width_ratios=[1, 1, 1])
    axes = np.empty((2, 3), dtype=object)
    # Top row: confidence (wide), detection confidence
    axes[0, 0] = fig1.add_subplot(gs[0, 0:2])  # wide confidence
    axes[0, 1] = None  # placeholder
    axes[0, 2] = fig1.add_subplot(gs[0, 2])
    axes[1, 0] = fig1.add_subplot(gs[1, 0])
    axes[1, 1] = fig1.add_subplot(gs[1, 1])
    axes[1, 2] = fig1.add_subplot(gs[1, 2])

    scoring_label = 'Detection + Temporal'
    fig1.suptitle(
        f'Neuron Quality Summary — {n} neurons\n'
        f'Scoring: {scoring_label}',
        fontsize=13, fontweight='700', y=0.98, color='#1e293b',
    )

    # 1a. Confidence distribution
    ax = axes[0, 0]
    ax.hist(result.confidence, bins=30, range=(0, 1), color=C_BLUE,
            alpha=0.85, edgecolor='white', linewidth=0.5)
    _annotate_median(ax, result.confidence)
    q25, q75 = np.percentile(result.confidence, [25, 75])
    ax.axvspan(q25, q75, alpha=0.08, color=C_BLUE)
    ax.text(0.97, 0.92, f'IQR: {q25:.2f}–{q75:.2f}',
            transform=ax.transAxes, ha='right', fontsize=8, color=C_GREY)
    _style(ax, 'Overall Confidence Score', 'confidence (0–1)', 'number of neurons')

    # 1b. Detection confidence if available
    ax = axes[0, 2]
    if result.detection_confidence is not None:
        ax.hist(result.detection_confidence, bins=25, range=(0, 1),
                color=C_GREEN, alpha=0.85, edgecolor='white', linewidth=0.5)
        _annotate_median(ax, result.detection_confidence, C_GREEN)
        _style(ax, 'Detection Stage Confidence', 'seed confidence (0–1)', 'number of neurons')
    else:
        ax.text(0.5, 0.5, 'Detection confidence\nnot available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color=C_GREY)
        _style(ax, 'Detection Stage Confidence')

    # 1d. Calcium events per neuron
    ax = axes[1, 0]
    tc = result.transient_count
    max_bin = min(int(np.percentile(tc, 99)) + 5, int(tc.max()) + 1) if tc.max() > 0 else 10
    ax.hist(tc, bins=min(max_bin, 40), color=C_AMBER, alpha=0.85,
            edgecolor='white', linewidth=0.5)
    _annotate_median(ax, tc, C_AMBER)
    _style(ax, 'Calcium Events per Neuron', 'number of detected events', 'number of neurons')

    # 1e. Time spent active
    ax = axes[1, 1]
    af_pct = result.activity_fraction * 100
    ax.hist(af_pct, bins=30, range=(0, 100), color=C_AMBER, alpha=0.85,
            edgecolor='white', linewidth=0.5)
    ax.axvspan(1, 60, alpha=0.06, color=C_GREEN)
    ax.text(30, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else 0.5,
            'typical range', ha='center', fontsize=7, color=C_GREEN, alpha=0.7)
    _annotate_median(ax, af_pct, C_AMBER)
    _style(ax, 'Time Spent Active', '% of recording active', 'number of neurons')

    # 1f. Indicator decay match
    ax = axes[1, 2]
    dv_pct = result.dynamics_validity * 100
    ax.hist(dv_pct, bins=20, range=(0, 100), color=C_AMBER, alpha=0.85,
            edgecolor='white', linewidth=0.5)
    _annotate_median(ax, dv_pct, C_AMBER)
    _style(ax, 'Calcium Indicator Decay Match', '% events with expected decay', 'number of neurons')

    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    p1 = os.path.join(output_dir, f'neuron_quality_summary.{fmt}')
    fig1.savefig(p1, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    paths.append(p1)
    logger.info(f"  Saved: {p1}")

    # ── Figure 2: Quality Detail ─────────────────────────────────────

    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9))
    fig2.suptitle('Quality Detail', fontsize=13, fontweight='700',
                  y=0.98, color='#1e293b')

    # 2a. Baseline stability
    ax = axes2[0, 0]
    drift_pct = result.baseline_drift * 100
    ax.hist(drift_pct, bins=30, color=C_AMBER, alpha=0.85,
            edgecolor='white', linewidth=0.5)
    ax.axvline(30, color=C_RED, linestyle='--', linewidth=1, alpha=0.7)
    n_high = (result.baseline_drift > 0.3).sum()
    if n_high > 0:
        ax.text(0.97, 0.92, f'{n_high} neuron{"s" if n_high > 1 else ""} with unstable baseline',
                transform=ax.transAxes, ha='right', fontsize=8, color=C_RED)
    _style(ax, 'Baseline Stability', 'baseline drift (% of signal range)', 'number of neurons')

    # 2b. Events vs decay match (coloured by confidence)
    ax = axes2[0, 1]
    sc = ax.scatter(
        result.transient_count, result.dynamics_validity * 100,
        c=result.confidence, cmap='RdYlGn', s=14, alpha=0.6,
        edgecolors='none', vmin=0, vmax=1,
    )
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('confidence', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _style(ax, 'Events vs Decay Match', 'calcium events', '% events with expected decay')

    # 2c. Confidence source breakdown
    ax = axes2[1, 0]
    det_contrib = 0.45 * np.mean(
        result.detection_confidence
        if result.detection_confidence is not None
        else np.full(n, 0.5)
    )
    temp_contrib = 0.55 * np.mean(_temporal_score(
        result.transient_count, result.activity_fraction,
        result.baseline_drift, result.dynamics_validity,
    ))
    labels = ['Spatial\nDetection', 'Temporal\nActivity']
    values = [det_contrib, temp_contrib]
    colours = [C_BLUE, C_AMBER]

    bars = ax.bar(labels, values, color=colours, alpha=0.85, edgecolor='white',
                  linewidth=1.5, width=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='600', color='#334155')
    ax.set_ylim(0, max(values) * 1.3 + 0.01)
    _style(ax, 'Mean Confidence Breakdown', None, 'weighted contribution to score')

    # 2d. Ranked confidence curve
    ax = axes2[1, 1]
    sorted_conf = np.sort(result.confidence)[::-1]
    ranks = np.arange(1, n + 1)
    ax.fill_between(ranks, sorted_conf, alpha=0.2, color=C_BLUE)
    ax.plot(ranks, sorted_conf, color=C_BLUE, linewidth=1.8)
    med = np.median(result.confidence)
    ax.axhline(med, color='#334155', linestyle='--', linewidth=1,
               label=f'median = {med:.2f}')
    ax.axhline(0.5, color=C_GREY, linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xlim(1, n)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower left',
              frameon=True, facecolor='white', edgecolor='#e2e8f0')
    _style(ax, 'Confidence by Neuron Rank', 'neuron (ranked best → worst)', 'confidence')

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = os.path.join(output_dir, f'quality_detail.{fmt}')
    fig2.savefig(p2, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    paths.append(p2)
    logger.info(f"  Saved: {p2}")

    # ── Figure 3: Spatial Activity Map ───────────────────────────────

    if A is not None and dims is not None:
        try:
            paths.extend(_generate_spatial_activity_map(
                result, A, dims, output_dir,
                C=C, frame_rate=frame_rate, decay_time=decay_time,
                max_projection=max_projection,
                fmt=fmt, dpi=dpi,
            ))
        except Exception as exc:
            logger.warning(f"  Spatial activity map failed: {exc}")

    return paths


def _generate_spatial_activity_map(
    result: DiagnosticResult,
    A,
    dims: tuple,
    output_dir: str,
    *,
    C: Optional[np.ndarray] = None,
    frame_rate: Optional[float] = None,
    decay_time: float = 1.0,
    max_projection: Optional[np.ndarray] = None,
    fmt: str = "png",
    dpi: int = 150,
) -> List[str]:
    """
    Generate spatial–temporal activity analysis.

    Six-panel figure (3×2):

    Row 1:
    1. **Max projection reference** — raw max projection with neuron
       centroids overlaid. Serves as orientation reference so the
       reader can verify spatial alignment with the raw movie.
    2. **Activity hotspot map (KDE)** — Gaussian-smoothed event rate
       density, weighted by event frequency.
    3. **Neuron confidence map** — FOV with neurons colour-coded by
       confidence score, size scaled by event count.

    Row 2:
    4. **Calcium event raster** — detected events as tick marks,
       neurons sorted by event frequency, colour-coded by confidence.
    5. **Population activity** — histogram of summed events per time
       bin, with synchronous burst detection.

    Parameters
    ----------
    max_projection : 2D array (H, W), optional
        Raw max-intensity projection from the movie. If None, a
        composite footprint image is used as background.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from scipy.sparse import issparse
    from scipy.ndimage import gaussian_filter
    import os

    paths = []
    n = result.n_components
    H, W = dims

    # --- Compute centroids ---
    A_dense = A.toarray() if issparse(A) else np.asarray(A)
    cy = np.zeros(n)
    cx = np.zeros(n)
    for i in range(n):
        col = A_dense[:, i]
        if col.sum() > 0:
            fp = col.reshape(dims)
            yy, xx = np.mgrid[:H, :W]
            total = fp.sum()
            cy[i] = (yy * fp).sum() / total
            cx[i] = (xx * fp).sum() / total

    # --- Event rate ---
    tc = result.transient_count.copy()
    has_time = frame_rate is not None and frame_rate > 0
    if has_time and C is not None:
        T_sec = C.shape[1] / frame_rate
        event_rate = tc / max(T_sec / 60.0, 0.01)
        rate_label = 'events / min'
    else:
        event_rate = tc.astype(float)
        rate_label = 'total events'

    # --- Detect event times for raster ---
    event_times = []
    if C is not None:
        for i in range(n):
            peaks = _detect_calcium_events(
                C[i, :], frame_rate or 2.0, decay_time=decay_time,
            )
            event_times.append(peaks)

        for i in range(n):
            tc[i] = len(event_times[i])
        if has_time:
            T_sec = C.shape[1] / frame_rate
            event_rate = tc / max(T_sec / 60.0, 0.01)
        else:
            event_rate = tc.astype(float)
    else:
        event_times = [np.array([]) for _ in range(n)]

    # --- Background images ---
    # Composite footprint background
    max_fp = np.zeros(dims, dtype=np.float32)
    for i in range(n):
        fp = A_dense[:, i].reshape(dims)
        max_fp = np.maximum(max_fp, fp)
    bg_fp = max_fp / (max_fp.max() + 1e-10)

    # Max projection (normalised for display)
    if max_projection is not None:
        mp = max_projection.astype(np.float32)
        mp_display = (mp - mp.min()) / (mp.max() - mp.min() + 1e-10)
    else:
        mp_display = bg_fp

    # --- Colour palette ---
    C_GREY = '#94a3b8'

    def _style(ax, title, xlabel=None, ylabel=None):
        ax.set_title(title, fontsize=11, fontweight='600', pad=8, color='#1e293b')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=9, color='#475569')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9, color='#475569')
        ax.tick_params(labelsize=8, colors='#64748b')

    # ── Layout: 3 columns × 2 rows ──────────────────────────────────
    # Top row: 3 equal-sized spatial panels (no colorbar stealing space)
    # Bottom row: full-width population activity

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.22, wspace=0.06,
                  height_ratios=[1.2, 0.65])
    ax_ref  = fig.add_subplot(gs[0, 0])
    ax_kde  = fig.add_subplot(gs[0, 1])
    ax_conf = fig.add_subplot(gs[0, 2])
    ax_pop  = fig.add_subplot(gs[1, :])  # full width

    fig.suptitle('Spatial & Temporal Activity Analysis',
                 fontsize=14, fontweight='700', y=0.99, color='#1e293b')

    def _clean_spatial(ax):
        """Remove all ticks and spines from spatial panels."""
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ── Panel 1: Max projection + ROI contours ────────────────────────

    ax_ref.imshow(mp_display, cmap='gray', vmin=0, vmax=1)
    for i in range(n):
        fp = A_dense[:, i].reshape(dims)
        if fp.max() > 0:
            ax_ref.contour(fp, levels=[fp.max() * 0.25],
                           colors=['#22d3ee'], linewidths=0.5, alpha=0.7)
    ax_ref.set_xlim(0, W); ax_ref.set_ylim(H, 0); ax_ref.set_aspect('equal')
    _clean_spatial(ax_ref)
    title_ref = 'Max Projection + ROI Contours' if max_projection is not None \
                else 'Composite Footprints + ROI Contours'
    ax_ref.set_title(title_ref, fontsize=11, fontweight='600', pad=8, color='#1e293b')

    # ── Panel 2: KDE activity heatmap ─────────────────────────────────

    density = np.zeros((H, W), dtype=np.float32)
    for i in range(n):
        yi, xi = int(round(cy[i])), int(round(cx[i]))
        if 0 <= yi < H and 0 <= xi < W:
            density[yi, xi] += event_rate[i]
    density_smooth = gaussian_filter(density, sigma=max(H, W) / 20.0)

    ax_kde.imshow(mp_display, cmap='gray', vmin=0, vmax=1, alpha=0.3)
    im_kde = ax_kde.imshow(density_smooth, cmap='magma', alpha=0.75,
                           vmin=0, vmax=np.percentile(density_smooth, 98) + 1e-10)
    ax_kde.scatter(cx, cy, s=6, c='white', edgecolors='none', alpha=0.35, zorder=5)
    ax_kde.set_xlim(0, W); ax_kde.set_ylim(H, 0); ax_kde.set_aspect('equal')
    _clean_spatial(ax_kde)
    ax_kde.set_title('Activity Hotspot Map (KDE)', fontsize=11,
                     fontweight='600', pad=8, color='#1e293b')

    # Inset colorbar — sits inside the panel, bottom-right
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax1 = inset_axes(ax_kde, width="35%", height="3%", loc='lower right',
                      borderpad=1.5)
    cb1 = fig.colorbar(im_kde, cax=cax1, orientation='horizontal')
    cb1.set_label(rate_label, fontsize=7, color='white')
    cb1.ax.tick_params(labelsize=6, colors='white', length=0)
    cb1.outline.set_visible(False)

    # ── Panel 3: Confidence spatial map ───────────────────────────────

    ax_conf.imshow(mp_display, cmap='gray', vmin=0, vmax=1, alpha=0.3)
    size = np.clip(tc / (tc.max() + 1e-10) * 80 + 6, 6, 100)
    sc = ax_conf.scatter(
        cx, cy, c=result.confidence, cmap='RdYlGn',
        s=size, vmin=0, vmax=1,
        edgecolors='white', linewidths=0.3, alpha=0.85, zorder=5,
    )
    ax_conf.set_xlim(0, W); ax_conf.set_ylim(H, 0); ax_conf.set_aspect('equal')
    _clean_spatial(ax_conf)
    ax_conf.set_title(f'Neuron Confidence Map ({n} neurons)', fontsize=11,
                      fontweight='600', pad=8, color='#1e293b')

    cax2 = inset_axes(ax_conf, width="35%", height="3%", loc='lower right',
                      borderpad=1.5)
    cb2 = fig.colorbar(sc, cax=cax2, orientation='horizontal')
    cb2.set_label('confidence', fontsize=7, color='white')
    cb2.ax.tick_params(labelsize=6, colors='white', length=0)
    cb2.outline.set_visible(False)

    # ── Bottom panel: Population activity (frame-based, full width) ────

    if C is not None:
        T_frames = C.shape[1]

        # Per-frame event count across all neurons
        pop_per_frame = np.zeros(T_frames, dtype=int)
        for i in range(n):
            peaks = event_times[i]
            for pk in peaks:
                if 0 <= pk < T_frames:
                    pop_per_frame[pk] += 1

        # Smooth with a small window for visualisation
        from scipy.ndimage import uniform_filter1d
        smooth_width = max(3, int(frame_rate * 0.5)) if has_time else 3
        pop_smooth = uniform_filter1d(pop_per_frame.astype(float), size=smooth_width)

        frames = np.arange(T_frames)
        ax_pop.fill_between(frames, pop_per_frame, alpha=0.3, color='#3b82f6',
                            step='mid')
        ax_pop.step(frames, pop_per_frame, where='mid',
                    color='#3b82f6', alpha=0.4, linewidth=0.5)
        ax_pop.plot(frames, pop_smooth, color='#1e40af',
                    linewidth=1.8, alpha=0.9)

        mean_rate = pop_per_frame.mean()
        ax_pop.axhline(mean_rate, color='#94a3b8', linestyle='--',
                       linewidth=1, alpha=0.6,
                       label=f'mean: {mean_rate:.2f} events/frame')

        # Flag synchronous bursts (>2σ above mean)
        std_rate = pop_per_frame.std()
        if std_rate > 0:
            burst_thresh = mean_rate + 2 * std_rate
            burst_mask = pop_per_frame > burst_thresh
            if burst_mask.any():
                ax_pop.fill_between(
                    frames, 0, pop_per_frame,
                    where=burst_mask, alpha=0.3, color='#ef4444',
                    step='mid',
                    label=f'synchronous bursts (>{burst_thresh:.0f})',
                )

        ax_pop.set_xlim(0, T_frames)
        ax_pop.set_ylim(0, None)
        ax_pop.legend(fontsize=7.5, loc='upper right',
                      frameon=True, facecolor='white', edgecolor='#e2e8f0')
        _style(ax_pop, 'Population Activity',
               'frame', 'simultaneous events')
    else:
        ax_pop.text(0.5, 0.5, 'Temporal data not available',
                    ha='center', va='center', transform=ax_pop.transAxes,
                    fontsize=11, color=C_GREY)
        _style(ax_pop, 'Population Activity')

    p3 = os.path.join(output_dir, f'spatial_activity_map.{fmt}')
    fig.savefig(p3, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    paths.append(p3)
    logger.info(f"  Saved: {p3}")

    return paths



