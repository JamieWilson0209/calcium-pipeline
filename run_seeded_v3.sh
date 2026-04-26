#!/bin/bash
#$ -N calcium_pipeline
#$ -cwd
#$ -l h_rt=06:00:00
#$ -l h_vmem=32G
#$ -j y
#$ -V


# Developed as needed over time, needs reworking and script separation

# =============================================================================
# Calcium Pipeline v2.1 with Contour-Based Detection
# =============================================================================
#
# Usage via run.sh (recommended):
#   bash run.sh single --movie /path/to/file.nd2
#   bash run.sh batch  --data-dir /path/to/data
#
# Or run this script directly (inherits MOVIE / OUTPUT_DIR / CONFIG_PATH from env):
#   MOVIE=/path/to/movie.nd2 bash run_seeded_v3.sh
#
# =============================================================================

# Create logs directory immediately (SGE needs it before job starts,
# but this ensures it exists for interactive runs too)
mkdir -p logs 2>/dev/null || true

# NOTE: set -e is applied AFTER environment setup below, because module/conda
# activation can return non-zero even when successful.

echo "=========================================="
echo "Calcium Pipeline v2.1"
echo "with Contour-Based Detection"
echo "=========================================="
echo "Job ID: ${JOB_ID:-interactive}"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

# Check if running on compute node vs login node
if [[ $(hostname) == login* ]]; then
    echo "WARNING: Running on login node - memory limits may apply!"
    echo "Consider submitting via: qsub run_seeded_v3.sh"
fi

# Show available space on /tmp (local to compute node)
echo ""
echo "Local /tmp space:"
df -h /tmp 2>/dev/null || echo "  (could not check /tmp)"
echo ""

# =============================================================================
# CONFIGURATION
# =============================================================================
#
# All pipeline parameters are defined in: config/default.yaml
# Override individual values with CLI flags (see --help) or via a custom
# YAML file passed via --config.
#
# =============================================================================

# Your scratch directory on the HPC cluster
SCRATCH_DIR="${SCRATCH_DIR:-$(pwd)}"

# Pipeline directory (where this script is located)
PIPELINE_DIR="${PIPELINE_DIR:-${SCRATCH_DIR}/calcium_pipeline}"

# YAML configuration file
CONFIG_PATH="${CONFIG_PATH:-${PIPELINE_DIR}/config/default.yaml}"

# Conda environment path
# The pipeline needs: numpy, scipy, scikit-image, matplotlib, pyyaml, tifffile, opencv-python
# Override with: CONDA_ENV=/path/to/env qsub run_seeded_v3.sh
CONDA_ENV="${CONDA_ENV:-caiman}"

# =============================================================================
# PATHS (pipeline parameters come from $CONFIG_PATH)
# =============================================================================

# Input movie file
MOVIE="${MOVIE:-}"

# Imaging parameters
INDICATOR="${INDICATOR:-fluo4}"
FRAME_RATE="${FRAME_RATE:-2}"

# Detection parameters
MIN_RADIUS="${MIN_RADIUS:-10}"
MAX_RADIUS="${MAX_RADIUS:-25}"
AUTO_RADIUS="${AUTO_RADIUS:-true}"        # Auto-estimate radius from data
INTENSITY_THRESHOLD="${INTENSITY_THRESHOLD:-0.18}"
CORRELATION_THRESHOLD="${CORRELATION_THRESHOLD:-0.12}"
MAX_SEEDS="${MAX_SEEDS:-500}"
SMOOTH_SIGMA="${SMOOTH_SIGMA:-4.0}"
BORDER_MARGIN="${BORDER_MARGIN:-20}"

# Contour extraction
USE_TEMPORAL_PROJECTION="${USE_TEMPORAL_PROJECTION:-true}"
N_PEAK_FRAMES="${N_PEAK_FRAMES:-10}"
PEAK_PERCENTILE="${PEAK_PERCENTILE:-90}"
USE_MEAN_PROJ="${USE_MEAN_PROJ:-true}"   # blob detection on mean projection
USE_STD_PROJ="${USE_STD_PROJ:-true}"     # blob detection on std  projection

# Motion correction
MOTION_CORRECTION="${MOTION_CORRECTION:-true}"   # Set to 'false' to disable
MOTION_MODE="${MOTION_MODE:-rigid}"              # rigid, piecewise_rigid, or auto
MAX_SHIFT="${MAX_SHIFT:-20}"                     # Max shift in pixels

# Gallery
GALLERY="${GALLERY:-false}"                      # Set to 'true' to generate inspection gallery
MOVIE_GALLERY="${MOVIE_GALLERY:-false}"           # Set to 'true' to generate MP4 movie + HTML overlay viewer

# Temporal filtering
TEMPORAL_FILTER="${TEMPORAL_FILTER:-false}"       # Low-pass filter on traces
FILTER_CUTOFF="${FILTER_CUTOFF:-2.0}"            # Cutoff in Hz

# Deconvolution
DECONVOLUTION="${DECONVOLUTION:-true}"           # Spike deconvolution
DECONV_METHOD="${DECONV_METHOD:-oasis}"           # oasis or threshold

# Amplitude measurement
AMPLITUDE_METHOD="${AMPLITUDE_METHOD:-global_dff}"  # global_dff (default), direct, local_dff, local_background
EDGE_TRIM="${EDGE_TRIM:-false}"                     # Trim boundary artefacts from baseline estimation

# Output
if [ -n "$JOB_ID" ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_DIR}/results_seeded_v3_${JOB_ID}}"
else
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_DIR}/results_seeded_v3_$(date +%Y%m%d_%H%M%S)}"
fi

N_PROCESSES=${NSLOTS:-8}

# =============================================================================
# SETUP & ENVIRONMENT
# =============================================================================

echo ""
echo "Configuration:"
echo "  Config file: ${CONFIG_PATH}"
echo "  Movie: ${MOVIE}"
echo "  Indicator: ${INDICATOR}"
echo "  Frame rate: ${FRAME_RATE} Hz"
echo "  Output: ${OUTPUT_DIR}"
echo "  Processes: ${N_PROCESSES}"
echo ""
echo "Detection:"
echo "  Radius: ${MIN_RADIUS}-${MAX_RADIUS} px (auto: ${AUTO_RADIUS})"
echo "  Smooth sigma: ${SMOOTH_SIGMA}"
echo "  Temporal projection: ${USE_TEMPORAL_PROJECTION}"
echo "  Use mean projection: ${USE_MEAN_PROJ}"
echo "  Use std  projection: ${USE_STD_PROJ}"
echo ""

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/diagnostics"
mkdir -p "${SCRATCH_DIR}/logs"

# Environment setup — wrapped in set +e so failures don't kill the job
echo "Setting up environment..."
set +e
if command -v module &> /dev/null; then
    module load anaconda 2>/dev/null || module load anaconda/2024.02 2>/dev/null || true
    if [ -f ${CONDA_PREFIX:-/opt/conda}/etc/profile.d/conda.sh ]; then
        source ${CONDA_PREFIX:-/opt/conda}/etc/profile.d/conda.sh
    fi
    conda activate "${CONDA_ENV}" 2>/dev/null || conda activate caiman 2>/dev/null || true
else
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
    fi
    conda activate "${CONDA_ENV}" 2>/dev/null || conda activate caiman 2>/dev/null || true
fi
set -e

echo "  Python: $(which python)"
echo "  Python version: $(python --version 2>&1)"
echo ""

# Quick sanity check: can we import the core dependencies?
python -c "import numpy, scipy, skimage, matplotlib, yaml; print('  ✓ Core dependencies OK')" || {
    echo "ERROR: Missing core Python dependencies."
    echo "  Required: numpy, scipy, scikit-image, matplotlib, pyyaml"
    echo "  Install: pip install numpy scipy scikit-image matplotlib pyyaml tifffile opencv-python"
    exit 1
}

# Verify data
if [ ! -f "${MOVIE}" ] && [ ! -d "${MOVIE}" ]; then
    echo ""
    echo "ERROR: Movie not found: ${MOVIE}"
    echo ""
    echo "Set the MOVIE variable, e.g.:"
    echo "  MOVIE=/path/to/your/movie.nd2 qsub $(basename $0)"
    echo ""
    echo "Available .nd2 files in scratch:"
    find "${SCRATCH_DIR}" -maxdepth 3 -name '*.nd2' 2>/dev/null | head -10 || true
    exit 1
fi
echo "✓ Movie found: ${MOVIE} ($(du -h "${MOVIE}" 2>/dev/null | cut -f1 || echo '?'))"
echo ""

# =============================================================================
# RUN FULL PIPELINE
# =============================================================================

cd "${PIPELINE_DIR}"

echo "=========================================="
echo "Running Calcium Pipeline v2.1"
echo "=========================================="

# Create combined pipeline script
cat > "${OUTPUT_DIR}/run_full_pipeline.py" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Calcium Pipeline v2.1

Pipeline stages:
1. Load movie
2. Motion correction (CaImAn NoRMCorre)
3. Projection computation (single pass, shared downstream)
4. Auto-radius sweep (optional)
5. Contour-based seed detection
6. Trace extraction + baseline correction (ΔF/F₀)
7. Deconvolution (OASIS)
8. Diagnostics & confidence scoring
9. Save results & generate gallery
"""

import os
import sys
import argparse
import logging
import json
import gc
import tempfile
import shutil
import numpy as np
from datetime import datetime
from scipy.sparse import csc_matrix, save_npz

sys.path.insert(0, 'src')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def amplitude_diagnostic(data, label, frame_rate=None, is_movie=False):
    """
    Compute amplitude statistics for a trace matrix or movie at any pipeline stage.

    Returns a dict suitable for JSON serialisation tracking signal amplitude
    through each processing stage.  Call this at every stage to build a
    'amplitude_tracking' section in run_info.json.

    Parameters
    ----------
    data : ndarray
        (N, T) trace matrix or (T, d1, d2) movie.
    label : str
        Human-readable stage name, e.g. 'raw_traces', 'post_dff'.
    frame_rate : float, optional
        If provided, compute per-neuron transient-peak amplitudes.
    is_movie : bool
        If True, data is (T, d1, d2) movie — report per-frame stats.

    Returns
    -------
    dict with amplitude statistics for this stage.
    """
    info = {'stage': label}

    if is_movie:
        # Movie: per-frame statistics
        frame_means = np.mean(data, axis=(1, 2))
        info['type'] = 'movie'
        info['shape'] = list(data.shape)
        info['pixel_min'] = float(np.min(data))
        info['pixel_max'] = float(np.max(data))
        info['pixel_mean'] = float(np.mean(data))
        info['pixel_median'] = float(np.median(data))
        info['frame_mean_min'] = float(frame_means.min())
        info['frame_mean_max'] = float(frame_means.max())
        info['frame_mean_std'] = float(frame_means.std())
        return info

    # Trace matrix (N, T)
    N, T = data.shape
    info['type'] = 'traces'
    info['shape'] = [int(N), int(T)]

    # Global statistics
    info['global_min'] = float(np.min(data))
    info['global_max'] = float(np.max(data))
    info['global_mean'] = float(np.mean(data))
    info['global_median'] = float(np.median(data))

    # Per-trace range (max - min per neuron)
    trace_ranges = np.max(data, axis=1) - np.min(data, axis=1)
    info['trace_range_min'] = float(np.min(trace_ranges))
    info['trace_range_max'] = float(np.max(trace_ranges))
    info['trace_range_mean'] = float(np.mean(trace_ranges))
    info['trace_range_median'] = float(np.median(trace_ranges))

    # Per-trace peak amplitude (95th - 5th percentile, robust to outliers)
    peak_amps = np.percentile(data, 95, axis=1) - np.percentile(data, 5, axis=1)
    info['peak_amp_p5_p95_min'] = float(np.min(peak_amps))
    info['peak_amp_p5_p95_max'] = float(np.max(peak_amps))
    info['peak_amp_p5_p95_mean'] = float(np.mean(peak_amps))
    info['peak_amp_p5_p95_median'] = float(np.median(peak_amps))

    # Per-trace noise estimate (MAD of frame-to-frame differences)
    noise_levels = []
    for i in range(N):
        diff = np.diff(data[i])
        mad = np.median(np.abs(diff - np.median(diff)))
        noise_levels.append(1.4826 * mad / np.sqrt(2))
    noise_arr = np.array(noise_levels)
    info['noise_mad_mean'] = float(np.mean(noise_arr))
    info['noise_mad_median'] = float(np.median(noise_arr))

    # Per-trace SNR (robust: p95-p5 range / MAD noise)
    snr_arr = peak_amps / np.maximum(noise_arr, 1e-10)
    info['snr_mean'] = float(np.mean(snr_arr))
    info['snr_median'] = float(np.median(snr_arr))
    info['snr_min'] = float(np.min(snr_arr))
    info['snr_max'] = float(np.max(snr_arr))

    # Fraction of traces that are effectively flat (range < 1e-8)
    info['n_flat_traces'] = int(np.sum(trace_ranges < 1e-8))

    logger.info(f"  Amplitude tracking [{label}]: "
                f"range={info['trace_range_median']:.6f} (median), "
                f"peak_amp={info['peak_amp_p5_p95_median']:.6f} (median), "
                f"SNR={info['snr_median']:.1f} (median), "
                f"noise={info['noise_mad_median']:.6f}")

    return info


def load_movie(movie_path):
    """Load movie from various formats."""
    ext = os.path.splitext(movie_path)[1].lower()
    
    if os.path.isdir(movie_path):
        images_dir = os.path.join(movie_path, 'images')
        if os.path.isdir(images_dir):
            from skimage import io
            import glob
            tiff_files = sorted(glob.glob(os.path.join(images_dir, '*.tif')))
            logger.info(f"Loading {len(tiff_files)} TIFFs from {images_dir}")
            frames = [io.imread(f) for f in tiff_files]
            movie = np.stack(frames, axis=0)
        else:
            raise ValueError(f"No images/ folder in {movie_path}")
    elif ext == '.nd2':
        import nd2
        logger.info(f"Loading ND2: {movie_path}")
        movie = nd2.imread(movie_path)
        logger.info(f"  Raw ND2 shape: {movie.shape}, ndim: {movie.ndim}")
        
        # Handle multi-dimensional ND2 files
        # Common layouts: (T, Y, X), (T, C, Y, X), (T, Z, Y, X), (T, C, Z, Y, X)
        if movie.ndim == 4:
            # Assume (T, C, Y, X) — take first channel
            logger.info(f"  4D array — taking first channel: {movie.shape}")
            movie = movie[:, 0, :, :]
        elif movie.ndim == 5:
            # Assume (T, C, Z, Y, X) — take first channel, max-project Z
            logger.info(f"  5D array — first channel, max-Z: {movie.shape}")
            movie = movie[:, 0, :, :, :]
            movie = np.max(movie, axis=1)
        elif movie.ndim == 3:
            logger.info(f"  3D array (T, Y, X) — no reshaping needed")
        elif movie.ndim == 2:
            # Single frame? Treat as (1, Y, X)
            logger.info(f"  2D array — treating as single frame")
            movie = movie[np.newaxis, :, :]
        else:
            raise ValueError(f"Unexpected ND2 dimensions: {movie.shape}")
    elif ext in ['.tif', '.tiff']:
        from skimage import io
        logger.info(f"Loading TIFF: {movie_path}")
        movie = io.imread(movie_path)
    elif ext == '.npy':
        logger.info(f"Loading NPY: {movie_path}")
        movie = np.load(movie_path)
    else:
        raise ValueError(f"Unsupported: {ext}")
    
    logger.info(f"Movie shape: {movie.shape}")
    return movie.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Calcium Pipeline v2.1')
    
    # ── CLI: movie, output, config, and optional overrides ──────────────
    parser.add_argument('--movie', required=True, help='Input movie path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', default=None,
                        help='YAML configuration file (defaults to $PIPELINE_DIR/config/default.yaml)')

    # All remaining flags default to None so argparse values don't
    # silently override the YAML unless explicitly provided by the user.

    # Imaging overrides
    parser.add_argument('--indicator', default=None)
    parser.add_argument('--frame-rate', type=float, default=None)

    # Detection overrides
    parser.add_argument('--min-radius', type=float, default=None)
    parser.add_argument('--max-radius', type=float, default=None)
    parser.add_argument('--auto-radius', dest='auto_radius', action='store_const', const=True, default=None)
    parser.add_argument('--no-auto-radius', dest='auto_radius', action='store_const', const=False)
    parser.add_argument('--intensity-threshold', type=float, default=None)
    parser.add_argument('--correlation-threshold', type=float, default=None)
    parser.add_argument('--max-seeds', type=int, default=None)
    parser.add_argument('--smooth-sigma', type=float, default=None)
    parser.add_argument('--border-margin', type=int, default=None)
    parser.add_argument('--use-mean-proj', dest='use_mean_proj', action='store_const', const=True, default=None)
    parser.add_argument('--no-use-mean-proj', dest='use_mean_proj', action='store_const', const=False)
    parser.add_argument('--use-std-proj', dest='use_std_proj', action='store_const', const=True, default=None)
    parser.add_argument('--no-use-std-proj', dest='use_std_proj', action='store_const', const=False)

    # Motion correction overrides
    parser.add_argument('--motion-correction', dest='motion_enabled', action='store_const', const=True, default=None)
    parser.add_argument('--no-motion-correction', dest='motion_enabled', action='store_const', const=False)
    parser.add_argument('--motion-mode', default=None, choices=['rigid', 'piecewise_rigid', 'auto', None])
    parser.add_argument('--max-shift', type=int, default=None)

    # Deconvolution overrides
    parser.add_argument('--deconvolution', dest='deconv_enabled', action='store_const', const=True, default=None)
    parser.add_argument('--no-deconvolution', dest='deconv_enabled', action='store_const', const=False)
    parser.add_argument('--deconv-method', default=None)
    parser.add_argument('--temporal-filter', dest='temporal_filter', action='store_const', const=True, default=None)
    parser.add_argument('--no-temporal-filter', dest='temporal_filter', action='store_const', const=False)
    parser.add_argument('--filter-cutoff', type=float, default=None)

    # Baseline overrides
    parser.add_argument('--amplitude-method', default=None,
                        choices=['direct', 'global_dff', 'local_dff', 'local_background', None])
    parser.add_argument('--edge-trim', dest='edge_trim', action='store_const', const=True, default=None)
    parser.add_argument('--no-edge-trim', dest='edge_trim', action='store_const', const=False)

    # Output overrides
    parser.add_argument('--gallery', dest='gallery', action='store_const', const=True, default=None)
    parser.add_argument('--no-gallery', dest='gallery', action='store_const', const=False)

    # Dev overrides
    parser.add_argument('--dev-network-analysis', dest='dev_network', action='store_const', const=True, default=None)
    parser.add_argument('--no-dev-network-analysis', dest='dev_network', action='store_const', const=False)

    # Compute (not in YAML — pipeline-level, not per-dataset)
    parser.add_argument('--n-processes', type=int, default=8, help='Number of parallel processes')

    cli_args = parser.parse_args()

    # ── Resolve config path ────────────────────────────────────────────
    pipeline_dir = os.environ.get('PIPELINE_DIR', os.getcwd())
    config_path = cli_args.config or os.path.join(pipeline_dir, 'config', 'default.yaml')

    # ── Load YAML config with CLI overrides ────────────────────────────
    from config_loader import load_config
    overrides = {
        'imaging.indicator':              cli_args.indicator,
        'imaging.frame_rate':             cli_args.frame_rate,
        'detection.min_radius':           cli_args.min_radius,
        'detection.max_radius':           cli_args.max_radius,
        'detection.auto_radius.enabled':  cli_args.auto_radius,
        'detection.intensity_threshold':  cli_args.intensity_threshold,
        'detection.correlation_threshold': cli_args.correlation_threshold,
        'detection.max_seeds':            cli_args.max_seeds,
        'detection.smooth_sigma':         cli_args.smooth_sigma,
        'detection.border_margin':        cli_args.border_margin,
        'detection.use_mean_proj':        cli_args.use_mean_proj,
        'detection.use_std_proj':         cli_args.use_std_proj,
        'motion.enabled':                 cli_args.motion_enabled,
        'motion.mode':                    cli_args.motion_mode,
        'motion.max_shift':               cli_args.max_shift,
        'deconvolution.enabled':          cli_args.deconv_enabled,
        'deconvolution.method':           cli_args.deconv_method,
        'deconvolution.temporal_filter':  cli_args.temporal_filter,
        'deconvolution.filter_cutoff':    cli_args.filter_cutoff,
        'baseline.method':                cli_args.amplitude_method,
        'baseline.edge_trim':             cli_args.edge_trim,
        'output.gallery':                 cli_args.gallery,
        'dev.network_analysis':           cli_args.dev_network,
    }
    cfg = load_config(config_path, overrides=overrides)

    # ── Compatibility shim: build an `args` namespace from cfg ────────
    # The pipeline body below references args.X (e.g. args.min_radius).
    # We flatten cfg into a single namespace so existing code works.
    from argparse import Namespace
    args = Namespace(
        # CLI-only
        movie=cli_args.movie,
        output=cli_args.output,
        n_processes=cli_args.n_processes,
        # Imaging
        indicator=cfg.imaging.indicator,
        frame_rate=cfg.imaging.frame_rate,
        # Detection
        min_radius=cfg.detection.min_radius,
        max_radius=cfg.detection.max_radius,
        auto_radius=cfg.detection.auto_radius.enabled,
        intensity_threshold=cfg.detection.intensity_threshold,
        correlation_threshold=cfg.detection.correlation_threshold,
        max_seeds=cfg.detection.max_seeds,
        smooth_sigma=cfg.detection.smooth_sigma,
        border_margin=cfg.detection.border_margin,
        use_temporal_projection=cfg.detection.use_temporal_projection,
        use_mean_proj=cfg.detection.use_mean_proj,
        use_std_proj=cfg.detection.use_std_proj,
        n_peak_frames=cfg.detection.n_peak_frames,
        peak_percentile=cfg.detection.peak_percentile,
        # Motion
        motion_correction=cfg.motion.enabled,
        motion_mode=cfg.motion.mode,
        max_shift=cfg.motion.max_shift,
        motion_pw_strides=cfg.motion.pw_strides,
        motion_pw_overlaps=cfg.motion.pw_overlaps,
        motion_niter_rig=cfg.motion.niter_rig,
        motion_num_frames_split=cfg.motion.num_frames_split,
        # Deconvolution
        deconvolution=cfg.deconvolution.enabled,
        deconv_method=cfg.deconvolution.method,
        temporal_filter=cfg.deconvolution.temporal_filter,
        filter_cutoff=cfg.deconvolution.filter_cutoff,
        # Baseline
        amplitude_method=cfg.baseline.method,
        edge_trim=cfg.baseline.edge_trim,
        # Output
        gallery=cfg.output.gallery,
        # Dev
        dev_network_analysis=cfg.dev.network_analysis,
    )

    logger.info("="*70)
    logger.info("Configuration loaded")
    logger.info("="*70)
    logger.info(f"  Config file: {config_path}")
    logger.info(f"  Movie:       {args.movie}")
    logger.info(f"  Output:      {args.output}")
    logger.info(f"  Indicator:   {args.indicator} (decay_time={cfg.decay_time}s)")
    logger.info(f"  Frame rate:  {args.frame_rate} Hz")
    logger.info(f"  Baseline:    {args.amplitude_method}")
    logger.info(f"  Motion:      {args.motion_mode if args.motion_correction else 'disabled'}")
    logger.info(f"  Deconv:      {args.deconv_method if args.deconvolution else 'disabled'}")
    logger.info("="*70)
    
    os.makedirs(args.output, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix='calpipe_')
    start_time = datetime.now()
    results = {'config': vars(args)}
    
    try:
        # ==================================================================
        # STAGE 1: Load Movie
        # ==================================================================
        logger.info("="*70)
        logger.info("STAGE 1: Loading Movie")
        logger.info("="*70)
        
        movie = load_movie(args.movie)
        T, d1, d2 = movie.shape
        dims = (d1, d2)
        results['dims'] = dims
        results['n_frames'] = T
        
        # Keep reference to raw movie for gallery background images
        movie_raw = movie
        
        # ── Movie diagnostics (log properties that vary between datasets) ──
        logger.info(f"  Shape: {movie.shape}")
        logger.info(f"  dtype: {movie.dtype}")
        logger.info(f"  Intensity range: [{movie.min():.2f}, {movie.max():.2f}]")
        logger.info(f"  Mean intensity: {movie.mean():.2f}")
        logger.info(f"  Median intensity: {np.median(movie):.2f}")
        logger.info(f"  Std intensity: {movie.std():.2f}")
        
        # Check for problematic data
        if movie.max() == movie.min():
            logger.error("Movie has zero contrast — all pixels identical!")
        if movie.max() > 65000:
            logger.info("  Note: near uint16 saturation — possible clipping")
        if movie.min() < 0:
            logger.info("  Note: negative values present (denoised data?)")
        if movie.mean() < 1.0:
            logger.info("  Note: very low mean — data may already be ΔF/F")
        
        # Per-frame mean trace (quick look at signal drift)
        frame_means = np.mean(movie, axis=(1, 2))
        logger.info(f"  Frame mean range: [{frame_means.min():.2f}, {frame_means.max():.2f}]")
        logger.info(f"  First frame mean: {frame_means[0]:.2f}")
        logger.info(f"  Last frame mean:  {frame_means[-1]:.2f}")
        drift_pct = 100 * (frame_means[0] - frame_means[-1]) / (frame_means[0] + 1e-10)
        logger.info(f"  Signal drift: {drift_pct:.1f}%")
        
        results['movie_stats'] = {
            'min': float(movie.min()),
            'max': float(movie.max()),
            'mean': float(movie.mean()),
            'std': float(movie.std()),
            'drift_pct': float(drift_pct),
            'has_negatives': bool(movie.min() < 0),
        }
        
        # ── Amplitude tracking: accumulates diagnostics at each stage ────
        amp_tracking = []
        amp_tracking.append(amplitude_diagnostic(movie, '1_raw_movie', is_movie=True))
        
        # ==================================================================
        # STAGE 2: Motion Correction
        # ==================================================================
        logger.info("="*70)
        logger.info("STAGE 2: Motion Correction")
        logger.info("="*70)
        
        if args.motion_correction:
            from motion_correction import correct_motion, generate_motion_figure

            mc_result = correct_motion(
                movie,
                mode=args.motion_mode,
                max_shift=args.max_shift,
                pw_strides=tuple(args.motion_pw_strides),
                pw_overlaps=tuple(args.motion_pw_overlaps),
                niter_rig=args.motion_niter_rig,
                num_frames_split=args.motion_num_frames_split,
                n_processes=args.n_processes,
            )
            movie = mc_result.corrected
            results['motion_correction'] = mc_result.summary()
            # Record the resolved mode (not 'auto') for traceability
            results['motion_correction']['mode_resolved'] = mc_result.mode

            # Update dims in case motion correction cropped the borders
            T, d1, d2 = movie.shape
            dims = (d1, d2)
            logger.info(f"  Post-MC dims: {dims} ({T} frames)")

            # Update movie_raw to the cropped movie so gallery background
            # images match the coordinate system of seeds and projections
            movie_raw = movie

            # Save motion diagnostic figure
            generate_motion_figure(
                mc_result,
                os.path.join(args.output, 'motion_correction.png'),
            )
            # Save shifts for downstream reanalysis
            np.save(os.path.join(args.output, 'motion_shifts.npy'), mc_result.shifts)

            logger.info(f"Motion correction applied (mode={mc_result.mode}): "
                        f"max shift ({mc_result.max_shift_y:.1f}, "
                        f"{mc_result.max_shift_x:.1f}) px")
        else:
            logger.info("Motion correction disabled by config")
            results['motion_correction'] = {'mode': 'disabled'}
        
        amp_tracking.append(amplitude_diagnostic(movie, '2_post_motion_correction', is_movie=True))
        
        # ==================================================================
        # STAGE 3: Contour-Based Seed Detection
        # ==================================================================
        logger.info("="*70)
        logger.info("STAGE 3: Contour-Based Seed Detection")
        logger.info("="*70)
        
        # ── Compute projections ONCE and share across all downstream stages ──
        # Projections are invariant to radius choices, so the same set is used
        # by the auto-radius sweep, the final detection run, and the
        # visualization.  This replaces ~7 full projection computations with 1.
        from contour_seed_detection import compute_projections_extended

        logger.info("Computing projections (single pass, shared downstream)...")
        shared_projections = compute_projections_extended(
            movie,
            compute_correlation=True,
            smooth_sigma=args.smooth_sigma,
        )

        # ── Auto-radius estimation ──
        if args.auto_radius:
            from auto_radius import optimise_radius, generate_radius_figure

            logger.info("Optimising neuron radius for best trace quality...")
            radius_result = optimise_radius(
                movie,
                smooth_sigma=args.smooth_sigma,
                precomputed_projections=shared_projections,
            )

            if radius_result['reliable']:
                args.min_radius = radius_result['best_min_radius']
                args.max_radius = radius_result['best_max_radius']
                logger.info(f"Auto-radius: using min={args.min_radius:.1f}, "
                            f"max={args.max_radius:.1f} px "
                            f"({radius_result['best_n_good']} neurons with "
                            f"SNR≥{radius_result['snr_threshold']})")
            else:
                logger.warning(f"Auto-radius: insufficient good traces — "
                               f"keeping defaults: "
                               f"min={args.min_radius}, max={args.max_radius}")

            results['auto_radius'] = {
                k: v for k, v in radius_result.items()
                if k != 'all_results'
            }

            # Save diagnostic figure
            try:
                generate_radius_figure(
                    radius_result,
                    os.path.join(args.output, 'auto_radius.png'),
                )
            except Exception as e:
                logger.warning(f"Auto-radius figure failed: {e}")
        else:
            logger.info(f"Auto-radius disabled — using fixed: "
                        f"min={args.min_radius}, max={args.max_radius}")

        from contour_seed_detection import (
            detect_seeds_with_contours,
            contours_to_spatial_footprints,
            visualize_contour_detection,
        )

        seeds = detect_seeds_with_contours(
            movie,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            intensity_threshold=args.intensity_threshold,
            correlation_threshold=args.correlation_threshold,
            border_margin=args.border_margin,
            max_seeds=args.max_seeds,
            contour_method='otsu',
            smooth_sigma=args.smooth_sigma,
            use_mean=args.use_mean_proj,
            use_std=args.use_std_proj,
            use_temporal_projection=args.use_temporal_projection,
            n_peak_frames=args.n_peak_frames,
            peak_percentile=args.peak_percentile,
            diagnostics_dir=os.path.join(args.output, 'diagnostics'),
            precomputed_projections=shared_projections,
        )

        logger.info(f"Detected {seeds.n_seeds} seeds ({seeds.n_contours} with contours)")

        # Save hotspot suppression diagnostic if smoothing was applied
        if args.smooth_sigma > 0:
            try:
                from contour_seed_detection import suppress_hotspots, visualize_hotspot_suppression
                movie_smoothed = suppress_hotspots(movie, method='gaussian', sigma=args.smooth_sigma)
                visualize_hotspot_suppression(
                    movie, movie_smoothed,
                    os.path.join(args.output, 'diagnostics', 'hotspot_suppression.png'),
                    sigma=args.smooth_sigma, method='gaussian', dpi=150,
                )
                del movie_smoothed  # free memory
            except Exception as hs_err:
                logger.warning(f"Hotspot suppression diagnostic failed: {hs_err}")

        # Reuse shared projections for visualization (match what detection saw)
        projections = shared_projections

        # Unsmoothed projections for gallery background images (cheap — no correlation, no smoothing)
        projections_raw = compute_projections_extended(movie, compute_correlation=False, smooth_sigma=0.0)
        
        # Save clean projection arrays (unsmoothed max/std, smoothed correlation)
        np.save(os.path.join(args.output, 'max_projection_raw.npy'), projections_raw.max_proj)
        np.save(os.path.join(args.output, 'std_projection.npy'), projections_raw.std_proj)
        np.save(os.path.join(args.output, 'correlation_image.npy'), projections.correlation)
        
        # Original compact visualization
        visualize_contour_detection(
            projections, seeds,
            os.path.join(args.output, 'seed_detection_v3.png')
        )
        
        # Save individual projection figures to <output>/figures/
        from contour_seed_detection import save_projection_figures
        save_projection_figures(projections_raw, seeds, args.output, projections_corr=projections)
        
        # New detailed visualizations with zoom panels
        from contour_seed_detection import visualize_contour_detection_detailed
        visualize_contour_detection_detailed(
            projections, seeds,
            output_dir=os.path.join(args.output, 'detection_visualizations'),
            movie=movie,
            n_zoom_regions=6,
            zoom_size=200,
        )
        
        if seeds.n_seeds == 0:
            logger.error("No seeds detected!")
            return results
        
        # Convert to spatial footprints
        A_init = contours_to_spatial_footprints(seeds, dims, contour_fallback=True)
        
        # ==================================================================
        # STAGE 4: Trace Extraction & Baseline Correction
        # ==================================================================
        logger.info("="*70)
        logger.info("STAGE 4: Trace Extraction")
        logger.info("="*70)
        
        # Resolve decay time (needed by diagnostics)
        from config_loader import get_decay_time_for_indicator
        decay_time = get_decay_time_for_indicator(args.indicator)
        logger.info(f"Using decay_time={decay_time}s for {args.indicator}")
        logger.info(f"Amplitude method: {args.amplitude_method}")
        
        from trace_extraction import extract_traces
        
        try:
            A = A_init
            C_raw_fluorescence, _ = extract_traces(
                movie,     # raw motion-corrected movie
                A_init,
                chunk_size=500,
            )
            
            results['n_components'] = A.shape[1]
            results['trace_method'] = 'weighted_average'
            
        except Exception as e2:
            logger.error(f"Trace extraction failed: {e2}")
            import traceback
            traceback.print_exc()
            A = A_init
            C_raw_fluorescence = np.zeros((A_init.shape[1], T), dtype=np.float32)
            results['trace_method'] = 'failed'
            results['trace_error'] = str(e2)
        
        amp_tracking.append(amplitude_diagnostic(C_raw_fluorescence, '4a_raw_traces'))
        
        # ==================================================================
        # STAGE 4b: Baseline Correction
        # ==================================================================
        # Method selection (set via baseline.method in YAML):
        #   'direct'           — pass raw traces directly to OASIS.
        #                        OASIS estimates its own baseline.
        #   'global_dff'       — per-trace rolling percentile ΔF/F₀ (default)
        #   'local_dff'        — same as global_dff (reserved for future use)
        #   'local_background' — tissue-masked annulus background
        
        if args.amplitude_method == 'direct':
            # No baseline correction. Pass raw fluorescence directly to
            # OASIS — it estimates its own scalar baseline.
            logger.info("="*70)
            logger.info("STAGE 4b: Direct (no baseline correction)")
            logger.info("="*70)
            logger.info("  Raw traces passed directly to OASIS")
            
            C = C_raw_fluorescence.copy()
            C_raw = C_raw_fluorescence
            dff_info = {
                'method': 'direct',
                'note': 'Raw fluorescence passed to OASIS — no ΔF/F conversion',
            }
            
        elif args.amplitude_method == 'local_background':
            logger.info("="*70)
            logger.info("STAGE 4b: Local Tissue-Masked Background")
            logger.info("="*70)
            
            from preprocessing import compute_dff_local_background
            
            C, C_raw, dff_info = compute_dff_local_background(
                movie,
                A_init,
                frame_rate=args.frame_rate,
                percentile=8.0,
                window_fraction=0.25,
                min_window=50,
                max_window=500,
                annulus_inner_gap=2,
                annulus_outer_radius=20,
                edge_trim=args.edge_trim,
            )
            C_raw_fluorescence = C_raw
        else:
            # global_dff or local_dff: per-trace rolling percentile
            logger.info("="*70)
            logger.info("STAGE 4b: Per-Trace ΔF/F₀ Baseline Correction")
            logger.info("="*70)
            
            from preprocessing import compute_dff_traces
            
            C, C_raw, dff_info = compute_dff_traces(
                C_raw_fluorescence,
                frame_rate=args.frame_rate,
                percentile=8.0,
                window_fraction=0.25,
                min_window=50,
                max_window=500,
                edge_trim=args.edge_trim,
            )
        
        results['dff_correction'] = dff_info
        results['amplitude_method'] = args.amplitude_method
        amp_tracking.append(amplitude_diagnostic(C, '4b_corrected_traces'))
        
        # ── Baseline correction diagnostics ──────────────────────────────
        if args.amplitude_method != 'direct':
            try:
                from preprocessing import generate_dff_diagnostics
                generate_dff_diagnostics(
                    C_dff=C,
                    dff_info=dff_info,
                    output_dir=os.path.join(args.output, 'diagnostics'),
                    method=args.amplitude_method,
                    frame_rate=args.frame_rate,
                    movie=movie if args.amplitude_method == 'local_background' else None,
                    A=A_init if args.amplitude_method == 'local_background' else None,
                )
            except Exception as diag_err:
                logger.warning(f"DFF diagnostics failed: {diag_err}")
                import traceback
                traceback.print_exc()
        
        # ==================================================================
        # STAGE 5b: Temporal Filtering (optional)
        # ==================================================================
        C_raw_unfilt = C.copy()
        
        if args.temporal_filter:
            logger.info("="*70)
            logger.info("STAGE 5b: Temporal Filtering")
            logger.info("="*70)
            
            from deconvolution import temporal_filter
            C = temporal_filter(C, args.frame_rate, cutoff_hz=args.filter_cutoff)
            results['temporal_filter'] = {
                'applied': True,
                'cutoff_hz': args.filter_cutoff,
            }
        else:
            results['temporal_filter'] = {'applied': False}
        
        amp_tracking.append(amplitude_diagnostic(C, '5_pre_deconvolution'))
        
        # ==================================================================
        # STAGE 5b: Population Drift Removal
        # ==================================================================
        # Recording-wide artefacts (focus shifts, illumination changes,
        # medium disturbances) cause all ROIs to drift simultaneously.
        # These are SLOW (>5s) compared to calcium transients (~0.5-2s).
        #
        # Fix: compute the population median trace, smooth it with a wide
        # rolling window (~10s) to remove any fast synchronous transients,
        # then subtract the smoothed drift from each trace.
        # This preserves synchronous calcium events while removing slow
        # recording artefacts.
        logger.info("="*70)
        logger.info("STAGE 5b: Population Drift Removal")
        logger.info("="*70)
        
        pop_median = np.median(C, axis=0)
        
        # Smooth with ~10s rolling median to preserve fast synchronous events
        drift_window = max(5, int(args.frame_rate * 10))  # ~10 seconds
        if drift_window % 2 == 0:
            drift_window += 1  # must be odd for median filter
        
        from scipy.ndimage import median_filter
        pop_drift = median_filter(pop_median, size=drift_window, mode='reflect')
        
        drift_range = np.max(pop_drift) - np.min(pop_drift)
        logger.info(f"  Population drift (smoothed, {drift_window}-frame window): "
                    f"range={drift_range:.4f}")
        
        if drift_range > 0.005:  # only subtract if there's meaningful slow drift
            C = C - pop_drift[np.newaxis, :]
            logger.info(f"  Subtracted slow population drift (range={drift_range:.4f})")
        else:
            logger.info(f"  Population drift negligible, skipping subtraction")
        
        # ==================================================================
        # STAGE 5c: Deconvolution
        # ==================================================================
        deconv_result = None
        
        if args.deconvolution:
            logger.info("="*70)
            logger.info("STAGE 5c: Spike Deconvolution")
            logger.info("="*70)
            
            # ── OASIS / threshold deconvolution ──
            from deconvolution import deconvolve_traces, generate_deconvolution_figure
            
            deconv_result = deconvolve_traces(
                C,
                frame_rate=args.frame_rate,
                decay_time=decay_time,
                method=args.deconv_method,
            )
            
            # Save standard deconvolution outputs
            np.save(os.path.join(args.output, 'spike_trains.npy'),
                    deconv_result['S'])
            np.save(os.path.join(args.output, 'traces_denoised.npy'),
                    deconv_result['C_denoised'])
            np.save(os.path.join(args.output, 'deconv_noise.npy'),
                    deconv_result['noise'])
            
            results['deconvolution'] = {
                'method': deconv_result['method'],
                'total_spikes': int(deconv_result['n_spikes'].sum()),
                'median_spikes_per_neuron': float(np.median(deconv_result['n_spikes'])),
                'mean_noise': float(np.mean(deconv_result['noise'])),
            }
            
            # ── Amplitude tracking through deconvolution stages ──────────
            if 'C_dff' in deconv_result:
                amp_tracking.append(amplitude_diagnostic(
                    deconv_result['C_dff'], '6a_dff_input_to_oasis'))
            
            amp_tracking.append(amplitude_diagnostic(
                deconv_result['C_denoised'], '6b_oasis_denoised'))
            
            # Spike amplitude statistics (from the S matrix directly)
            S_all = deconv_result['S']
            spike_vals = S_all[S_all > 0]
            spike_amp_info = {
                'stage': '6c_spike_amplitudes',
                'type': 'spike_values',
                'n_neurons_with_spikes': int(np.sum(deconv_result['n_spikes'] > 0)),
                'n_neurons_zero_spikes': int(np.sum(deconv_result['n_spikes'] == 0)),
                'total_spike_events': int(len(spike_vals)),
            }
            if len(spike_vals) > 0:
                spike_amp_info.update({
                    'spike_val_min': float(np.min(spike_vals)),
                    'spike_val_max': float(np.max(spike_vals)),
                    'spike_val_mean': float(np.mean(spike_vals)),
                    'spike_val_median': float(np.median(spike_vals)),
                    'spike_val_p25': float(np.percentile(spike_vals, 25)),
                    'spike_val_p75': float(np.percentile(spike_vals, 75)),
                })
            amp_tracking.append(spike_amp_info)
            
            # Transient amplitudes (peak - baseline on denoised traces)
            C_den = deconv_result['C_denoised']
            N_den = C_den.shape[0]
            transient_amps = []
            for i_neuron in range(N_den):
                sp_frames = np.where(S_all[i_neuron] > 0)[0]
                for t_sp in sp_frames:
                    # Baseline: median of 1s window before spike
                    bl_start = max(0, t_sp - int(args.frame_rate))
                    bl_end = max(0, t_sp - 1)
                    if bl_end > bl_start:
                        bl = np.median(C_den[i_neuron, bl_start:bl_end])
                    else:
                        bl = 0.0
                    # Peak: max in 0.5s window after spike
                    pk_end = min(C_den.shape[1], t_sp + max(1, int(0.5 * args.frame_rate)))
                    pk = np.max(C_den[i_neuron, t_sp:pk_end])
                    amp = pk - bl
                    if amp > 0:
                        transient_amps.append(amp)
            
            ta = np.array(transient_amps) if transient_amps else np.array([0.0])
            amp_tracking.append({
                'stage': '6d_transient_amplitudes_peak_minus_baseline',
                'type': 'transient_amplitudes',
                'n_transients': len(transient_amps),
                'amp_min': float(np.min(ta)),
                'amp_max': float(np.max(ta)),
                'amp_mean': float(np.mean(ta)),
                'amp_median': float(np.median(ta)),
                'amp_p25': float(np.percentile(ta, 25)),
                'amp_p75': float(np.percentile(ta, 75)),
                'note': 'peak ΔF/F₀ minus pre-spike baseline on denoised trace',
            })
            
            logger.info(f"  Amplitude tracking summary:")
            logger.info(f"    Raw traces median range: "
                        f"{amp_tracking[3]['trace_range_median']:.4f}" 
                        if len(amp_tracking) > 3 and 'trace_range_median' in amp_tracking[3]
                        else "    (raw traces not available)")
            logger.info(f"    ΔF/F₀ median range: "
                        f"{amp_tracking[-4]['peak_amp_p5_p95_median']:.6f}"
                        if len(amp_tracking) > 4 and 'peak_amp_p5_p95_median' in amp_tracking[-4]
                        else "    (ΔF/F₀ not available)")
            logger.info(f"    Denoised median range: "
                        f"{amp_tracking[-3]['peak_amp_p5_p95_median']:.6f}"
                        if len(amp_tracking) > 3 and 'peak_amp_p5_p95_median' in amp_tracking[-3]
                        else "    (denoised not available)")
            logger.info(f"    Transient amp median: {float(np.median(ta)):.6f} ΔF/F₀")
            
            
            # Generate deconvolution diagnostic figure
            try:
                from deconvolution import generate_deconvolution_figure
                generate_deconvolution_figure(
                    C_raw_unfilt, deconv_result, args.frame_rate,
                    os.path.join(args.output, 'deconvolution.png'),
                    C_filtered=C if args.temporal_filter else None,
                )
            except Exception as fig_err:
                logger.warning(f"Deconv figure failed: {fig_err}")
            
            # Save individual ROI trace figures for presentation
            try:
                from deconvolution import save_roi_trace_figures
                save_roi_trace_figures(
                    C_raw_unfilt, deconv_result, args.frame_rate,
                    args.output,
                )
            except Exception as roi_fig_err:
                logger.warning(f"ROI trace figures failed: {roi_fig_err}")
            
            # Generate decay parameter diagnostics
            try:
                from deconvolution import generate_decay_diagnostics
                generate_decay_diagnostics(
                    deconv_result, C_raw_unfilt, args.frame_rate,
                    decay_time, args.output,
                )
            except Exception as decay_err:
                logger.warning(f"Decay diagnostics failed: {decay_err}")
            
            logger.info(f"  Spikes: {deconv_result['n_spikes'].sum()} total "
                        f"median {np.median(deconv_result['n_spikes']):.0f}/neuron")
        else:
            results['deconvolution'] = {'method': 'disabled'}
        
        # ==================================================================
        # STAGE 6: Diagnostics & Confidence Scoring
        # ==================================================================
        logger.info("="*70)
        logger.info("STAGE 6: Diagnostics & Confidence Scoring")
        logger.info("="*70)
        
        # No filtering — all components are kept.
        # Confidence scores combine detection and temporal signals.
        A_final = A
        C_final = C
        n_final = A.shape[1]
        
        try:
            from diagnostics import run_diagnostics, generate_diagnostic_figures
            
            # Gather detection-stage signals
            det_conf = seeds.confidence if hasattr(seeds, 'confidence') else None
            contour_ok = seeds.contour_success if hasattr(seeds, 'contour_success') else None
            
            # Detection signals may have different length than output
            if det_conf is not None and len(det_conf) != n_final:
                logger.info(f"  Detection signals ({len(det_conf)}) != output ({n_final}), skipping detection confidence")
                det_conf = None
                contour_ok = None
            
            diag = run_diagnostics(
                C_final, args.frame_rate, decay_time,
                detection_confidence=det_conf,
                contour_success=contour_ok,
            )
            
            confidence = diag.confidence
            results['confidence_source'] = diag.confidence_source
            results['confidence_median'] = float(np.median(confidence))
            
            # Generate diagnostic figures
            diag_dir = os.path.join(args.output, 'diagnostics')
            os.makedirs(diag_dir, exist_ok=True)
            try:
                fig_paths = generate_diagnostic_figures(
                    diag, diag_dir,
                    A=A_final, dims=dims, C=C_final,
                    frame_rate=args.frame_rate,
                    decay_time=decay_time,
                    max_projection=projections.max_proj if hasattr(projections, 'max_proj') else None,
                )
                logger.info(f"  Generated {len(fig_paths)} diagnostic figures")
            except Exception as fig_err:
                logger.warning(f"  Diagnostic figures failed: {fig_err}")
            
            # Save diagnostics NPZ
            np.savez(
                os.path.join(args.output, 'diagnostics.npz'),
                **diag.to_npz_dict(),
            )
            
        except Exception as e:
            logger.warning(f"Diagnostics failed: {e}, using uniform confidence")
            confidence = np.ones(n_final) * 0.5
        
        results['n_final'] = n_final
        
        # ==================================================================
        # STAGE 7: Save Results
        # ==================================================================
        logger.info("="*70)
        logger.info("STAGE 7: Saving Results")
        logger.info("="*70)
        
        save_npz(os.path.join(args.output, 'spatial_footprints.npz'), csc_matrix(A_final))
        np.save(os.path.join(args.output, 'temporal_traces.npy'), C_final)
        np.save(os.path.join(args.output, 'confidence_scores.npy'), confidence)
        
        # Save projection images for downstream analysis (flagged neuron inspection etc.)
        if hasattr(projections, 'max_proj') and projections.max_proj is not None:
            np.save(os.path.join(args.output, 'max_projection.npy'), projections.max_proj)
        if hasattr(projections, 'mean_proj') and projections.mean_proj is not None:
            np.save(os.path.join(args.output, 'mean_projection.npy'), projections.mean_proj)
        
        # Save raw traces for downstream reanalysis
        if C_raw is not None:
            np.save(os.path.join(args.output, 'temporal_traces_raw.npy'), C_raw)
        
        # Save run info
        elapsed = (datetime.now() - start_time).total_seconds()
        results['elapsed_seconds'] = elapsed
        results['amplitude_tracking'] = amp_tracking
        
        with open(os.path.join(args.output, 'run_info.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"Final neurons: {n_final}")
        
        # ==================================================================
        # STAGE 7: Visual Outputs
        # ==================================================================
        logger.info("="*70)
        logger.info("STAGE 7: Visual Outputs")
        logger.info("="*70)
        
        # Free large arrays before visualisation to reduce memory pressure.
        gc.collect()
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from scipy.signal import find_peaks
            import base64
            
            # ------------------------------------------------------------------
            # Helper functions
            # ------------------------------------------------------------------
            def get_roi_weights(A, roi_idx, dims):
                """Extract spatial weight map for an ROI."""
                if hasattr(A, 'toarray'):
                    w = A[:, roi_idx].toarray().flatten()
                else:
                    w = A[:, roi_idx].flatten()
                return w.reshape(dims)

            def get_roi_centroid(weights):
                """Return (cy, cx) weighted centroid of footprint."""
                ys, xs = np.where(weights > 0)
                if len(ys) == 0:
                    return None
                cy = float(np.average(ys, weights=weights[ys, xs]))
                cx = float(np.average(xs, weights=weights[ys, xs]))
                return cy, cx

            def pad_crop(frame, cy, cx, half_size):
                H, W = frame.shape
                size = 2 * half_size
                out = np.full((size, size), frame.min(), dtype=frame.dtype)
                src_y0 = int(round(cy)) - half_size
                src_x0 = int(round(cx)) - half_size
                dst_y0 = max(0, -src_y0)
                dst_x0 = max(0, -src_x0)
                sy0 = max(0, src_y0);  sy1 = min(H, src_y0 + size)
                sx0 = max(0, src_x0);  sx1 = min(W, src_x0 + size)
                h = sy1 - sy0;  w = sx1 - sx0
                if h > 0 and w > 0:
                    out[dst_y0:dst_y0+h, dst_x0:dst_x0+w] = frame[sy0:sy1, sx0:sx1]
                return out

            def pad_crop_mask(mask2d, cy, cx, half_size):
                tmp = pad_crop(mask2d.astype(np.float32), cy, cx, half_size)
                return tmp

            def find_peak_frame(trace):
                F0 = np.percentile(trace, 20)
                scale = np.percentile(trace, 95) - F0 + 1e-10
                norm = (trace - F0) / scale
                peaks, _ = find_peaks(norm, height=0.3, distance=int(args.frame_rate), prominence=0.2)
                if len(peaks) == 0:
                    return int(np.argmax(trace))
                return int(peaks[np.argmax(trace[peaks])])

            def find_baseline_frame(trace):
                F0 = np.percentile(trace, 20)
                thresh = F0 + 0.1 * (trace.max() - F0)
                quiet = np.where(trace <= thresh)[0]
                if len(quiet) == 0:
                    return int(np.argmin(trace))
                mid = len(trace) // 2
                return int(quiet[np.argmin(np.abs(quiet - mid))])

            def compute_dff(trace):
                F0 = np.percentile(trace, 20)
                if F0 > 1e-6:
                    return (trace - F0) / F0 * 100, 'ΔF/F (%)'
                return trace - trace.min(), 'F - F_min'

            max_proj = projections.max_proj
            mp_lo = np.percentile(max_proj, 2)
            mp_hi = np.percentile(max_proj, 99)
            max_proj_display = np.clip((max_proj - mp_lo) / (mp_hi - mp_lo + 1e-10), 0, 1)

            sort_idx = np.argsort(confidence)[::-1]
            n_rois_to_plot = n_final
            roi_image_paths = []

            # ==============================================================
            # ROI Inspection Images (off by default, slow)
            # ==============================================================
            if args.gallery:
                logger.info(f"  Generating {n_rois_to_plot} ROI inspection images...")
                inspection_dir = os.path.join(args.output, 'inspection')
                os.makedirs(inspection_dir, exist_ok=True)

                for plot_i, roi_idx in enumerate(sort_idx):
                    try:
                        weights = get_roi_weights(A_final, roi_idx, dims)
                        centroid = get_roi_centroid(weights)
                        if centroid is None:
                            continue

                        cy, cx = centroid
                        w_thresh = max(weights.max() * 0.2, 1e-10)
                        TIGHT = 60

                        trace = C_final[roi_idx, :]
                        dff, ylabel = compute_dff(trace)
                        peak_t  = find_peak_frame(trace)
                        base_t  = find_baseline_frame(trace)

                        T_trace = len(trace)
                        T_movie = len(movie)
                        peak_f = min(int(peak_t * T_movie / T_trace), T_movie - 1)

                        conf_score = confidence[roi_idx]
                        if conf_score >= 0.7:
                            tier, tier_color = "HIGH", '#00c853'
                        elif conf_score >= 0.4:
                            tier, tier_color = "MED", '#ff9900'
                        else:
                            tier, tier_color = "LOW", '#ff5252'

                        title = (f"ROI #{roi_idx}  "
                                 f"(Rank {plot_i+1}/{n_rois_to_plot})  "
                                 f"Confidence: {conf_score:.2f} [{tier}]")

                        import matplotlib.gridspec as mpl_gridspec
                        fig = plt.figure(figsize=(20, 6), facecolor='#1a1a2e')
                        gs_roi = mpl_gridspec.GridSpec(
                            2, 11, height_ratios=[1, 1.5], hspace=0.4, wspace=0.08)
                        fig.suptitle(title, fontsize=12, fontweight='bold',
                                     color=tier_color)

                        offsets = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
                        w_tight = pad_crop_mask(weights, cy, cx, TIGHT)

                        sequence_crops = []
                        for offset in offsets:
                            f_idx = max(0, min(T_movie - 1, peak_f + offset))
                            sequence_crops.append(pad_crop(movie[f_idx], cy, cx, TIGHT))

                        vlo = np.percentile(sequence_crops, 1)
                        vhi = np.percentile(sequence_crops, 99.5)
                        if vhi <= vlo:
                            vhi = vlo + 1

                        for i, (offset, crop) in enumerate(zip(offsets, sequence_crops)):
                            ax = fig.add_subplot(gs_roi[0, i])
                            ax.set_facecolor('#1a1a2e')
                            ax.imshow(crop, cmap='gray', vmin=vlo, vmax=vhi,
                                      interpolation='none')
                            ax.contour(w_tight, levels=[w_thresh],
                                       colors=['cyan'], linewidths=1.2, alpha=0.8)
                            if offset == 0:
                                ax.set_title(
                                    f"Peak (t={peak_t/args.frame_rate:.1f}s)",
                                    fontsize=10, color='lime', fontweight='bold')
                                for spine in ax.spines.values():
                                    spine.set_edgecolor('lime')
                                    spine.set_linewidth(2)
                            else:
                                label = f"+{offset}" if offset > 0 else str(offset)
                                ax.set_title(f"Peak {label}", fontsize=9, color='#ccc')
                                for spine in ax.spines.values():
                                    spine.set_edgecolor('#444')
                            ax.axis('off')

                        ax_trace = fig.add_subplot(gs_roi[1, :])
                        ax_trace.set_facecolor('#111')
                        t_ax = np.arange(len(dff)) / args.frame_rate
                        ax_trace.plot(t_ax, dff, color='#4fc3f7', linewidth=1.0)
                        ax_trace.axhline(0, color='#555', linewidth=0.5)
                        ax_trace.axvline(base_t / args.frame_rate, color='cyan',
                                         linestyle='--', linewidth=1.2, label='Baseline')
                        ax_trace.axvline(peak_t / args.frame_rate, color='lime',
                                         linestyle='--', linewidth=1.2, label='Peak')
                        win_lo = max(0, (peak_f - 5)) / args.frame_rate
                        win_hi = min(T_movie - 1, (peak_f + 5)) / args.frame_rate
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
                        plt.savefig(img_path, dpi=100, bbox_inches='tight',
                                    facecolor='white')
                        plt.close()
                        roi_image_paths.append(img_path)

                    except Exception as roi_err:
                        logger.warning(f"    ROI #{roi_idx} image failed: {roi_err}")
                        plt.close('all')
                        continue

                    if (plot_i + 1) % 50 == 0:
                        logger.info(f"    Generated {plot_i + 1}/{n_rois_to_plot} ROI images")

                logger.info(f"  Generated {len(roi_image_paths)} ROI inspection images")
            else:
                logger.info("  ROI inspection images disabled (use GALLERY=true to enable)")
            
            # ==============================================================
            # Interactive HTML Gallery (always on)
            # ==============================================================
            logger.info("  Creating interactive HTML gallery...")
            
            try:
                from interactive_gallery import generate_interactive_gallery
                gallery_path = generate_interactive_gallery(
                    seeds=seeds,
                    projections=projections_raw,
                    movie=movie_raw,            # Raw movie for background images
                    movie_processed=movie_raw,
                    output_path=os.path.join(args.output, 'gallery.html'),
                    title=f"ROI Detection - {os.path.basename(args.movie)}",
                    max_rois=500,
                    traces_denoised=deconv_result['C_denoised'] if deconv_result else None,
                    spike_trains=deconv_result['S'] if deconv_result else None,
                    pipeline_traces_dff=C,                     # Pipeline's corrected ΔF/F₀
                    pipeline_traces_raw=C_raw_fluorescence,    # Raw fluorescence traces
                )
                logger.info(f"  Interactive gallery saved to: {gallery_path}")
            except Exception as gallery_err:
                logger.warning(f"Interactive gallery failed: {gallery_err}")
                # Fallback to simple gallery
                logger.info("  Falling back to simple gallery...")
                
                def embed_image(img_path):
                    with open(img_path, 'rb') as f:
                        data = base64.b64encode(f.read()).decode('utf-8')
                    return f"data:image/png;base64,{data}"
                
                html = f'''<!DOCTYPE html>
<html>
<head>
    <title>ROI Gallery (Fallback)</title>
    <style>
        body {{ font-family: Arial; margin: 20px; background: #1a1a2e; color: #eee; }}
        .gallery {{ display: flex; flex-wrap: wrap; gap: 15px; }}
        .roi-card {{ background: #16213e; padding: 10px; border-radius: 8px; }}
        .roi-card img {{ max-width: 600px; }}
    </style>
</head>
<body>
    <h1>ROI Gallery (Fallback Version)</h1>
    <p>Interactive gallery failed - showing simple version. {n_final} ROIs detected.</p>
    <div class="gallery">
'''
                for i, img_path in enumerate(roi_image_paths[:50]):
                    embedded = embed_image(img_path)
                    html += f'<div class="roi-card"><img src="{embedded}"><p>ROI #{sort_idx[i]}</p></div>\n'
                
                html += '</div></body></html>'
                
                gallery_path = os.path.join(args.output, 'gallery.html')
                with open(gallery_path, 'w') as f:
                    f.write(html)
                logger.info(f"  Fallback gallery saved to: {gallery_path}")
            
            # ==============================================================
            # Movie Gallery — full movie with contour overlay (off by default)
            # Enable with: MOVIE_GALLERY=true
            # ==============================================================
            if os.environ.get('MOVIE_GALLERY', 'false').lower() == 'true':
                logger.info("  Creating frame gallery (self-contained HTML)...")
                try:
                    from movie_gallery import generate_movie_gallery
                    mg_result = generate_movie_gallery(
                        movie=movie_raw,
                        seeds=seeds,
                        output_dir=args.output,
                        frame_rate=args.frame_rate,
                        subsample=int(os.environ.get('MOVIE_GALLERY_SUBSAMPLE', '1')),
                        traces_denoised=deconv_result['C_denoised'] if deconv_result else None,
                        spike_trains=deconv_result['S'] if deconv_result else None,
                        deconv_noise=deconv_result['noise'] if deconv_result else None,
                        movie_processed=movie_raw,
                        title=f"Frame Gallery - {os.path.basename(args.movie)}",
                    )
                    logger.info(f"  Frame gallery: {mg_result['file_size_mb']:.1f} MB HTML, "
                                f"{mg_result['n_rois']} ROIs, "
                                f"{mg_result['n_frames_display']} frames")
                except Exception as mg_err:
                    logger.warning(f"Movie gallery failed: {mg_err}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.info("  Movie gallery disabled (use MOVIE_GALLERY=true to enable)")
            
        except Exception as e:
            logger.warning(f"Visual outputs failed: {e}")
            import traceback
            traceback.print_exc()
        
        # ==================================================================
        # STAGE 8: Development / Experimental Modules
        # ==================================================================
        
        if args.dev_network_analysis:
            try:
                logger.info("")
                logger.info("="*70)
                logger.info("STAGE 8: Development — Network Analysis (detection-free)")
                logger.info("="*70)
                
                # Use motion-corrected movie if available, otherwise raw
                analysis_movie = movie
                
                sys.path.insert(0, os.path.join(os.environ.get('PIPELINE_DIR', '.'), 'src', 'dev'))
                
                # 8a: Variance-weighted global trace + spectral analysis
                from network_spectral import run_network_spectral
                spectral_result = run_network_spectral(
                    analysis_movie,
                    frame_rate=args.frame_rate,
                    output_dir=args.output,
                    dataset_name=os.path.basename(args.movie),
                )
                results['dev_spectral'] = spectral_result['spectral_summary']
                
                # 8b: PCA spatial decomposition
                from network_pca import run_network_pca
                
                # Auto-select downsampling based on image size
                _, Y_m, X_m = analysis_movie.shape
                n_pixels = Y_m * X_m
                ds_factor = 1
                if n_pixels > 500000:
                    ds_factor = 2
                if n_pixels > 2000000:
                    ds_factor = 4
                
                pca_result = run_network_pca(
                    analysis_movie,
                    frame_rate=args.frame_rate,
                    output_dir=args.output,
                    dataset_name=os.path.basename(args.movie),
                    n_components=10,
                    downsample_spatial=ds_factor,
                )
                results['dev_pca'] = pca_result['summary']
                
                logger.info("")
                logger.info("  Development network analysis complete.")
                
            except Exception as e:
                logger.warning(f"Development network analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Seeds detected: {seeds.n_seeds}")
        print(f"Contour success: {seeds.contour_success_rate:.1%}")
        print(f"Final neurons: {n_final}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Output: {args.output}")
        print(f"\nOpen gallery.html in browser for visual inspection")
        print("="*60)
        
        return results
        
    finally:
        # Cleanup temp directories
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        gc.collect()


if __name__ == '__main__':
    main()
PYTHON_SCRIPT

# Run the full pipeline
# CONFIG_PATH is exported by run.sh (cmd_single/cmd_batch) or inherited from
# the parent shell if running directly.
CONFIG_PATH="${CONFIG_PATH:-${PIPELINE_DIR}/config/default.yaml}"
export PIPELINE_DIR CONFIG_PATH

python "${OUTPUT_DIR}/run_full_pipeline.py" \
    --movie "${MOVIE}" \
    --output "${OUTPUT_DIR}" \
    --config "${CONFIG_PATH}"

EXIT_CODE=$?

# =============================================================================
# RESULTS
# =============================================================================

echo ""
echo "=========================================="
echo "Pipeline finished"
echo "=========================================="
echo "Exit code: ${EXIT_CODE}"
echo "Completed: $(date)"
echo ""
echo "Results: ${OUTPUT_DIR}"
echo ""
ls -lh "${OUTPUT_DIR}"
echo ""

if [ -f "${OUTPUT_DIR}/spatial_footprints.npz" ]; then
    echo "✓ Final results saved"
    python -c "from scipy.sparse import load_npz; A=load_npz('${OUTPUT_DIR}/spatial_footprints.npz'); print(f'  Final neurons: {A.shape[1]}')"
fi

echo ""
echo "=========================================="

exit ${EXIT_CODE}
