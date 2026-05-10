#!/usr/bin/env python3
"""
Calcium Pipeline — Full Pipeline Orchestrator

Pipeline stages
---------------
1. Load movie
2. Motion correction (CaImAn NoRMCorre)
3. Seed detection + contour extraction (with optional auto-radius)
4. Trace extraction + baseline correction (ΔF/F₀)
5. Population drift removal + optional temporal filtering
6. Spike deconvolution (OASIS)
7. Save results
8. Visual outputs (interactive gallery, optional per-ROI PNGs, optional movie gallery)
9. Optional dev / experimental modules

Invoked as a module from run.sh's generated SGE scripts, or directly
on an interactive compute node:

    python -m src.run_full_pipeline \\
        --movie /path/to/movie.nd2 \\
        --output /path/to/output_dir \\
        --config /path/to/config.yaml

All pipeline parameters come from the YAML config; CLI flags override
individual values when explicitly passed.
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

# Ensure src/ is importable when invoked as a script outside the package
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT LAYOUT
# =============================================================================
# Per-recording outputs are organised as
#   <output>/
#     ├── run_info.json
#     ├── gallery.html  (etc.)
#     ├── data/         numerical results (.npy, .npz)
#     ├── figures/      diagnostic / inspection images (.png)
#     └── diagnostics/  pipeline JSON dumps + a few targeted figures

DATA_SUBDIR        = 'data'
FIGURES_SUBDIR     = 'figures'
DIAGNOSTICS_SUBDIR = 'diagnostics'


def _ensure_output_layout(output: str) -> None:
    """Create data/, figures/, and diagnostics/ inside the output dir."""
    for sub in (DATA_SUBDIR, FIGURES_SUBDIR, DIAGNOSTICS_SUBDIR):
        os.makedirs(os.path.join(output, sub), exist_ok=True)


def data_path(output: str, *parts: str) -> str:
    """Path inside <output>/data/."""
    return os.path.join(output, DATA_SUBDIR, *parts)


def figure_path(output: str, *parts: str) -> str:
    """Path inside <output>/figures/."""
    return os.path.join(output, FIGURES_SUBDIR, *parts)


def diagnostics_path(output: str, *parts: str) -> str:
    """Path inside <output>/diagnostics/."""
    return os.path.join(output, DIAGNOSTICS_SUBDIR, *parts)


# =============================================================================
# MOVIE LOADING
# =============================================================================

def load_movie(movie_path):
    """Load movie from various formats (.nd2, .tif/.tiff, .npy, image folder)."""
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

        # Common ND2 layouts: (T, Y, X), (T, C, Y, X), (T, Z, Y, X), (T, C, Z, Y, X)
        if movie.ndim == 4:
            logger.info(f"  4D array — taking first channel: {movie.shape}")
            movie = movie[:, 0, :, :]
        elif movie.ndim == 5:
            logger.info(f"  5D array — first channel, max-Z: {movie.shape}")
            movie = movie[:, 0, :, :, :]
            movie = np.max(movie, axis=1)
        elif movie.ndim == 3:
            logger.info(f"  3D array (T, Y, X) — no reshaping needed")
        elif movie.ndim == 2:
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


# =============================================================================
# CLI / CONFIG
# =============================================================================

def _build_parser():
    parser = argparse.ArgumentParser(description='Calcium Pipeline')

    # Required
    parser.add_argument('--movie', required=True, help='Input movie path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', default=None,
                        help='YAML configuration file (defaults to $PIPELINE_DIR/config/default.yaml)')

    # All overrides default to None so argparse values don't silently
    # override the YAML unless explicitly provided by the user.

    # Imaging
    parser.add_argument('--indicator', default=None)
    parser.add_argument('--frame-rate', type=float, default=None)

    # Detection
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
    parser.add_argument('--contour-merge-min-overlap', type=float, default=None,
                        help='Min-overlap-fraction threshold for contour merge (default 0.4)')
    parser.add_argument('--contour-merge-iou', type=float, default=None,
                        help='IoU threshold for contour merge (default 0.2)')
    parser.add_argument('--contour-merge-max-growth', type=float, default=None,
                        help='Reject merge if hull > N× largest member area (default 4.0)')

    # Motion correction
    parser.add_argument('--motion-correction', dest='motion_enabled', action='store_const', const=True, default=None)
    parser.add_argument('--no-motion-correction', dest='motion_enabled', action='store_const', const=False)
    parser.add_argument('--motion-mode', default=None, choices=['rigid', 'piecewise_rigid', 'auto', None])
    parser.add_argument('--max-shift', type=int, default=None)

    # Deconvolution
    parser.add_argument('--deconvolution', dest='deconv_enabled', action='store_const', const=True, default=None)
    parser.add_argument('--no-deconvolution', dest='deconv_enabled', action='store_const', const=False)
    parser.add_argument('--deconv-method', default=None)
    parser.add_argument('--temporal-filter', dest='temporal_filter', action='store_const', const=True, default=None)
    parser.add_argument('--no-temporal-filter', dest='temporal_filter', action='store_const', const=False)
    parser.add_argument('--filter-cutoff', type=float, default=None)

    # Baseline
    parser.add_argument('--amplitude-method', default=None,
                        choices=['direct', 'global_dff', 'local_dff', 'local_background', None])
    parser.add_argument('--edge-trim', dest='edge_trim', action='store_const', const=True, default=None)
    parser.add_argument('--no-edge-trim', dest='edge_trim', action='store_const', const=False)

    # Output: per-ROI inspection PNGs (one PNG per ROI showing peak frame sequence + trace)
    parser.add_argument('--per-roi-pngs', dest='per_roi_pngs',
                        action='store_const', const=True, default=None)
    parser.add_argument('--no-per-roi-pngs', dest='per_roi_pngs',
                        action='store_const', const=False)

    # Output: interactive HTML inspection gallery (single self-contained HTML file)
    parser.add_argument('--inspection-gallery', dest='inspection_gallery',
                        action='store_const', const=True, default=None)
    parser.add_argument('--no-inspection-gallery', dest='inspection_gallery',
                        action='store_const', const=False)

    # Output: full-movie HTML viewer with contour overlay (large file)
    parser.add_argument('--movie-gallery', dest='movie_gallery',
                        action='store_const', const=True, default=None)
    parser.add_argument('--no-movie-gallery', dest='movie_gallery',
                        action='store_const', const=False)

    # Dev / experimental
    parser.add_argument('--dev-network-analysis', dest='dev_network',
                        action='store_const', const=True, default=None)
    parser.add_argument('--no-dev-network-analysis', dest='dev_network',
                        action='store_const', const=False)

    # Compute (pipeline-level, not per-dataset, so not in YAML)
    parser.add_argument('--n-processes', type=int, default=8,
                        help='Number of parallel processes')

    return parser


def _resolve_config(cli_args):
    """Load YAML config, apply CLI overrides, flatten into an args namespace."""
    pipeline_dir = os.environ.get('PIPELINE_DIR', os.getcwd())
    config_path = cli_args.config or os.path.join(pipeline_dir, 'config', 'default.yaml')

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
        'detection.contour_merge_min_overlap': cli_args.contour_merge_min_overlap,
        'detection.contour_merge_iou':       cli_args.contour_merge_iou,
        'detection.contour_merge_max_growth': cli_args.contour_merge_max_growth,
        'motion.enabled':                 cli_args.motion_enabled,
        'motion.mode':                    cli_args.motion_mode,
        'motion.max_shift':               cli_args.max_shift,
        'deconvolution.enabled':          cli_args.deconv_enabled,
        'deconvolution.method':           cli_args.deconv_method,
        'deconvolution.temporal_filter':  cli_args.temporal_filter,
        'deconvolution.filter_cutoff':    cli_args.filter_cutoff,
        'baseline.method':                cli_args.amplitude_method,
        'baseline.edge_trim':             cli_args.edge_trim,
        'output.per_roi_pngs':            cli_args.per_roi_pngs,
        'output.inspection_gallery':      cli_args.inspection_gallery,
        'output.movie_gallery':           cli_args.movie_gallery,
        'dev.network_analysis':           cli_args.dev_network,
    }
    cfg = load_config(config_path, overrides=overrides)

    # Flatten cfg into a single namespace so existing code reads args.X
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
        use_mean_proj=cfg.detection.use_mean_proj,
        use_std_proj=cfg.detection.use_std_proj,
        n_peak_frames=cfg.detection.n_peak_frames,
        peak_percentile=cfg.detection.peak_percentile,
        contour_merge_min_overlap=cfg.detection.contour_merge_min_overlap,
        contour_merge_iou=cfg.detection.contour_merge_iou,
        contour_merge_max_growth=cfg.detection.contour_merge_max_growth,
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
        # Output (gallery flags split: per-ROI PNGs vs interactive HTML)
        per_roi_pngs=getattr(cfg.output, 'per_roi_pngs', False),
        inspection_gallery=getattr(cfg.output, 'inspection_gallery', True),
        movie_gallery=getattr(cfg.output, 'movie_gallery', False),
        movie_gallery_subsample=getattr(cfg.output, 'movie_gallery_subsample', 1),
        # Dev
        dev_network_analysis=cfg.dev.network_analysis,
    )
    return args, cfg, config_path


# =============================================================================
# MAIN
# =============================================================================

def _check_environment(args) -> None:
    """
    Verify required Python packages are importable before any pipeline work.

    Core packages are always checked.  ``nd2`` is checked only if the input
    is a Nikon file; ``caiman`` is checked only if motion correction or
    deconvolution is enabled (both stages depend on it).

    Raises SystemExit with a clear message if anything is missing, pointing
    the user at ``setup/install.sh``.
    """
    import importlib.util

    required = [
        ('numpy', 'numerical core'),
        ('scipy', 'signal processing'),
        ('skimage', 'image processing (scikit-image)'),
        ('sklearn', 'preprocessing utilities (scikit-learn)'),
        ('cv2', 'contour extraction (opencv-python)'),
        ('matplotlib', 'figure generation'),
        ('tifffile', 'TIFF I/O'),
        ('yaml', 'config loading (PyYAML)'),
        ('PIL', 'image utilities (Pillow)'),
        ('pandas', 'feature tables'),
        ('statsmodels', 'group-level statistics'),
        ('joblib', 'auto-radius parallelism'),
    ]
    if str(args.movie).lower().endswith('.nd2'):
        required.append(('nd2', 'Nikon ND2 file reading'))
    if args.motion_correction or args.deconvolution:
        why = []
        if args.motion_correction: why.append('motion correction')
        if args.deconvolution:     why.append('deconvolution')
        required.append(('caiman', ' / '.join(why)))

    missing = [(pkg, why) for pkg, why in required
               if importlib.util.find_spec(pkg) is None]

    if missing:
        lines = [f"  - {pkg:<14} (needed for: {why})" for pkg, why in missing]
        msg = (
            "Missing required Python packages:\n"
            + "\n".join(lines)
            + "\n\nInstall everything with:\n"
            + "    bash setup/install.sh\n"
            + "and then `conda activate calpipe` before re-running.\n"
        )
        raise SystemExit(msg)


def main():
    parser = _build_parser()
    cli_args = parser.parse_args()
    args, cfg, config_path = _resolve_config(cli_args)

    _check_environment(args)

    from config_loader import get_decay_time_for_indicator
    decay_time = get_decay_time_for_indicator(args.indicator)

    logger.info("=" * 70)
    logger.info("Configuration loaded")
    logger.info("=" * 70)
    logger.info(f"  Config file: {config_path}")
    logger.info(f"  Movie:       {args.movie}")
    logger.info(f"  Output:      {args.output}")
    logger.info(f"  Indicator:   {args.indicator} (decay_time={decay_time}s)")
    logger.info(f"  Frame rate:  {args.frame_rate} Hz")
    logger.info(f"  Baseline:    {args.amplitude_method}")
    logger.info(f"  Motion:      {args.motion_mode if args.motion_correction else 'disabled'}")
    logger.info(f"  Deconv:      {args.deconv_method if args.deconvolution else 'disabled'}")
    logger.info("=" * 70)

    os.makedirs(args.output, exist_ok=True)
    _ensure_output_layout(args.output)
    temp_dir = tempfile.mkdtemp(prefix='calpipe_')
    start_time = datetime.now()
    results = {'config': vars(args)}

    try:
        # ──────────────────────────────────────────────────────────────────
        # STAGE 1: Load Movie
        # ──────────────────────────────────────────────────────────────────
        logger.info("=" * 70)
        logger.info("STAGE 1: Loading Movie")
        logger.info("=" * 70)

        movie = load_movie(args.movie)
        T, d1, d2 = movie.shape
        dims = (d1, d2)
        results['dims'] = dims
        results['n_frames'] = T

        # Keep reference to raw movie for gallery background images
        movie_raw = movie

        logger.info(f"  Shape: {movie.shape}")
        logger.info(f"  dtype: {movie.dtype}")
        logger.info(f"  Intensity range: [{movie.min():.2f}, {movie.max():.2f}]")
        logger.info(f"  Mean intensity: {movie.mean():.2f}")
        logger.info(f"  Median intensity: {np.median(movie):.2f}")
        logger.info(f"  Std intensity: {movie.std():.2f}")

        if movie.max() == movie.min():
            logger.error("Movie has zero contrast — all pixels identical!")
        if movie.max() > 65000:
            logger.info("  Note: near uint16 saturation — possible clipping")
        if movie.min() < 0:
            logger.info("  Note: negative values present (denoised data?)")
        if movie.mean() < 1.0:
            logger.info("  Note: very low mean — data may already be ΔF/F")

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

        # ──────────────────────────────────────────────────────────────────
        # STAGE 2: Motion Correction
        # ──────────────────────────────────────────────────────────────────
        logger.info("=" * 70)
        logger.info("STAGE 2: Motion Correction")
        logger.info("=" * 70)

        if args.motion_correction:
            from motion_correction import correct_motion
            from diagnostics import generate_motion_figure

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
            results['motion_correction']['mode_resolved'] = mc_result.mode

            T, d1, d2 = movie.shape
            dims = (d1, d2)
            logger.info(f"  Post-MC dims: {dims} ({T} frames)")

            # Update movie_raw to the cropped movie so gallery background
            # images match the coordinate system of seeds and projections
            movie_raw = movie

            generate_motion_figure(
                mc_result, figure_path(args.output, 'motion_correction.png'),
            )
            np.save(data_path(args.output, 'motion_shifts.npy'), mc_result.shifts)

            logger.info(f"Motion correction applied (mode={mc_result.mode}): "
                        f"max shift ({mc_result.max_shift_y:.1f}, "
                        f"{mc_result.max_shift_x:.1f}) px")
        else:
            logger.info("Motion correction disabled by config")
            results['motion_correction'] = {'mode': 'disabled'}

        # ──────────────────────────────────────────────────────────────────
        # STAGE 3: Contour-Based Seed Detection
        # ──────────────────────────────────────────────────────────────────
        logger.info("=" * 70)
        logger.info("STAGE 3: Contour-Based Seed Detection")
        logger.info("=" * 70)

        # Compute projections ONCE — invariant to radius choices, used by
        # the auto-radius sweep, final detection, and visualization.
        from contour_seed_detection import compute_projections

        logger.info("Computing projections (single pass, shared downstream)...")
        shared_projections = compute_projections(
            movie, compute_correlation=True, smooth_sigma=args.smooth_sigma,
        )

        # Auto-radius estimation
        if args.auto_radius:
            from auto_radius import optimise_radius
            from diagnostics import generate_radius_figure

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
                            f"max={args.max_radius:.1f} px ")
            else:
                logger.warning(f"Auto-radius: insufficient good traces — "
                               f"keeping defaults: "
                               f"min={args.min_radius}, max={args.max_radius}")

            results['auto_radius'] = {
                k: v for k, v in radius_result.items() if k != 'all_results'
            }

            try:
                generate_radius_figure(
                    radius_result, figure_path(args.output, 'auto_radius.png'),
                )
            except Exception as e:
                logger.warning(f"Auto-radius figure failed: {e}")
        else:
            logger.info(f"Auto-radius disabled — using fixed: "
                        f"min={args.min_radius}, max={args.max_radius}")

        from contour_seed_detection import (
            detect_seeds_with_contours,
            contours_to_spatial_footprints,
        )

        seeds = detect_seeds_with_contours(
            movie,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            intensity_threshold=args.intensity_threshold,
            correlation_threshold=args.correlation_threshold,
            border_margin=args.border_margin,
            max_seeds=args.max_seeds,
            smooth_sigma=args.smooth_sigma,
            use_mean=args.use_mean_proj,
            use_std=args.use_std_proj,
            n_peak_frames=args.n_peak_frames,
            peak_percentile=args.peak_percentile,
            contour_merge_min_overlap=args.contour_merge_min_overlap,
            contour_merge_iou=args.contour_merge_iou,
            contour_merge_max_growth=args.contour_merge_max_growth,
            diagnostics_dir=diagnostics_path(args.output),
            precomputed_projections=shared_projections,
        )

        logger.info(f"Detected {seeds.n_seeds} seeds ({seeds.n_contours} with contours)")

        # Hotspot suppression diagnostic (only if smoothing applied)
        if args.smooth_sigma > 0:
            try:
                from contour_seed_detection import suppress_hotspots, visualize_hotspot_suppression
                movie_smoothed = suppress_hotspots(movie, method='gaussian', sigma=args.smooth_sigma)
                visualize_hotspot_suppression(
                    movie, movie_smoothed,
                    diagnostics_path(args.output, 'hotspot_suppression.png'),
                    sigma=args.smooth_sigma, method='gaussian', dpi=150,
                )
                del movie_smoothed
            except Exception as hs_err:
                logger.warning(f"Hotspot suppression diagnostic failed: {hs_err}")

        # Reuse shared projections for visualization (match what detection saw)
        projections = shared_projections

        # Unsmoothed projections for gallery backgrounds (cheap — no correlation, no smoothing)
        projections_raw = compute_projections(movie, compute_correlation=False, smooth_sigma=0.0)

        np.save(data_path(args.output, 'max_projection_raw.npy'), projections_raw.max_proj)
        np.save(data_path(args.output, 'std_projection.npy'), projections_raw.std_proj)
        np.save(data_path(args.output, 'correlation_image.npy'), projections.correlation)

        if seeds.n_seeds == 0:
            logger.error("No seeds detected!")
            return results

        A_init = contours_to_spatial_footprints(seeds, dims, contour_fallback=True)

        # ──────────────────────────────────────────────────────────────────
        # STAGE 4a: Trace Extraction
        # ──────────────────────────────────────────────────────────────────
        logger.info("=" * 70)
        logger.info("STAGE 4a: Trace Extraction")
        logger.info("=" * 70)

        logger.info(f"Using decay_time={decay_time}s for {args.indicator}")
        logger.info(f"Amplitude method: {args.amplitude_method}")

        from trace_extraction import extract_traces

        try:
            A = A_init
            C_raw_fluorescence, _ = extract_traces(
                movie, A_init, chunk_size=500,
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

        # ──────────────────────────────────────────────────────────────────
        # STAGE 4b: Baseline Correction
        # ──────────────────────────────────────────────────────────────────
        # Method selection (set via baseline.method in YAML):
        #   'direct'           — raw traces straight to OASIS (it estimates its own baseline)
        #   'global_dff'       — per-trace rolling percentile ΔF/F₀ (default)
        #   'local_dff'        — same as global_dff (reserved for future use)
        #   'local_background' — tissue-masked annulus background

        if args.amplitude_method == 'direct':
            logger.info("=" * 70)
            logger.info("STAGE 4b: Direct (no baseline correction)")
            logger.info("=" * 70)
            logger.info("  Raw traces passed directly to OASIS")

            C = C_raw_fluorescence.copy()
            C_raw = C_raw_fluorescence
            dff_info = {
                'method': 'direct',
                'note': 'Raw fluorescence passed to OASIS — no ΔF/F conversion',
            }

        elif args.amplitude_method == 'local_background':
            logger.info("=" * 70)
            logger.info("STAGE 4b: Local Tissue-Masked Background")
            logger.info("=" * 70)

            from preprocessing import compute_dff_local_background

            C, C_raw, dff_info = compute_dff_local_background(
                movie, A_init,
                frame_rate=args.frame_rate,
                percentile=8.0,
                window_fraction=0.25,
                min_window=50, max_window=500,
                annulus_inner_gap=2, annulus_outer_radius=20,
                edge_trim=args.edge_trim,
            )
            C_raw_fluorescence = C_raw
        else:
            # global_dff or local_dff: per-trace rolling percentile
            logger.info("=" * 70)
            logger.info("STAGE 4b: Per-Trace ΔF/F₀ Baseline Correction")
            logger.info("=" * 70)

            from preprocessing import compute_dff_traces

            C, C_raw, dff_info = compute_dff_traces(
                C_raw_fluorescence,
                frame_rate=args.frame_rate,
                percentile=8.0,
                window_fraction=0.25,
                min_window=50, max_window=500,
                edge_trim=args.edge_trim,
            )

        results['dff_correction'] = dff_info
        results['amplitude_method'] = args.amplitude_method

        # Baseline correction diagnostic figure (local-background only)
        if args.amplitude_method == 'local_background':
            try:
                from diagnostics import generate_local_background_diagnostic
                generate_local_background_diagnostic(
                    movie=movie,
                    A=A_init,
                    C_raw=C_raw,
                    C_dff=C,
                    dff_info=dff_info,
                    output_dir=diagnostics_path(args.output),
                    frame_rate=args.frame_rate,
                )
            except Exception as diag_err:
                logger.warning(f"Local-background diagnostic failed: {diag_err}")
                import traceback
                traceback.print_exc()

        # ──────────────────────────────────────────────────────────────────
        # STAGE 5a: Temporal Filtering (optional)
        # ──────────────────────────────────────────────────────────────────
        C_raw_unfilt = C.copy()

        if args.temporal_filter:
            logger.info("=" * 70)
            logger.info("STAGE 5a: Temporal Filtering")
            logger.info("=" * 70)

            from deconvolution import temporal_filter
            C = temporal_filter(C, args.frame_rate, cutoff_hz=args.filter_cutoff)
            results['temporal_filter'] = {
                'applied': True,
                'cutoff_hz': args.filter_cutoff,
            }
        else:
            results['temporal_filter'] = {'applied': False}

        # ──────────────────────────────────────────────────────────────────
        # STAGE 5b: Population Drift Removal
        # ──────────────────────────────────────────────────────────────────
        # Recording-wide artefacts (focus shifts, illumination changes,
        # medium disturbances) cause all ROIs to drift simultaneously.
        # These are SLOW (>5s) compared to calcium transients (~0.5-2s).
        #
        # Compute the population median trace, smooth it with a wide rolling
        # window (~10s) to remove fast synchronous transients, then subtract
        # the smoothed drift from each trace. Preserves synchronous calcium
        # events while removing slow recording artefacts.
        logger.info("=" * 70)
        logger.info("STAGE 5b: Population Drift Removal")
        logger.info("=" * 70)

        pop_median = np.median(C, axis=0)

        drift_window = max(5, int(args.frame_rate * 10))  # ~10 seconds
        if drift_window % 2 == 0:
            drift_window += 1  # must be odd for median filter

        from scipy.ndimage import median_filter
        pop_drift = median_filter(pop_median, size=drift_window, mode='reflect')

        drift_range = np.max(pop_drift) - np.min(pop_drift)
        logger.info(f"  Population drift (smoothed, {drift_window}-frame window): "
                    f"range={drift_range:.4f}")

        if drift_range > 0.005:
            C = C - pop_drift[np.newaxis, :]
            logger.info(f"  Subtracted slow population drift (range={drift_range:.4f})")
            results['population_drift'] = {
                'subtracted': True, 'range': float(drift_range),
                'window_frames': int(drift_window),
            }
        else:
            logger.info(f"  Population drift negligible, skipping subtraction")
            results['population_drift'] = {
                'subtracted': False, 'range': float(drift_range),
            }

        # ──────────────────────────────────────────────────────────────────
        # STAGE 5c: Spike Deconvolution
        # ──────────────────────────────────────────────────────────────────
        deconv_result = None

        if args.deconvolution:
            logger.info("=" * 70)
            logger.info("STAGE 5c: Spike Deconvolution")
            logger.info("=" * 70)

            from deconvolution import deconvolve_traces

            deconv_result = deconvolve_traces(
                C,
                frame_rate=args.frame_rate,
                decay_time=decay_time,
                method=args.deconv_method,
            )

            np.save(data_path(args.output, 'spike_trains.npy'),
                    deconv_result['S'])
            np.save(data_path(args.output, 'traces_denoised.npy'),
                    deconv_result['C_denoised'])
            np.save(data_path(args.output, 'deconv_noise.npy'),
                    deconv_result['noise'])

            results['deconvolution'] = {
                'method': deconv_result['method'],
                'total_spikes': int(deconv_result['n_spikes'].sum()),
                'median_spikes_per_neuron': float(np.median(deconv_result['n_spikes'])),
                'mean_noise': float(np.mean(deconv_result['noise'])),
            }

            # Deconvolution diagnostic figure
            try:
                from diagnostics import generate_deconvolution_figure
                generate_deconvolution_figure(
                    C_raw_unfilt, deconv_result, args.frame_rate,
                    figure_path(args.output, 'deconvolution.png'),
                    C_filtered=C if args.temporal_filter else None,
                )
            except Exception as fig_err:
                logger.warning(f"Deconv figure failed: {fig_err}")

            # Per-ROI presentation trace figures
            try:
                from diagnostics import save_roi_trace_figures
                save_roi_trace_figures(
                    C_raw_unfilt, deconv_result, args.frame_rate, args.output,
                )
            except Exception as roi_fig_err:
                logger.warning(f"ROI trace figures failed: {roi_fig_err}")

            # Decay parameter diagnostics
            try:
                from diagnostics import generate_decay_diagnostics
                generate_decay_diagnostics(
                    deconv_result, C_raw_unfilt, args.frame_rate,
                    decay_time, figure_path(args.output),
                )
            except Exception as decay_err:
                logger.warning(f"Decay diagnostics failed: {decay_err}")

            logger.info(f"  Spikes: {deconv_result['n_spikes'].sum()} total, "
                        f"median {np.median(deconv_result['n_spikes']):.0f}/neuron")
        else:
            results['deconvolution'] = {'method': 'disabled'}

        # ──────────────────────────────────────────────────────────────────
        # STAGE 6: Save Results
        # ──────────────────────────────────────────────────────────────────
        logger.info("=" * 70)
        logger.info("STAGE 6: Saving Results")
        logger.info("=" * 70)

        A_final = A
        C_final = C
        n_final = A.shape[1]

        save_npz(data_path(args.output, 'spatial_footprints.npz'), csc_matrix(A_final))
        np.save(data_path(args.output, 'temporal_traces.npy'), C_final)

        # Projection images (used by group analysis for ROI inspection)
        if hasattr(projections, 'max_proj') and projections.max_proj is not None:
            np.save(data_path(args.output, 'max_projection.npy'), projections.max_proj)
        if hasattr(projections, 'mean_proj') and projections.mean_proj is not None:
            np.save(data_path(args.output, 'mean_projection.npy'), projections.mean_proj)

        # Raw traces for downstream local-ΔF/F amplitude measurement
        if C_raw is not None:
            np.save(data_path(args.output, 'temporal_traces_raw.npy'), C_raw)

        results['n_final'] = n_final

        elapsed = (datetime.now() - start_time).total_seconds()
        results['elapsed_seconds'] = elapsed

        with open(os.path.join(args.output, 'run_info.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"Final neurons: {n_final}")

        # ──────────────────────────────────────────────────────────────────
        # STAGE 7: Visual Outputs
        # ──────────────────────────────────────────────────────────────────
        logger.info("=" * 70)
        logger.info("STAGE 7: Visual Outputs")
        logger.info("=" * 70)

        # Free large arrays before visualisation to reduce memory pressure
        gc.collect()

        # 7a. Per-ROI inspection PNGs (off by default, slow)
        if args.per_roi_pngs:
            try:
                from diagnostics import generate_per_roi_pngs
                generate_per_roi_pngs(
                    A_final, C_final, movie, dims, projections,
                    args.frame_rate, figure_path(args.output),
                )
            except Exception as e:
                logger.warning(f"Per-ROI PNG generation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.info("  Per-ROI PNGs disabled "
                        "(set output.per_roi_pngs: true to enable)")

        # 7b. Interactive HTML inspection gallery
        if args.inspection_gallery:
            logger.info("  Creating interactive HTML inspection gallery...")
            try:
                from interactive_gallery import generate_interactive_gallery
                gallery_path = generate_interactive_gallery(
                    seeds=seeds,
                    projections=projections_raw,
                    movie=movie_raw,
                    movie_processed=movie_raw,
                    output_path=os.path.join(args.output, 'gallery.html'),
                    title=f"ROI Detection - {os.path.basename(args.movie)}",
                    max_rois=500,
                    traces_denoised=deconv_result['C_denoised'] if deconv_result else None,
                    spike_trains=deconv_result['S'] if deconv_result else None,
                    pipeline_traces_dff=C,
                    pipeline_traces_raw=C_raw_fluorescence,
                )
                logger.info(f"  Interactive gallery saved to: {gallery_path}")
            except Exception as gallery_err:
                logger.warning(f"Interactive gallery failed: {gallery_err}")
                import traceback
                traceback.print_exc()
        else:
            logger.info("  Interactive gallery disabled "
                        "(set output.inspection_gallery: true to enable)")

        # 7c. Movie gallery (full-movie HTML viewer with contour overlay)
        if args.movie_gallery:
            logger.info("  Creating movie gallery (self-contained HTML)...")
            try:
                from movie_gallery import generate_movie_gallery
                mg_result = generate_movie_gallery(
                    movie=movie_raw,
                    seeds=seeds,
                    output_dir=args.output,
                    frame_rate=args.frame_rate,
                    subsample=int(args.movie_gallery_subsample),
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
            logger.info("  Movie gallery disabled "
                        "(set output.movie_gallery: true to enable)")

        # ──────────────────────────────────────────────────────────────────
        # STAGE 8: Development / Experimental Modules
        # ──────────────────────────────────────────────────────────────────
        if args.dev_network_analysis:
            try:
                logger.info("")
                logger.info("=" * 70)
                logger.info("STAGE 8: Development — Network Analysis (detection-free)")
                logger.info("=" * 70)

                analysis_movie = movie  # motion-corrected if MC was applied

                sys.path.insert(0, os.path.join(
                    os.environ.get('PIPELINE_DIR', '.'), 'src', 'dev'))

                # 8a: Variance-weighted global trace + spectral analysis
                from network_spectral import run_network_spectral
                spectral_result = run_network_spectral(
                    analysis_movie,
                    frame_rate=args.frame_rate,
                    output_dir=args.output,
                    dataset_name=os.path.basename(args.movie),
                )
                results['dev_spectral'] = spectral_result['spectral_summary']

                # 8b: PCA spatial decomposition (auto-downsample on large FOVs)
                from network_pca import run_network_pca
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

                # Re-write run_info.json so it includes the dev results too
                with open(os.path.join(args.output, 'run_info.json'), 'w') as f:
                    json.dump(results, f, indent=2, default=str)

            except Exception as e:
                logger.warning(f"Development network analysis failed: {e}")
                import traceback
                traceback.print_exc()

        # ──────────────────────────────────────────────────────────────────
        # Summary
        # ──────────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Seeds detected: {seeds.n_seeds}")
        print(f"Contour success: {seeds.contour_success_rate:.1%}")
        print(f"Final neurons: {n_final}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Output: {args.output}")
        if args.inspection_gallery:
            print(f"\nOpen gallery.html in browser for visual inspection")
        print("=" * 60)

        return results

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        gc.collect()


if __name__ == '__main__':
    main()
