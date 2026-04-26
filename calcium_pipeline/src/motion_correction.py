"""
Motion Correction Module — CaImAn NoRMCorre
=============================================

Wrapper around CaImAn's NoRMCorre for rigid and piecewise-rigid
motion correction.  NoRMCorre is the gold-standard online algorithm
for calcium imaging motion correction.

References:
- Pnevmatikakis & Giovannucci, "NoRMCorre: An online algorithm for
  piecewise rigid motion correction of calcium imaging data",
  J. Neurosci. Methods 2017.
"""

import numpy as np
import logging
import tempfile
import os
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MotionCorrectionResult:
    """Container for motion correction outputs."""
    corrected: np.ndarray              # (T, d1, d2) corrected movie
    shifts: np.ndarray                 # (T, 2) per-frame [dy, dx] shifts
    template: np.ndarray               # (d1, d2) reference template
    correlations: np.ndarray           # (T,) frame-to-template correlation
    mode: str = 'rigid'
    max_shift_y: float = 0.0
    max_shift_x: float = 0.0
    mean_shift_y: float = 0.0
    mean_shift_x: float = 0.0
    elapsed_seconds: float = 0.0
    crop_y: int = 0                    # pixels cropped from each Y border
    crop_x: int = 0                    # pixels cropped from each X border

    def summary(self) -> Dict:
        return {
            'mode': self.mode,
            'max_shift_y': float(self.max_shift_y),
            'max_shift_x': float(self.max_shift_x),
            'mean_shift_y': float(self.mean_shift_y),
            'mean_shift_x': float(self.mean_shift_x),
            'elapsed_seconds': float(self.elapsed_seconds),
            'crop_y': self.crop_y,
            'crop_x': self.crop_x,
        }


def _resolve_tmpdir() -> str:
    """
    Find a temp directory with sufficient space for NoRMCorre memmaps.

    On HPC clusters, the user's home and scratch directories share a disk
    quota that is easily exceeded by large memmaps.  The SGE job-local
    directory ($TMPDIR → /local/<jobid>/) is on node-local storage with
    no quota, so we try it first.

    Priority order:
    1. $TMPDIR  — SGE/SLURM/PBS job-local scratch (no quota, fast)
    2. $SCRATCH_DIR/tmp — shared scratch (has quota but more space)
    3. ~/tmp — last resort before system default
    """
    candidates = []

    # 1. SGE/SLURM/PBS job-local scratch — quota-free, node-local disk
    job_tmp = os.environ.get('TMPDIR')
    if job_tmp:
        candidates.append(job_tmp)

    # 2. System /tmp — on most HPC nodes this is local disk, quota-free.
    #    Prefer it over scratch/home which are NFS with user quotas.
    candidates.append('/tmp')

    # 3. Scratch directory (HPC convention) — NFS, subject to quota
    scratch = os.environ.get('SCRATCH_DIR')
    if scratch:
        candidates.append(os.path.join(scratch, 'tmp'))

    # 4. User home tmp (last resort before system default)
    home_tmp = os.path.expanduser('~/tmp')
    candidates.append(home_tmp)

    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            # Quick space check: need at least 2 GB free
            stat = os.statvfs(candidate)
            free_bytes = stat.f_bavail * stat.f_frsize
            if free_bytes < 2 * 1024**3:
                logger.info(f"  Skipping tmpdir {candidate}: "
                            f"only {free_bytes / 1024**3:.1f} GB free")
                continue
            # statvfs reports filesystem free space but can't detect
            # per-user quota limits.  Do a small write test to confirm
            # we can actually write to this location.
            test_file = os.path.join(candidate, '.write_test')
            try:
                with open(test_file, 'wb') as f:
                    f.write(b'x' * 4096)
                os.remove(test_file)
            except OSError as e:
                logger.info(f"  Skipping tmpdir {candidate}: "
                            f"write test failed ({e})")
                continue
            logger.info(f"  Using tmpdir: {candidate} "
                        f"({free_bytes / 1024**3:.1f} GB free)")
            return candidate
        except OSError as e:
            logger.info(f"  Skipping tmpdir {candidate}: {e}")
            continue

    # Fall back to system default (tempfile.gettempdir())
    default = tempfile.gettempdir()
    logger.info(f"  Falling back to system tmpdir: {default}")
    return default


def correct_motion(
    movie: np.ndarray,
    *,
    mode: str = 'rigid',
    max_shift: int = 20,
    pw_strides: Tuple[int, int] = (96, 96),
    pw_overlaps: Tuple[int, int] = (48, 48),
    niter_rig: int = 2,
    num_frames_split: int = 100,
    n_processes: int = 1,
) -> MotionCorrectionResult:
    """
    Correct motion using CaImAn's NoRMCorre.

    Parameters
    ----------
    movie : array (T, d1, d2)
    mode : 'rigid', 'piecewise_rigid', or 'auto'
    max_shift : int
        Maximum shift in pixels.
    pw_strides, pw_overlaps : tuple
        Patch parameters for piecewise-rigid mode.
    niter_rig : int
        Number of rigid correction passes (default 2).
    num_frames_split : int
        Chunk size for online processing.
    n_processes : int
        Number of parallel processes.

    Returns
    -------
    MotionCorrectionResult
    """
    import time
    t0 = time.time()

    T, d1, d2 = movie.shape

    requested_mode = mode
    if mode == 'auto':
        mode = 'piecewise_rigid' if max(d1, d2) > 256 else 'rigid'

    logger.info(f"Motion correction (CaImAn NoRMCorre)")
    logger.info(f"  Mode:        {mode}"
                + (f"  (auto-selected from '{requested_mode}')"
                   if requested_mode == 'auto' else ""))
    logger.info(f"  Max shift:   {max_shift} px")
    logger.info(f"  Movie shape: ({T}, {d1}, {d2})")
    if mode == 'piecewise_rigid':
        logger.info(f"  Patch size:  {pw_strides} px (stride)")
        logger.info(f"  Overlap:     {pw_overlaps} px")
        logger.info(f"  Rigid iters: {niter_rig}")
    logger.info(f"  Chunk size:  {num_frames_split} frames")
    logger.info(f"  Processes:   {n_processes}")

    result = _run_normcorre(
        movie, mode=mode, max_shift=max_shift,
        pw_strides=pw_strides, pw_overlaps=pw_overlaps,
        niter_rig=niter_rig, num_frames_split=num_frames_split,
        n_processes=n_processes,
    )

    result.elapsed_seconds = time.time() - t0
    logger.info(f"  Done in {result.elapsed_seconds:.1f}s  |  "
                f"max shift=({result.max_shift_y:.2f}, {result.max_shift_x:.2f}) px, "
                f"mean shift=({result.mean_shift_y:.2f}, {result.mean_shift_x:.2f}) px")

    return result


def _run_normcorre(
    movie, *, mode, max_shift, pw_strides, pw_overlaps,
    niter_rig, num_frames_split, n_processes,
) -> MotionCorrectionResult:
    """Run CaImAn NoRMCorre."""
    import caiman as cm
    from caiman.motion_correction import MotionCorrect

    T, d1, d2 = movie.shape

    # Use a temp directory with enough disk space (avoids /tmp overflow
    # on HPC nodes where /tmp is often a small RAM disk)
    base_tmpdir = _resolve_tmpdir()
    tmpdir = tempfile.mkdtemp(prefix='normcorre_', dir=base_tmpdir)
    logger.info(f"  NoRMCorre tmpdir: {tmpdir}")

    # Redirect CaImAn's own temp files to the same location.
    # By default CaImAn writes to ~/caiman_data/temp/ which is on the
    # home filesystem — on HPC clusters this often has a small quota and fills up
    # with large movies.  Setting CAIMAN_DATA overrides this.
    old_caiman_data = os.environ.get('CAIMAN_DATA')
    caiman_tempdir = os.path.join(tmpdir, 'caiman_data')
    os.makedirs(os.path.join(caiman_tempdir, 'temp'), exist_ok=True)
    os.environ['CAIMAN_DATA'] = caiman_tempdir

    try:
        # CaImAn's MotionCorrect works most reliably with file paths.
        # Save movie as a TIFF that CaImAn can load natively.
        #
        # Sanitise first: edge/border pixels can be NaN or Inf (e.g. from
        # prior shifts or padding), and CaImAn internally computes
        # -movie.min() to make the data non-negative.  If min() returns
        # NaN/None this causes "bad operand type for unary -: 'NoneType'".
        import tifffile
        movie_clean = movie.astype(np.float32)
        bad_mask = ~np.isfinite(movie_clean)
        if bad_mask.any():
            n_bad = int(bad_mask.sum())
            logger.info(f"  Replacing {n_bad} NaN/Inf pixels before NoRMCorre")
            movie_clean[bad_mask] = 0.0
        # Ensure non-negative — CaImAn expects this
        movie_min = float(movie_clean.min())
        if movie_min < 0:
            logger.info(f"  Offsetting movie by {-movie_min:.2f} to make non-negative")
            movie_clean -= movie_min
        fname_tif = os.path.join(tmpdir, 'movie.tif')
        tifffile.imwrite(fname_tif, movie_clean)
        del movie_clean
        logger.info(f"  Wrote temp TIFF: {fname_tif} "
                    f"({os.path.getsize(fname_tif) / 1024**3:.2f} GB)")

        # Build NoRMCorre parameters
        mc_dict = {
            'max_shifts': (max_shift, max_shift),
            'niter_rig': niter_rig,
            'splits_rig': max(1, T // num_frames_split),
            'num_frames_split': num_frames_split,
            'border_nan': 'copy',
            'min_mov': 0,  # movie is already non-negative; skip CaImAn's
                           # internal min_mov detection which can return None
                           # on large TIFFs and crash with
                           # "bad operand type for unary -: 'NoneType'"
        }

        if mode == 'piecewise_rigid':
            mc_dict.update({
                'strides': pw_strides,
                'overlaps': pw_overlaps,
                'splits_els': max(1, T // num_frames_split),
                'max_deviation_rigid': max_shift // 2,
                'pw_rigid': True,
            })

        # Run correction on the file
        mc = MotionCorrect([fname_tif], dview=None, **mc_dict)

        if mode == 'rigid':
            mc.motion_correct_rigid()
        else:
            mc.motion_correct_pwrigid()

        # Load corrected movie from CaImAn's output memmap.
        # Different CaImAn versions store the output path in different
        # attributes; try them in order of likelihood.
        #
        # Log all available attributes for debugging version-specific
        # quirks, then try to load from each candidate path.
        for attr in ('fname_tot_rig', 'fname_tot_els', 'mmap_file',
                     'fname_tot_rig', 'total_template_rig'):
            val = getattr(mc, attr, 'NOT_SET')
            logger.info(f"  mc.{attr} = {type(val).__name__}: {repr(val)[:120]}")

        corrected = None
        # Strategy 1: load from known file-path attributes
        for attr in ('fname_tot_rig', 'fname_tot_els', 'mmap_file'):
            candidate = getattr(mc, attr, None)
            if candidate is None:
                continue
            if isinstance(candidate, list):
                candidate = candidate[0] if candidate else None
            if candidate is None:
                continue
            if not isinstance(candidate, (str, bytes)):
                logger.info(f"  Skipping mc.{attr}: not a path ({type(candidate).__name__})")
                continue
            try:
                corrected = np.array(cm.load(candidate), dtype=np.float32)
                logger.info(f"  Loaded corrected movie from mc.{attr}: {candidate}")
                break
            except Exception as load_err:
                logger.info(f"  Failed to load from mc.{attr}: {load_err}")
                continue

        # Strategy 2: look for memmap files CaImAn wrote in our tmpdir
        if corrected is None:
            import glob
            mmap_files = glob.glob(os.path.join(tmpdir, '*.mmap'))
            logger.info(f"  Searching tmpdir for mmaps: found {len(mmap_files)}")
            for mf in sorted(mmap_files, key=os.path.getsize, reverse=True):
                try:
                    corrected = np.array(cm.load(mf), dtype=np.float32)
                    logger.info(f"  Loaded corrected movie from mmap: {mf}")
                    break
                except Exception as mmap_err:
                    logger.info(f"  Failed to load {mf}: {mmap_err}")

        # Strategy 3: apply shifts manually using mode='nearest'
        if corrected is None:
            logger.warning("  No CaImAn output found — applying shifts manually")
            from scipy.ndimage import shift as ndi_shift

            shifts_rig = mc.shifts_rig
            if shifts_rig is None:
                raise RuntimeError("NoRMCorre produced no shifts — correction failed entirely")
            corrected = np.empty_like(movie, dtype=np.float32)
            for t in range(T):
                dy, dx = shifts_rig[t]
                corrected[t] = ndi_shift(
                    movie[t].astype(np.float64),
                    [dy, dx],
                    order=1,
                    mode='nearest',
                ).astype(np.float32)
            logger.info(f"  Applied {T} frame shifts manually")

        # Extract shifts
        if mode == 'rigid':
            shifts_list = mc.shifts_rig
            shifts = np.array([[s[0], s[1]] for s in shifts_list])
        else:
            shifts_list = mc.x_shifts_els if hasattr(mc, 'x_shifts_els') else []
            if hasattr(mc, 'shifts_rig') and mc.shifts_rig is not None:
                shifts = np.array([[s[0], s[1]] for s in mc.shifts_rig])
            else:
                shifts = np.zeros((T, 2))

        # Crop the border region that was shifted on/off the image.
        # These pixels are either NaN (border_nan='copy' can still leave
        # artefacts) or filled with edge values that don't reflect real
        # fluorescence.  Cropping avoids false-zero contamination of
        # baseline, ΔF/F, and quality scores for border ROIs.
        abs_shifts = np.abs(shifts)
        crop_y = int(np.ceil(abs_shifts[:, 0].max())) + 1 if len(shifts) > 0 else 0
        crop_x = int(np.ceil(abs_shifts[:, 1].max())) + 1 if len(shifts) > 0 else 0

        if crop_y > 0 or crop_x > 0:
            old_shape = corrected.shape
            corrected = corrected[:,
                                  crop_y : d1 - crop_y,
                                  crop_x : d2 - crop_x]
            logger.info(f"  Cropped motion borders: {crop_y}px (Y), {crop_x}px (X)  "
                        f"{old_shape} -> {corrected.shape}")

        # Safety: replace any remaining NaN/Inf (shouldn't happen after
        # crop, but guard against piecewise-rigid interior artefacts)
        nan_mask = ~np.isfinite(corrected)
        if nan_mask.any():
            logger.info(f"  Replacing {int(nan_mask.sum())} residual NaN pixels")
            corrected[nan_mask] = np.nanmedian(corrected)

        template = mc.total_template_rig if hasattr(mc, 'total_template_rig') and mc.total_template_rig is not None else np.mean(corrected[:200], axis=0)
        # Crop template to match
        if crop_y > 0 or crop_x > 0:
            if template.shape[0] == d1:
                template = template[crop_y : d1 - crop_y, crop_x : d2 - crop_x]

        # Correlations
        correlations = np.array([
            np.corrcoef(template.ravel(), corrected[t].ravel())[0, 1]
            for t in range(min(T, 500))
        ])
        if T > 500:
            correlations = np.concatenate([
                correlations,
                np.full(T - 500, correlations[-1])
            ])

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        # Restore original CAIMAN_DATA
        if old_caiman_data is not None:
            os.environ['CAIMAN_DATA'] = old_caiman_data
        else:
            os.environ.pop('CAIMAN_DATA', None)

    return MotionCorrectionResult(
        corrected=corrected.astype(np.float32),
        shifts=shifts,
        template=template.astype(np.float32),
        correlations=correlations,
        mode=mode,
        max_shift_y=float(abs_shifts[:, 0].max()) if len(shifts) > 0 else 0,
        max_shift_x=float(abs_shifts[:, 1].max()) if len(shifts) > 0 else 0,
        mean_shift_y=float(abs_shifts[:, 0].mean()) if len(shifts) > 0 else 0,
        mean_shift_x=float(abs_shifts[:, 1].mean()) if len(shifts) > 0 else 0,
        crop_y=crop_y,
        crop_x=crop_x,
    )


def generate_motion_figure(
    result: MotionCorrectionResult,
    output_path: str,
) -> str:
    """Save motion correction diagnostic figure."""
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
