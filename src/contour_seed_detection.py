"""
Contour-Based Seed Detection Module
=========================================

Hybrid approach combining:
- Multi-projection blob detection (LoG on normalised projections)
- Contour-based ROI segmentation (per-blob Otsu thresholding)

Key features:
- Contour-based ROIs capture irregular neuron morphology
- Otsu thresholding adapts to local intensity variations
- Per-blob segmentation handles variable fluorescence
- Gaussian locality mask prevents contour leakage into neighbours
- Multi-projection fusion (max + correlation + mean + std)
- Extensive diagnostics and fallback mechanisms
"""

import numpy as np
import logging
import cv2
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from scipy import ndimage
from scipy.sparse import csc_matrix, lil_matrix
from skimage.feature import blob_log
import time
import json
import os

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ContourInfo:
    """Information about a single extracted contour."""
    contour: np.ndarray          # OpenCV contour array (N, 1, 2)
    center: Tuple[float, float]  # (row, col) centroid from moments
    area: float                  # Contour area in pixels
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) bounding box
    circularity: float           # 4π × area / perimeter²
    solidity: float              # area / convex_hull_area
    mean_intensity: float        # Mean intensity within contour
    
    def to_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Convert contour to binary mask."""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, 255, -1)
        return mask


@dataclass
class ContourSeedResult:
    """Extended seed detection results with contour information."""
    # Core seed data (compatible with original SeedResult)
    centers: np.ndarray          # (N, 2) array of (row, col) coordinates
    radii: np.ndarray            # (N,) array of estimated radii
    intensities: np.ndarray      # (N,) array of seed intensities
    confidence: np.ndarray       # (N,) confidence scores 0-1
    
    # Extended contour data
    contours: List[Optional[ContourInfo]]  # Contour for each seed (None if fallback)
    contour_success: np.ndarray  # (N,) bool - whether contour extraction succeeded
    source_projection: np.ndarray  # (N,) string codes: 'max', 'corr', 'both'
    boundary_touching: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    # (N,) bool - whether contour touches image boundary (confidence penalised, flagged in gallery)
    
    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_seeds(self) -> int:
        return len(self.centers)
    
    @property
    def n_contours(self) -> int:
        return int(self.contour_success.sum())
    
    @property
    def contour_success_rate(self) -> float:
        if self.n_seeds == 0:
            return 0.0
        return self.n_contours / self.n_seeds
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (contours excluded due to size)."""
        return {
            'centers': self.centers.tolist(),
            'radii': self.radii.tolist(),
            'intensities': self.intensities.tolist(),
            'confidence': self.confidence.tolist(),
            'n_seeds': self.n_seeds,
            'n_contours': self.n_contours,
            'contour_success_rate': self.contour_success_rate,
            'source_projection': self.source_projection.tolist(),
            'boundary_touching': self.boundary_touching.tolist() if len(self.boundary_touching) else [],
            'diagnostics': self.diagnostics,
        }
    
@dataclass
class ProjectionSet:
    """Collection of projection images with metadata."""
    max_proj: np.ndarray
    mean_proj: np.ndarray
    std_proj: np.ndarray
    correlation: np.ndarray
    percentile_95: np.ndarray
    
    # Normalized versions (0-1 range)
    max_norm: np.ndarray = field(init=False)
    mean_norm: np.ndarray = field(init=False)
    std_norm: np.ndarray = field(init=False)
    correlation_norm: np.ndarray = field(init=False)
    percentile_95_norm: np.ndarray = field(init=False)
    
    # Computation metadata
    compute_time_seconds: float = 0.0
    movie_shape: Tuple[int, int, int] = (0, 0, 0)
    
    def __post_init__(self):
        """Compute normalized versions."""
        self.max_norm = self._normalize(self.max_proj)
        self.mean_norm = self._normalize(self.mean_proj)
        self.std_norm = self._normalize(self.std_proj)
        self.correlation_norm = self._normalize(self.correlation)
        self.percentile_95_norm = self._normalize(self.percentile_95)
    
    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range using min-max scaling.
        
        Used by detection (blob finding operates on these normalised images).
        For figure output, use percentile clipping separately — see
        save_projection_figures().
        """
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return ((img - img_min) / (img_max - img_min)).astype(np.float32)
        return np.zeros_like(img, dtype=np.float32)
    
def compute_projections_extended(
    movie: np.ndarray,
    compute_correlation: bool = True,
    smooth_sigma: float = 0.0,
) -> ProjectionSet:
    """
    Compute multiple projection images from movie.

    Parameters
    ----------
    movie : np.ndarray
        Movie with shape (T, Y, X)
    compute_correlation : bool
        Whether to compute local correlation image (slower)
    smooth_sigma : float
        If > 0, apply per-frame Gaussian smoothing before computing projections.
        Suppresses small bright features (hotspots) while preserving larger
        cell-body-scale structures. Recommended: 3-5.

        This function is designed to be called ONCE per pipeline run;
        results should be cached and passed via ``precomputed_projections``
        to downstream functions (auto_radius, detect_seeds_with_contours).

    Returns
    -------
    ProjectionSet
        Collection of projection images
    """
    start_time = time.time()
    T, d1, d2 = movie.shape

    logger.info(f"Computing projections for movie shape {movie.shape}...")

    # Optional per-frame spatial smoothing for hotspot suppression
    if smooth_sigma > 0:
        logger.info(f"  Applying per-frame Gaussian smoothing (sigma={smooth_sigma})...")
        from scipy.ndimage import gaussian_filter
        movie_smooth = np.zeros_like(movie, dtype=np.float32)
        for t in range(T):
            movie_smooth[t] = gaussian_filter(movie[t].astype(np.float32), sigma=smooth_sigma)
        movie_for_proj = movie_smooth
        logger.info(f"  Smoothing complete ({T} frames)")
    else:
        movie_for_proj = movie

    # Basic projections
    max_proj = np.max(movie_for_proj, axis=0).astype(np.float32)
    mean_proj = np.mean(movie_for_proj, axis=0).astype(np.float32)
    std_proj = np.std(movie_for_proj, axis=0).astype(np.float32)
    percentile_95 = np.percentile(movie_for_proj, 95, axis=0).astype(np.float32)

    logger.info("  Computed: max, mean, std, 95th percentile")

    # Local correlation image
    if compute_correlation:
        logger.info("  Computing local correlation image...")

        # Z-score along the time axis.
        # Use the smoothed movie if available — makes correlation reflect
        # cell-body-scale activity, not hotspot-scale.
        # Always copy to avoid mutating the caller's data.
        movie_zs = movie_for_proj.astype(np.float32, copy=True)
        tmean = movie_zs.mean(axis=0)
        movie_zs -= tmean
        tstd = movie_zs.std(axis=0)
        tstd[tstd == 0] = 1
        movie_zs /= tstd
        del tmean, tstd

        # Correlate with 8 immediate neighbours using sliced views.
        corr_img = np.zeros((d1, d2), dtype=np.float64)

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                sy = slice(max(0, -di), d1 - max(0, di))
                sx = slice(max(0, -dj), d2 - max(0, dj))
                ny = slice(max(0, di), d1 - max(0, -di))
                nx = slice(max(0, dj), d2 - max(0, -dj))

                corr_img[sy, sx] += np.mean(
                    movie_zs[:, sy, sx] * movie_zs[:, ny, nx],
                    axis=0,
                )

        corr_img /= 8
        correlation = corr_img.astype(np.float32)
        del movie_zs
    else:
        correlation = np.zeros((d1, d2), dtype=np.float32)

    # Free the smoothed copy if we made one
    if smooth_sigma > 0:
        del movie_smooth

    compute_time = time.time() - start_time
    logger.info(f"  Projection computation completed in {compute_time:.2f}s")

    return ProjectionSet(
        max_proj=max_proj,
        mean_proj=mean_proj,
        std_proj=std_proj,
        correlation=correlation,
        percentile_95=percentile_95,
        compute_time_seconds=compute_time,
        movie_shape=(T, d1, d2),
    )


def suppress_hotspots(
    movie: np.ndarray,
    method: str = 'gaussian',
    sigma: float = 4.0,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Suppress small bright calcium hotspots/puncta in movie frames.
    
    This is useful for data with calcium microdomains (synaptic terminals,
    dendritic spines, axonal varicosities) that are brighter than cell bodies.
    
    Parameters
    ----------
    movie : np.ndarray
        Movie with shape (T, Y, X)
    method : str
        Suppression method:
        - 'gaussian': Gaussian blur (smooths hotspots into surroundings)
        - 'median': Median filter (removes outlier bright pixels)
        - 'opening': Morphological opening (removes small bright features)
    sigma : float
        Gaussian sigma (for 'gaussian' method). Default 4.0.
    kernel_size : int
        Kernel size for 'median' and 'opening' methods. Default 5.
        
    Returns
    -------
    np.ndarray
        Processed movie with suppressed hotspots
        
    Example
    -------
    >>> # Suppress hotspots before detection
    >>> movie_clean = suppress_hotspots(movie, method='gaussian', sigma=4.0)
    >>> seeds = detect_seeds_with_contours(movie_clean, ...)
    """
    from scipy.ndimage import gaussian_filter, median_filter, grey_opening
    
    T, d1, d2 = movie.shape
    logger.info(f"Suppressing hotspots using method='{method}'...")
    
    movie_out = np.zeros_like(movie, dtype=np.float32)
    
    if method == 'gaussian':
        for t in range(T):
            movie_out[t] = gaussian_filter(movie[t].astype(np.float32), sigma=sigma)
    
    elif method == 'median':
        for t in range(T):
            movie_out[t] = median_filter(movie[t].astype(np.float32), size=kernel_size)
    
    elif method == 'opening':
        # Morphological opening removes bright features smaller than kernel
        from skimage.morphology import disk
        selem = disk(kernel_size // 2)
        for t in range(T):
            movie_out[t] = grey_opening(movie[t].astype(np.float32), footprint=selem)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'gaussian', 'median', or 'opening'.")
    
    logger.info(f"  Hotspot suppression complete")
    return movie_out


def visualize_hotspot_suppression(
    movie_raw: np.ndarray,
    movie_smoothed: np.ndarray,
    output_path: str,
    *,
    sigma: float = 0.0,
    method: str = 'gaussian',
    dpi: int = 150,
):
    """
    Save a diagnostic figure comparing projections before and after
    hotspot suppression.

    Produces a 2×3 panel figure:

    Row 1 (max projections):
      - Raw max projection
      - Smoothed max projection
      - Difference (raw − smoothed), highlighting suppressed hotspots

    Row 2 (detail zoom — central 25% crop):
      - Same three views zoomed in so individual hotspots are visible

    Parameters
    ----------
    movie_raw : array (T, H, W)
    movie_smoothed : array (T, H, W)
    output_path : str
    sigma : float
        Smoothing parameter used (for title annotation).
    method : str
        Method name (for title annotation).
    dpi : int
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    max_raw = np.max(movie_raw, axis=0).astype(np.float32)
    max_smooth = np.max(movie_smoothed, axis=0).astype(np.float32)
    diff = max_raw - max_smooth

    H, W = max_raw.shape

    # Display range from raw (shared for raw and smoothed)
    vlo = np.percentile(max_raw, 1)
    vhi = np.percentile(max_raw, 99.5)
    if vhi <= vlo:
        vhi = vlo + 1

    # Difference range (symmetric)
    dlim = np.percentile(np.abs(diff), 99)
    if dlim < 1e-6:
        dlim = 1.0

    # Central 25% crop for detail view
    cy, cx = H // 2, W // 2
    qh, qw = H // 4, W // 4
    s = np.s_[cy - qh:cy + qh, cx - qw:cx + qw]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.12)

    fig.suptitle(
        f'Hotspot Suppression Diagnostic  —  method={method}, sigma={sigma}',
        fontsize=13, fontweight='700', y=0.98, color='#1e293b')

    panels = [
        # (row, col, data, cmap, vmin, vmax, title)
        (0, 0, max_raw,    'gray',   vlo,  vhi,  'Raw Max Projection'),
        (0, 1, max_smooth,  'gray',   vlo,  vhi,  'Smoothed Max Projection'),
        (0, 2, diff,        'RdBu_r', -dlim, dlim, 'Difference (Raw − Smoothed)'),
        (1, 0, max_raw[s],    'gray',   vlo,  vhi,  'Raw (centre detail)'),
        (1, 1, max_smooth[s], 'gray',   vlo,  vhi,  'Smoothed (centre detail)'),
        (1, 2, diff[s],       'RdBu_r', -dlim, dlim, 'Difference (centre detail)'),
    ]

    for row, col, data, cmap, v0, v1, title in panels:
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(data, cmap=cmap, vmin=v0, vmax=v1)
        ax.set_title(title, fontsize=10, fontweight='600', color='#1e293b')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # Colorbar only on difference panels
        if cmap == 'RdBu_r':
            cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cb.ax.tick_params(labelsize=7)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  Saved hotspot suppression diagnostic: {output_path}")


def compute_blob_temporal_projection(
    movie: np.ndarray,
    blob_center: Tuple[float, float],
    blob_radius: float,
    n_peak_frames: int = 10,
    percentile: float = 95,
) -> np.ndarray:
    """
    Compute a local projection for a specific blob using its peak activity frames.
    
    Instead of using a global max projection (which may be dominated by other
    neurons or hotspots), this extracts the temporal trace at the blob location
    and creates a projection from the frames where THIS blob is most active.
    
    Parameters
    ----------
    movie : np.ndarray
        Movie with shape (T, Y, X)
    blob_center : tuple
        (row, col) center of the blob
    blob_radius : float
        Radius of the blob in pixels
    n_peak_frames : int
        Number of peak frames to use for projection. Default 10.
    percentile : float
        Percentile threshold for selecting "active" frames. Default 95.
        Frames above this percentile of the blob's trace are considered active.
        
    Returns
    -------
    np.ndarray
        2D projection image (same shape as movie frames)
    """
    T, d1, d2 = movie.shape
    y, x = blob_center
    r = int(blob_radius)
    
    # Define a small ROI to extract temporal trace
    y_min = max(0, int(y) - r)
    y_max = min(d1, int(y) + r + 1)
    x_min = max(0, int(x) - r)
    x_max = min(d2, int(x) + r + 1)
    
    # Extract mean trace within the blob region
    blob_trace = movie[:, y_min:y_max, x_min:x_max].mean(axis=(1, 2))
    
    # Find frames where this blob is most active
    threshold = np.percentile(blob_trace, percentile)
    active_mask = blob_trace >= threshold
    
    # If not enough active frames, take top N by intensity
    n_active = active_mask.sum()
    if n_active < n_peak_frames:
        # Get indices of top N frames
        top_indices = np.argsort(blob_trace)[-n_peak_frames:]
        active_mask = np.zeros(T, dtype=bool)
        active_mask[top_indices] = True
    
    # Create projection from active frames only
    active_frames = movie[active_mask]
    
    # Use max projection of active frames
    projection = np.max(active_frames, axis=0)
    
    return projection.astype(np.float32)


def extract_contour_with_temporal_projection(
    movie: np.ndarray,
    blob: 'BlobDetection',
    padding: int = 10,
    morphology_kernel_size: int = 3,
    threshold_method: str = 'otsu',
    min_contour_area: float = 20.0,
    max_contour_area_ratio: float = 5.0,
    n_peak_frames: int = 10,
    peak_percentile: float = 90,
    smooth_sigma: float = 0.0,
) -> Tuple[Optional['ContourInfo'], Dict[str, Any]]:
    """
    Extract contour for a blob using per-blob temporal projection.
    
    This method creates a custom projection for each blob based on the frames
    where that specific blob is most active. This helps when:
    - Different neurons are active at different times
    - Hotspots create bright spots in global max projection
    - The blob of interest is dimmer than surrounding features in max projection
    
    Parameters
    ----------
    movie : np.ndarray
        Full movie with shape (T, Y, X)
    blob : BlobDetection
        Blob to extract contour for
    padding : int
        Extra padding around blob for ROI extraction
    morphology_kernel_size : int
        Size of kernel for morphological opening
    threshold_method : str
        'otsu' or 'triangle' thresholding
    min_contour_area : float
        Minimum valid contour area
    max_contour_area_ratio : float
        Maximum contour area as multiple of expected blob area
    n_peak_frames : int
        Number of peak activity frames to use for projection
    peak_percentile : float
        Percentile threshold for selecting active frames
        
    Returns
    -------
    contour_info : Optional[ContourInfo]
        Extracted contour, or None if extraction failed
    diagnostics : dict
        Detailed diagnostic information
    """
    diagnostics = {
        'blob_center': blob.center,
        'blob_radius': blob.radius,
        'success': False,
        'failure_reason': None,
        'roi_size': None,
        'threshold_value': None,
        'threshold_method_used': threshold_method,
        'n_contours_found': 0,
        'selected_contour_area': None,
        'selected_contour_distance': None,
        'n_peak_frames_used': 0,
        'used_temporal_projection': True,
    }
    
    T, d1, d2 = movie.shape
    y, x = blob.center
    r = blob.radius
    
    # Define ROI around blob
    roi_size = int(r * 2) + padding
    
    y_min = max(0, int(y - roi_size))
    y_max = min(d1, int(y + roi_size))
    x_min = max(0, int(x - roi_size))
    x_max = min(d2, int(x + roi_size))
    
    diagnostics['roi_bounds'] = (y_min, y_max, x_min, x_max)
    diagnostics['roi_size'] = (y_max - y_min, x_max - x_min)
    
    # Extract temporal trace for this blob's region
    blob_trace = movie[:, y_min:y_max, x_min:x_max].mean(axis=(1, 2))
    
    # Find peak frames for this blob
    threshold = np.percentile(blob_trace, peak_percentile)
    active_mask = blob_trace >= threshold
    n_active = active_mask.sum()
    
    if n_active < n_peak_frames:
        # Get indices of top N frames
        top_indices = np.argsort(blob_trace)[-n_peak_frames:]
        active_mask = np.zeros(T, dtype=bool)
        active_mask[top_indices] = True
        n_active = n_peak_frames
    
    diagnostics['n_peak_frames_used'] = int(n_active)
    
    # Create local projection from peak frames only
    active_movie_roi = movie[active_mask, y_min:y_max, x_min:x_max]
    roi = np.max(active_movie_roi, axis=0).astype(np.float32)
    
    # Apply smoothing to suppress hotspots in the local projection
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        roi = gaussian_filter(roi, sigma=smooth_sigma)
    
    if roi.size == 0:
        diagnostics['failure_reason'] = 'empty_roi'
        return None, diagnostics
    
    # Apply Gaussian locality mask to suppress distant bright regions
    # This prevents the contour from expanding into brighter neighboring cells
    roi_h, roi_w = roi.shape
    center_y_local = int(y - y_min)
    center_x_local = int(x - x_min)
    
    # Create distance-based weight mask centered on blob
    yy, xx = np.ogrid[:roi_h, :roi_w]
    dist_from_center = np.sqrt((yy - center_y_local)**2 + (xx - center_x_local)**2)
    
    # Gaussian falloff - pixels beyond ~2*radius get suppressed
    locality_sigma = r * 1.5  # Controls how quickly distant pixels are suppressed
    locality_mask = np.exp(-0.5 * (dist_from_center / locality_sigma)**2)
    
    # Apply mask to ROI before thresholding
    roi_masked = roi * locality_mask
    
    # Convert to uint8 for OpenCV (normalize to 0-255)
    roi_min, roi_max = roi_masked.min(), roi_masked.max()
    if roi_max > roi_min:
        roi_uint8 = ((roi_masked - roi_min) / (roi_max - roi_min) * 255).astype(np.uint8)
    else:
        diagnostics['failure_reason'] = 'no_contrast_in_roi'
        return None, diagnostics
    
    # Also keep unmasked version for intensity measurements
    roi_orig_min, roi_orig_max = roi.min(), roi.max()
    if roi_orig_max > roi_orig_min:
        roi_uint8_orig = ((roi - roi_orig_min) / (roi_orig_max - roi_orig_min) * 255).astype(np.uint8)
    else:
        roi_uint8_orig = roi_uint8
    
    # Apply thresholding
    try:
        if threshold_method == 'otsu':
            thresh_val, roi_thresh = cv2.threshold(
                roi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif threshold_method == 'triangle':
            thresh_val, roi_thresh = cv2.threshold(
                roi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
            )
        else:
            # Fallback to percentile threshold
            thresh_val = np.percentile(roi_uint8, 70)
            roi_thresh = (roi_uint8 > thresh_val).astype(np.uint8) * 255
        
        diagnostics['threshold_value'] = float(thresh_val)
        diagnostics['locality_mask_applied'] = True
        
    except Exception as e:
        diagnostics['failure_reason'] = f'threshold_error: {e}'
        return None, diagnostics
    
    # Morphological cleanup
    kernel = np.ones((morphology_kernel_size, morphology_kernel_size), np.uint8)
    roi_clean = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    diagnostics['n_contours_found'] = len(contours)
    
    if len(contours) == 0:
        diagnostics['failure_reason'] = 'no_contours_found'
        return None, diagnostics
    
    # Find contour closest to blob center (in ROI coordinates)
    center_roi = (int(x - x_min), int(y - y_min))  # (col, row) for OpenCV
    expected_area = np.pi * r * r
    
    best_contour = None
    best_score = float('inf')
    best_stats = {}
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Area filters
        if area < min_contour_area:
            continue
        if area > expected_area * max_contour_area_ratio:
            continue
        
        # Compute centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Distance from blob center
        dist = np.sqrt((cx - center_roi[0])**2 + (cy - center_roi[1])**2)
        
        # Skip if too far from blob center
        if dist > roi_size * 0.8:
            continue
        
        # Score: prefer close to center and similar to expected size
        size_diff = abs(np.log(area / (expected_area + 1)))
        score = dist + size_diff * r
        
        if score < best_score:
            best_score = score
            best_contour = contour
            best_stats = {
                'area': area,
                'centroid_roi': (cy, cx),
                'distance': dist,
            }
    
    if best_contour is None:
        diagnostics['failure_reason'] = 'no_valid_contour_near_center'
        return None, diagnostics
    
    # Offset contour to global coordinates
    offset_contour = best_contour.copy()
    offset_contour[:, 0, 0] += x_min  # x/col offset
    offset_contour[:, 0, 1] += y_min  # y/row offset
    
    # Compute contour properties
    area = cv2.contourArea(offset_contour)
    perimeter = cv2.arcLength(offset_contour, True)
    
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0.0
    
    # Solidity
    hull = cv2.convexHull(offset_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    
    # Centroid in global coordinates
    M = cv2.moments(offset_contour)
    if M["m00"] > 0:
        cx_global = M["m10"] / M["m00"]
        cy_global = M["m01"] / M["m00"]
    else:
        cx_global, cy_global = x, y
    
    # Bounding box
    bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(offset_contour)
    
    # Mean intensity within contour (from the temporal projection)
    full_projection = compute_blob_temporal_projection(
        movie, blob.center, blob.radius, n_peak_frames, peak_percentile
    )
    mask = np.zeros((d1, d2), dtype=np.uint8)
    cv2.drawContours(mask, [offset_contour], -1, 255, -1)
    mean_intensity = float(np.mean(full_projection[mask > 0])) if np.any(mask > 0) else 0.0
    
    diagnostics['success'] = True
    diagnostics['selected_contour_area'] = area
    diagnostics['selected_contour_distance'] = best_stats['distance']
    diagnostics['circularity'] = circularity
    diagnostics['solidity'] = solidity
    
    contour_info = ContourInfo(
        contour=offset_contour,
        center=(cy_global, cx_global),  # (row, col)
        area=area,
        bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
        circularity=circularity,
        solidity=solidity,
        mean_intensity=mean_intensity,
    )
    
    return contour_info, diagnostics


# =============================================================================
# BLOB DETECTION
# =============================================================================

@dataclass
class BlobDetection:
    """Single blob detection result."""
    center: Tuple[float, float]  # (row, col)
    sigma: float
    radius: float  # sigma * sqrt(2)
    intensity: float
    source: str  # 'max' or 'correlation'


def detect_blobs_on_projection(
    image: np.ndarray,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int = 10,
    threshold: float = 0.05,
    min_intensity: float = 0.2,
    border_margin: int = 20,
    local_contrast_threshold: float = 0.6,
    source_name: str = 'unknown',
) -> Tuple[List[BlobDetection], Dict[str, Any]]:
    """
    Detect blobs on a single projection image with filtering.
    
    Returns both the detections and diagnostic information.
    """
    diagnostics = {
        'source': source_name,
        'image_shape': image.shape,
        'params': {
            'min_sigma': min_sigma,
            'max_sigma': max_sigma,
            'threshold': threshold,
            'min_intensity': min_intensity,
            'border_margin': border_margin,
        },
        'raw_detections': 0,
        'rejected': {
            'border': 0,
            'intensity': 0,
            'local_contrast': 0,
            'not_local_max': 0,
        },
        'final_detections': 0,
    }
    
    # Normalize image
    img_norm = image.astype(np.float32)
    img_min, img_max = img_norm.min(), img_norm.max()
    if img_max > img_min:
        img_norm = (img_norm - img_min) / (img_max - img_min)
    else:
        logger.warning(f"  [{source_name}] Image has no contrast")
        return [], diagnostics
    
    # Run LoG blob detection
    logger.info(f"  [{source_name}] Running LoG (sigma={min_sigma:.1f}-{max_sigma:.1f}, thresh={threshold})...")
    
    try:
        blobs = blob_log(
            img_norm,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
        )
    except Exception as e:
        logger.error(f"  [{source_name}] blob_log failed: {e}")
        return [], diagnostics
    
    diagnostics['raw_detections'] = len(blobs)
    logger.info(f"  [{source_name}] Raw detections: {len(blobs)}")
    
    # Filter blobs
    d1, d2 = image.shape
    detections = []
    
    for blob in blobs:
        y, x, sigma = blob
        radius = sigma * np.sqrt(2)
        
        # Border check
        if y < border_margin or y >= d1 - border_margin:
            diagnostics['rejected']['border'] += 1
            continue
        if x < border_margin or x >= d2 - border_margin:
            diagnostics['rejected']['border'] += 1
            continue
        
        # Intensity check
        y_int, x_int = int(round(y)), int(round(x))
        local_intensity = img_norm[y_int, x_int]
        
        if local_intensity < min_intensity:
            diagnostics['rejected']['intensity'] += 1
            continue
        
        # Local contrast check
        ns = max(10, int(radius * 1.5))
        y_lo, y_hi = max(0, y_int - ns), min(d1, y_int + ns + 1)
        x_lo, x_hi = max(0, x_int - ns), min(d2, x_int + ns + 1)
        local_region = img_norm[y_lo:y_hi, x_lo:x_hi]
        
        local_min, local_max = local_region.min(), local_region.max()
        if local_max > local_min:
            local_contrast = (local_intensity - local_min) / (local_max - local_min)
        else:
            local_contrast = 0.5
        
        if local_contrast < local_contrast_threshold:
            diagnostics['rejected']['local_contrast'] += 1
            continue
        
        # Local maximum check
        if local_intensity < local_max * 0.90:
            diagnostics['rejected']['not_local_max'] += 1
            continue
        
        detections.append(BlobDetection(
            center=(y, x),
            sigma=sigma,
            radius=radius,
            intensity=local_intensity,
            source=source_name,
        ))
    
    diagnostics['final_detections'] = len(detections)
    logger.info(f"  [{source_name}] After filtering: {len(detections)} blobs")
    logger.info(f"  [{source_name}] Rejected - border: {diagnostics['rejected']['border']}, "
                f"intensity: {diagnostics['rejected']['intensity']}, "
                f"contrast: {diagnostics['rejected']['local_contrast']}, "
                f"not_max: {diagnostics['rejected']['not_local_max']}")
    
    return detections, diagnostics


def merge_blob_detections(
    blobs_list: List[List[BlobDetection]],
    min_distance: float,
) -> Tuple[List[BlobDetection], Dict[str, Any]]:
    """
    Merge blob detections from multiple sources, removing duplicates.
    
    Blobs within min_distance of each other are considered duplicates;
    the one with higher intensity is kept.
    """
    diagnostics = {
        'total_input': sum(len(b) for b in blobs_list),
        'per_source': {f'source_{i}': len(b) for i, b in enumerate(blobs_list)},
        'duplicates_removed': 0,
        'final_count': 0,
        'multi_source_detections': 0,  # Detected in multiple projections
    }
    
    # Flatten all detections
    all_blobs = []
    for blob_list in blobs_list:
        all_blobs.extend(blob_list)
    
    if len(all_blobs) == 0:
        return [], diagnostics
    
    # Sort by intensity (descending) so we keep the brightest
    all_blobs.sort(key=lambda b: b.intensity, reverse=True)
    
    # Greedy deduplication
    keep = []
    kept_centers = []
    source_counts = {}  # Track which sources each kept blob was found in
    
    for blob in all_blobs:
        cy, cx = blob.center
        
        # Check distance to all kept blobs
        is_duplicate = False
        duplicate_idx = -1
        
        for idx, (ky, kx) in enumerate(kept_centers):
            dist = np.sqrt((cy - ky)**2 + (cx - kx)**2)
            if dist < min_distance:
                is_duplicate = True
                duplicate_idx = idx
                break
        
        if is_duplicate:
            diagnostics['duplicates_removed'] += 1
            # Track that this source also found this blob
            if duplicate_idx in source_counts:
                source_counts[duplicate_idx].add(blob.source)
        else:
            keep.append(blob)
            kept_centers.append(blob.center)
            source_counts[len(keep) - 1] = {blob.source}
    
    # Count multi-source detections
    for sources in source_counts.values():
        if len(sources) > 1:
            diagnostics['multi_source_detections'] += 1
    
    diagnostics['final_count'] = len(keep)
    
    logger.info(f"  Merged {diagnostics['total_input']} blobs -> {len(keep)} unique "
                f"({diagnostics['duplicates_removed']} duplicates, "
                f"{diagnostics['multi_source_detections']} found in multiple projections)")
    
    return keep, diagnostics


# =============================================================================
# CONTOUR EXTRACTION
# =============================================================================

def extract_contour_for_blob(
    image: np.ndarray,
    blob: BlobDetection,
    padding: int = 10,
    morphology_kernel_size: int = 3,
    threshold_method: str = 'adaptive',
    min_contour_area: float = 20.0,
    max_contour_area_ratio: float = 5.0,
    correlation_image: Optional[np.ndarray] = None,
    adaptive_block_size: int = 0,  # 0 = auto-compute from blob radius
    adaptive_C: float = 2.0,
) -> Tuple[Optional[ContourInfo], Dict[str, Any]]:
    """
    Extract contour for a single blob using adaptive thresholding.
    
    Supports multiple thresholding methods:
    - 'adaptive': Gaussian-weighted local adaptive thresholding (best for uneven illumination)
    - 'adaptive_mean': Mean-based adaptive thresholding  
    - 'otsu': Global Otsu thresholding (good for bimodal histograms)
    - 'triangle': Triangle thresholding (good for unimodal with tail)
    - 'correlation': Use local correlation image for segmentation
    - 'combined': Try adaptive first, fall back to correlation if available
    
    Parameters
    ----------
    image : np.ndarray
        Projection image (should be max projection or similar)
    blob : BlobDetection
        Blob to extract contour for
    padding : int
        Extra padding around blob for ROI extraction
    morphology_kernel_size : int
        Size of kernel for morphological opening
    threshold_method : str
        Thresholding method (see above)
    min_contour_area : float
        Minimum valid contour area
    max_contour_area_ratio : float
        Maximum contour area as multiple of expected blob area
    correlation_image : np.ndarray, optional
        Local correlation image for correlation-based segmentation
    adaptive_block_size : int
        Block size for adaptive threshold (0 = auto from blob radius)
    adaptive_C : float
        Constant subtracted from mean/Gaussian weighted mean
        
    Returns
    -------
    contour_info : Optional[ContourInfo]
        Extracted contour, or None if extraction failed
    diagnostics : dict
        Detailed diagnostic information
    """
    diagnostics = {
        'blob_center': blob.center,
        'blob_radius': blob.radius,
        'success': False,
        'failure_reason': None,
        'roi_size': None,
        'threshold_value': None,
        'threshold_method_used': threshold_method,
        'n_contours_found': 0,
        'selected_contour_area': None,
        'selected_contour_distance': None,
    }
    
    y, x = blob.center
    r = blob.radius
    
    # Define ROI around blob
    roi_size = int(r * 2) + padding
    d1, d2 = image.shape
    
    y_min = max(0, int(y - roi_size))
    y_max = min(d1, int(y + roi_size))
    x_min = max(0, int(x - roi_size))
    x_max = min(d2, int(x + roi_size))
    
    diagnostics['roi_bounds'] = (y_min, y_max, x_min, x_max)
    diagnostics['roi_size'] = (y_max - y_min, x_max - x_min)
    
    # Extract ROI
    roi = image[y_min:y_max, x_min:x_max].copy().astype(np.float32)
    
    if roi.size == 0:
        diagnostics['failure_reason'] = 'empty_roi'
        return None, diagnostics
    
    # Apply Gaussian locality mask to suppress distant bright regions
    # This prevents the contour from expanding into brighter neighboring cells
    roi_h, roi_w = roi.shape
    center_y_local = int(y - y_min)
    center_x_local = int(x - x_min)
    
    # Create distance-based weight mask centered on blob
    yy, xx = np.ogrid[:roi_h, :roi_w]
    dist_from_center = np.sqrt((yy - center_y_local)**2 + (xx - center_x_local)**2)
    
    # Gaussian falloff - pixels beyond ~2*radius get suppressed
    locality_sigma = r * 1.5
    locality_mask = np.exp(-0.5 * (dist_from_center / locality_sigma)**2)
    
    # Apply mask to ROI before thresholding
    roi_masked = roi * locality_mask
    
    # Convert to uint8 for OpenCV (normalize to 0-255)
    roi_min, roi_max = roi_masked.min(), roi_masked.max()
    if roi_max > roi_min:
        roi_uint8 = ((roi_masked - roi_min) / (roi_max - roi_min) * 255).astype(np.uint8)
    else:
        diagnostics['failure_reason'] = 'no_contrast_in_roi'
        return None, diagnostics
    
    diagnostics['locality_mask_applied'] = True
    
    # Also extract correlation ROI if available
    corr_roi_uint8 = None
    if correlation_image is not None:
        corr_roi = correlation_image[y_min:y_max, x_min:x_max].copy()
        corr_min, corr_max = corr_roi.min(), corr_roi.max()
        if corr_max > corr_min:
            corr_roi_uint8 = ((corr_roi - corr_min) / (corr_max - corr_min) * 255).astype(np.uint8)
    
    # Compute adaptive block size if auto (only used for adaptive methods)
    if adaptive_block_size == 0:
        # Block size should be odd and roughly 2x the neuron diameter
        block_size = int(r * 2) | 1  # Make odd
        block_size = max(block_size, 11)  # Minimum 11
        block_size = min(block_size, min(roi.shape) - 2)  # Don't exceed ROI
        if block_size % 2 == 0:
            block_size += 1
    else:
        block_size = adaptive_block_size
    
    diagnostics['adaptive_block_size'] = block_size
    
    # Apply thresholding
    roi_thresh = None
    methods_tried = []
    
    try:
        if threshold_method in ['adaptive', 'adaptive_gaussian', 'combined']:
            # Adaptive Gaussian thresholding - best for uneven illumination
            roi_thresh = cv2.adaptiveThreshold(
                roi_uint8, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                adaptive_C
            )
            diagnostics['threshold_value'] = f'adaptive_gaussian_C={adaptive_C}'
            methods_tried.append('adaptive_gaussian')
            
        elif threshold_method == 'adaptive_mean':
            # Adaptive mean thresholding
            roi_thresh = cv2.adaptiveThreshold(
                roi_uint8, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                block_size,
                adaptive_C
            )
            diagnostics['threshold_value'] = f'adaptive_mean_C={adaptive_C}'
            methods_tried.append('adaptive_mean')
            
        elif threshold_method == 'correlation' and corr_roi_uint8 is not None:
            # Use correlation image with Otsu
            thresh_val, roi_thresh = cv2.threshold(
                corr_roi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            diagnostics['threshold_value'] = float(thresh_val)
            methods_tried.append('correlation_otsu')
            
        elif threshold_method == 'otsu':
            thresh_val, roi_thresh = cv2.threshold(
                roi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            diagnostics['threshold_value'] = float(thresh_val)
            methods_tried.append('otsu')
            
        elif threshold_method == 'triangle':
            thresh_val, roi_thresh = cv2.threshold(
                roi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
            )
            diagnostics['threshold_value'] = float(thresh_val)
            methods_tried.append('triangle')
            
        else:
            # Fallback to simple percentile threshold
            thresh_val = np.percentile(roi_uint8, 70)
            roi_thresh = (roi_uint8 > thresh_val).astype(np.uint8) * 255
            diagnostics['threshold_value'] = float(thresh_val)
            methods_tried.append('percentile')
        
        diagnostics['methods_tried'] = methods_tried
        
    except Exception as e:
        diagnostics['failure_reason'] = f'threshold_error: {e}'
        return None, diagnostics
    
    # Morphological cleanup
    kernel = np.ones((morphology_kernel_size, morphology_kernel_size), np.uint8)
    roi_clean = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel)
    
    # For combined method: if adaptive fails, try correlation
    if threshold_method == 'combined' and corr_roi_uint8 is not None:
        # Check if we got a reasonable result
        contours_test, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        center_roi = (int(x - x_min), int(y - y_min))
        expected_area = np.pi * r * r
        
        has_valid_contour = False
        for cnt in contours_test:
            area = cv2.contourArea(cnt)
            if area < min_contour_area or area > expected_area * max_contour_area_ratio:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            dist = np.sqrt((cx - center_roi[0])**2 + (cy - center_roi[1])**2)
            if dist < roi_size * 0.5:
                has_valid_contour = True
                break
        
        if not has_valid_contour:
            # Try correlation-based thresholding as fallback
            thresh_val, corr_thresh = cv2.threshold(
                corr_roi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            roi_clean_corr = cv2.morphologyEx(corr_thresh, cv2.MORPH_OPEN, kernel)
            
            # Check if correlation gives better result
            contours_corr, _ = cv2.findContours(roi_clean_corr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours_corr:
                area = cv2.contourArea(cnt)
                if area < min_contour_area or area > expected_area * max_contour_area_ratio:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                dist = np.sqrt((cx - center_roi[0])**2 + (cy - center_roi[1])**2)
                if dist < roi_size * 0.5:
                    # Correlation worked better, use it
                    roi_clean = roi_clean_corr
                    diagnostics['threshold_value'] = f'correlation_fallback_{thresh_val}'
                    methods_tried.append('correlation_fallback')
                    break
    
    # Find contours
    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    diagnostics['n_contours_found'] = len(contours)
    
    if len(contours) == 0:
        diagnostics['failure_reason'] = 'no_contours_found'
        return None, diagnostics
    
    # Find contour closest to blob center (in ROI coordinates)
    center_roi = (int(x - x_min), int(y - y_min))  # (col, row) for OpenCV
    expected_area = np.pi * r * r
    
    best_contour = None
    best_score = float('inf')
    best_stats = {}
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Area filters
        if area < min_contour_area:
            continue
        if area > expected_area * max_contour_area_ratio:
            continue
        
        # Compute centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Distance from blob center
        dist = np.sqrt((cx - center_roi[0])**2 + (cy - center_roi[1])**2)
        
        # Skip if too far from blob center
        if dist > roi_size * 0.8:
            continue
        
        # Score: prefer close to center and similar to expected size
        size_diff = abs(np.log(area / (expected_area + 1)))
        score = dist + size_diff * r  # Weight size difference by radius
        
        if score < best_score:
            best_score = score
            best_contour = contour
            best_stats = {
                'area': area,
                'centroid_roi': (cy, cx),  # (row, col)
                'distance': dist,
            }
    
    if best_contour is None:
        diagnostics['failure_reason'] = 'no_valid_contour_near_center'
        return None, diagnostics
    
    # Offset contour to global coordinates
    offset_contour = best_contour.copy()
    offset_contour[:, 0, 0] += x_min  # x/col offset
    offset_contour[:, 0, 1] += y_min  # y/row offset
    
    # Compute contour properties
    area = cv2.contourArea(offset_contour)
    perimeter = cv2.arcLength(offset_contour, True)
    
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0.0
    
    # Solidity
    hull = cv2.convexHull(offset_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    
    # Centroid in global coordinates
    M = cv2.moments(offset_contour)
    if M["m00"] > 0:
        cx_global = M["m10"] / M["m00"]
        cy_global = M["m01"] / M["m00"]
    else:
        cx_global, cy_global = x, y
    
    # Bounding box
    bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(offset_contour)
    
    # Mean intensity within contour
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [offset_contour], -1, 255, -1)
    mean_intensity = float(np.mean(image[mask > 0])) if np.any(mask > 0) else 0.0
    
    diagnostics['success'] = True
    diagnostics['selected_contour_area'] = area
    diagnostics['selected_contour_distance'] = best_stats['distance']
    diagnostics['circularity'] = circularity
    diagnostics['solidity'] = solidity
    
    contour_info = ContourInfo(
        contour=offset_contour,
        center=(cy_global, cx_global),  # (row, col)
        area=area,
        bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
        circularity=circularity,
        solidity=solidity,
        mean_intensity=mean_intensity,
    )
    
    return contour_info, diagnostics


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================

def detect_seeds_with_contours(
    movie: np.ndarray,
    min_radius: float = 10,
    max_radius: float = 30,
    intensity_threshold: float = 0.20,
    correlation_threshold: float = 0.15,
    border_margin: int = 20,
    max_seeds: int = 2000,
    contour_method: str = 'otsu',
    use_correlation: bool = True,
    use_mean: bool = False,
    use_std: bool = False,
    use_percentile: bool = False,
    smooth_sigma: float = 0.0,
    use_temporal_projection: bool = False,
    n_peak_frames: int = 10,
    peak_percentile: float = 90,
    diagnostics_dir: Optional[str] = None,
    precomputed_projections: Optional['ProjectionSet'] = None,
) -> ContourSeedResult:
    """
    Main entry point for contour-based seed detection.
    
    Parameters
    ----------
    movie : np.ndarray
        Movie with shape (T, Y, X)
    min_radius : float
        Minimum expected neuron radius in pixels
    max_radius : float
        Maximum expected neuron radius in pixels
    intensity_threshold : float
        Minimum normalized intensity for seed detection on max projection
    correlation_threshold : float
        Minimum normalized correlation for seed detection
    border_margin : int
        Pixels to exclude from image border
    max_seeds : int
        Maximum number of seeds to return
    contour_method : str
        Thresholding method for contour extraction:
        - 'otsu': Global Otsu thresholding (recommended)
        - 'triangle': Triangle thresholding
        - 'adaptive': Adaptive Gaussian thresholding
        - 'combined': Adaptive + correlation fallback
    use_correlation : bool
        Whether to also detect on correlation image
    use_mean : bool
        Whether to also detect on mean projection. Useful for neurons that
        are consistently active but not necessarily bright at peak; can
        recover cells missed by max due to low peak-to-baseline ratio.
    use_std : bool
        Whether to also detect on std projection. Highlights pixels with
        high temporal variance — tends to find highly active neurons and
        can complement max/correlation on data with strong neuropil background.
    use_percentile : bool
        Whether to use 95th percentile instead of max for contour extraction
    smooth_sigma : float
        Gaussian smoothing sigma to apply before projections.
        Suppresses small bright features (hotspots/puncta) while preserving
        larger structures (cell bodies). Recommended: 3-5 for hotspot suppression.
        Set to 0 to disable smoothing.
    use_temporal_projection : bool
        If True, use per-blob temporal projection for contour extraction.
        For each blob candidate, finds the frames where that blob is most active
        and creates a local projection from those frames. This helps when
        hotspots or other neurons dominate the global max projection.
    n_peak_frames : int
        Number of peak activity frames to use per blob (for temporal projection).
    peak_percentile : float
        Percentile threshold for selecting "active" frames (for temporal projection).
    diagnostics_dir : str, optional
        Directory to save diagnostic outputs
        
    Returns
    -------
    ContourSeedResult
        Detection results with contour information
    """
    start_time = time.time()
    T, d1, d2 = movie.shape
    dims = (d1, d2)
    
    logger.info("="*70)
    logger.info("CONTOUR-BASED SEED DETECTION")
    logger.info("="*70)
    logger.info(f"Movie shape: {movie.shape}")
    logger.info(f"Radius range: {min_radius:.1f} - {max_radius:.1f} pixels")
    logger.info(f"Max seeds: {max_seeds}")
    
    # Initialize master diagnostics
    master_diagnostics = {
        'movie_shape': movie.shape,
        'parameters': {
            'min_radius': min_radius,
            'max_radius': max_radius,
            'intensity_threshold': intensity_threshold,
            'correlation_threshold': correlation_threshold,
            'border_margin': border_margin,
            'max_seeds': max_seeds,
            'contour_method': contour_method,
        },
        'timing': {},
        'projections': {},
        'blob_detection': {},
        'contour_extraction': {
            'total_attempts': 0,
            'successes': 0,
            'failures_by_reason': {},
        },
    }
    
    # =========================================================================
    # STEP 1: Compute projections (or reuse precomputed)
    # =========================================================================
    logger.info("\n" + "-"*50)
    logger.info("STEP 1: Computing projections")
    logger.info("-"*50)

    proj_start = time.time()
    if precomputed_projections is not None:
        logger.info("  Reusing precomputed projections")
        projections = precomputed_projections
    else:
        if smooth_sigma > 0:
            logger.info(f"  Hotspot suppression: Gaussian smoothing with sigma={smooth_sigma}")
        projections = compute_projections_extended(
            movie,
            compute_correlation=use_correlation,
            smooth_sigma=smooth_sigma,
        )
    master_diagnostics['timing']['projections'] = time.time() - proj_start
    master_diagnostics['smooth_sigma'] = smooth_sigma
    
    master_diagnostics['projections'] = {
        'max_range': (float(projections.max_proj.min()), float(projections.max_proj.max())),
        'mean_range': (float(projections.mean_proj.min()), float(projections.mean_proj.max())),
        'std_range': (float(projections.std_proj.min()), float(projections.std_proj.max())),
        'correlation_range': (float(projections.correlation.min()), float(projections.correlation.max())),
    }
    
    # =========================================================================
    # STEP 2: Blob detection on multiple projections
    # =========================================================================
    logger.info("\n" + "-"*50)
    logger.info("STEP 2: Blob detection")
    logger.info("-"*50)
    
    blob_start = time.time()
    
    # Convert radius to sigma
    min_sigma = min_radius / np.sqrt(2)
    max_sigma = max_radius / np.sqrt(2)
    
    all_blob_lists = []
    
    # Detect on max projection
    blobs_max, diag_max = detect_blobs_on_projection(
        projections.max_norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=0.05,
        min_intensity=intensity_threshold,
        border_margin=border_margin,
        local_contrast_threshold=0.6,
        source_name='max',
    )
    all_blob_lists.append(blobs_max)
    master_diagnostics['blob_detection']['max_projection'] = diag_max
    
    # Detect on correlation image
    if use_correlation:
        blobs_corr, diag_corr = detect_blobs_on_projection(
            projections.correlation_norm,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=0.03,  # Lower threshold for correlation
            min_intensity=correlation_threshold,
            border_margin=border_margin,
            local_contrast_threshold=0.5,
            source_name='correlation',
        )
        all_blob_lists.append(blobs_corr)
        master_diagnostics['blob_detection']['correlation'] = diag_corr

    # Detect on mean projection
    # Mean highlights neurons that are persistently active rather than
    # transiently bright; uses the same thresholds as max since the images
    # are independently normalised to [0,1] before detection.
    if use_mean:
        blobs_mean, diag_mean = detect_blobs_on_projection(
            projections.mean_norm,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=0.05,
            min_intensity=intensity_threshold,
            border_margin=border_margin,
            local_contrast_threshold=0.6,
            source_name='mean',
        )
        all_blob_lists.append(blobs_mean)
        master_diagnostics['blob_detection']['mean_projection'] = diag_mean

    # Detect on std projection
    # Std highlights pixels with high temporal variance — captures highly
    # active neurons that may be suppressed in the mean by long silent periods.
    if use_std:
        blobs_std, diag_std = detect_blobs_on_projection(
            projections.std_norm,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=0.05,
            min_intensity=intensity_threshold,
            border_margin=border_margin,
            local_contrast_threshold=0.6,
            source_name='std',
        )
        all_blob_lists.append(blobs_std)
        master_diagnostics['blob_detection']['std_projection'] = diag_std

    # Merge detections - use 1.5x min_radius to aggressively remove duplicates
    # This helps when multiple blobs are detected on the same cell
    merged_blobs, merge_diag = merge_blob_detections(all_blob_lists, min_distance=min_radius * 1.5)
    master_diagnostics['blob_detection']['merge'] = merge_diag
    master_diagnostics['timing']['blob_detection'] = time.time() - blob_start
    
    if len(merged_blobs) == 0:
        logger.warning("No blobs detected!")
        return ContourSeedResult(
            centers=np.array([]).reshape(0, 2),
            radii=np.array([]),
            intensities=np.array([]),
            confidence=np.array([]),
            contours=[],
            contour_success=np.array([], dtype=bool),
            source_projection=np.array([], dtype='U10'),
            diagnostics=master_diagnostics,
        )
    
    # Limit number of blobs before contour extraction
    if len(merged_blobs) > max_seeds:
        logger.info(f"Limiting from {len(merged_blobs)} to {max_seeds} blobs (by intensity)")
        merged_blobs.sort(key=lambda b: b.intensity, reverse=True)
        merged_blobs = merged_blobs[:max_seeds]
    
    # =========================================================================
    # STEP 3: Contour extraction for each blob
    # =========================================================================
    logger.info("\n" + "-"*50)
    logger.info(f"STEP 3: Contour extraction ({len(merged_blobs)} blobs)")
    logger.info("-"*50)
    
    contour_start = time.time()
    
    # Choose image for contour extraction (only used if not using temporal projection)
    if use_percentile:
        contour_image = projections.percentile_95
        logger.info("  Using 95th percentile projection for contours")
    else:
        contour_image = projections.max_proj
        logger.info("  Using max projection for contours")
    
    # Extract contours
    contours = []
    contour_success = []
    contour_diagnostics_list = []
    
    # Get correlation image for combined thresholding
    correlation_for_contours = projections.correlation if use_correlation else None
    
    # Log the method being used
    logger.info(f"  Contour method: {contour_method}")
    if use_temporal_projection:
        logger.info(f"  Using PER-BLOB TEMPORAL PROJECTION (n_peak_frames={n_peak_frames}, percentile={peak_percentile})")
    if contour_method == 'combined' and correlation_for_contours is not None:
        logger.info("  Using combined adaptive + correlation fallback")
    
    for i, blob in enumerate(merged_blobs):
        if use_temporal_projection:
            # Use per-blob temporal projection - extracts contour from frames
            # where THIS specific blob is most active
            contour_info, contour_diag = extract_contour_with_temporal_projection(
                movie,
                blob,
                padding=int(min_radius),
                threshold_method=contour_method,
                min_contour_area=min_radius * min_radius * 0.5,
                max_contour_area_ratio=6.0,
                n_peak_frames=n_peak_frames,
                peak_percentile=peak_percentile,
                smooth_sigma=smooth_sigma,
            )
        else:
            # Use global projection image
            contour_info, contour_diag = extract_contour_for_blob(
                contour_image,
                blob,
                padding=int(min_radius),
                threshold_method=contour_method,
                min_contour_area=min_radius * min_radius * 0.5,
                max_contour_area_ratio=6.0,
                correlation_image=correlation_for_contours,
            )
        
        contours.append(contour_info)
        contour_success.append(contour_info is not None)
        contour_diagnostics_list.append(contour_diag)
        
        # Track failure reasons
        if contour_info is None:
            reason = contour_diag.get('failure_reason', 'unknown')
            master_diagnostics['contour_extraction']['failures_by_reason'][reason] = \
                master_diagnostics['contour_extraction']['failures_by_reason'].get(reason, 0) + 1
        
        # Progress logging
        if (i + 1) % 100 == 0:
            success_so_far = sum(contour_success)
            logger.info(f"  Processed {i+1}/{len(merged_blobs)} blobs "
                       f"({success_so_far} contours extracted)")
    
    contour_success = np.array(contour_success)
    master_diagnostics['contour_extraction']['total_attempts'] = len(merged_blobs)
    master_diagnostics['contour_extraction']['successes'] = int(contour_success.sum())
    master_diagnostics['contour_extraction']['use_temporal_projection'] = use_temporal_projection
    master_diagnostics['timing']['contour_extraction'] = time.time() - contour_start
    
    logger.info(f"  Contour extraction complete: {contour_success.sum()}/{len(merged_blobs)} successful "
                f"({100*contour_success.mean():.1f}%)")
    
    # =========================================================================
    # STEP 4: Build output arrays
    # =========================================================================
    logger.info("\n" + "-"*50)
    logger.info("STEP 4: Building output")
    logger.info("-"*50)
    
    n_seeds = len(merged_blobs)
    
    centers = np.array([b.center for b in merged_blobs])
    radii = np.array([b.radius for b in merged_blobs])
    intensities = np.array([b.intensity for b in merged_blobs])
    sources = np.array([b.source for b in merged_blobs])
    
    # Update centers for successful contours (use contour centroid)
    for i, (contour_info, success) in enumerate(zip(contours, contour_success)):
        if success and contour_info is not None:
            centers[i] = contour_info.center
            # Update radius based on contour area
            radii[i] = np.sqrt(contour_info.area / np.pi)
    
    # Compute confidence scores
    # Base confidence from intensity
    int_min, int_max = intensities.min(), intensities.max()
    if int_max > int_min:
        intensity_score = (intensities - int_min) / (int_max - int_min)
    else:
        intensity_score = np.ones(n_seeds) * 0.5
    
    # Boost for successful contour extraction
    contour_boost = np.where(contour_success, 1.0, 0.7)
    
    # Boost for multi-source detection (found in both max and correlation)
    # For now, simple source-based boost
    source_boost = np.where(sources == 'max', 1.0, 0.9)
    
    # Contour quality scores (for successful extractions)
    contour_quality = np.ones(n_seeds) * 0.5
    for i, (contour_info, success) in enumerate(zip(contours, contour_success)):
        if success and contour_info is not None:
            # Higher circularity and solidity = higher quality
            contour_quality[i] = (contour_info.circularity + contour_info.solidity) / 2
    
    # Detect and EXCLUDE boundary-touching contours:
    # Any contour whose points include x==0, x==d2-1, y==0, or y==d1-1
    # These are real cells at the field-of-view edge with incomplete footprints
    # that would produce unreliable traces.  Excluded here so they never
    # receive footprints, traces, or deconvolution.
    boundary_touching = np.zeros(n_seeds, dtype=bool)
    for i, (contour_info, success) in enumerate(zip(contours, contour_success)):
        if success and contour_info is not None:
            pts = contour_info.contour.squeeze()
            if len(pts.shape) == 2:
                xs = pts[:, 0]
                ys = pts[:, 1]
                if (xs.min() == 0 or xs.max() == d2 - 1 or
                        ys.min() == 0 or ys.max() == d1 - 1):
                    boundary_touching[i] = True
    
    n_boundary = int(boundary_touching.sum())
    keep = ~boundary_touching
    if n_boundary > 0:
        logger.info(f"  Excluding {n_boundary} boundary-touching contours")
        centers = centers[keep]
        radii = radii[keep]
        intensities = intensities[keep]
        contours = [c for c, k in zip(contours, keep) if k]
        contour_success = contour_success[keep]
        sources = sources[keep]
        intensity_score = intensity_score[keep]
        contour_quality = contour_quality[keep]
        contour_boost = contour_boost[keep]
        source_boost = source_boost[keep]
        n_seeds = int(keep.sum())
    
    # Combined confidence (retained for diagnostics/gallery, not used for selection)
    confidence = (intensity_score * 0.4 + contour_quality * 0.3 + 
                  contour_boost * 0.2 + source_boost * 0.1)
    confidence = np.clip(confidence, 0, 1)
    
    # Timing
    total_time = time.time() - start_time
    master_diagnostics['timing']['total'] = total_time
    
    logger.info(f"\nDetection complete:")
    logger.info(f"  Total seeds: {n_seeds}")
    logger.info(f"  With contours: {contour_success.sum()} ({100*contour_success.mean():.1f}%)")
    logger.info(f"  Boundary excluded: {n_boundary}")
    logger.info(f"  Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    logger.info(f"  Total time: {total_time:.2f}s")
    
    # Save diagnostics if requested
    if diagnostics_dir is not None:
        os.makedirs(diagnostics_dir, exist_ok=True)
        
        # Save master diagnostics
        diag_path = os.path.join(diagnostics_dir, 'contour_detection_diagnostics.json')
        with open(diag_path, 'w') as f:
            json.dump(master_diagnostics, f, indent=2, default=str)
        logger.info(f"  Saved diagnostics to {diag_path}")
        
        # Save per-blob diagnostics
        blob_diag_path = os.path.join(diagnostics_dir, 'per_blob_diagnostics.json')
        with open(blob_diag_path, 'w') as f:
            json.dump(contour_diagnostics_list, f, indent=2, default=str)
    
    return ContourSeedResult(
        centers=centers,
        radii=radii,
        intensities=intensities,
        confidence=confidence,
        contours=contours,
        contour_success=contour_success,
        source_projection=sources,
        boundary_touching=np.zeros(n_seeds, dtype=bool),  # all boundary seeds removed
        diagnostics=master_diagnostics,
    )


# =============================================================================
# CONVERSION TO SPATIAL FOOTPRINTS
# =============================================================================

def generate_circular_footprint(
    center: Tuple[float, float],
    radius: float,
    dims: Tuple[int, int],
    method: str = 'gaussian',
) -> np.ndarray:
    """
    Generate a circular spatial footprint for a single seed.

    Used as a fallback when contour extraction (Otsu thresholding) fails
    for a particular seed — e.g. low contrast, overlapping neurons, or
    edge artefacts.

    Parameters
    ----------
    center : (float, float)
        (row, col) pixel coordinates.
    radius : float
        Estimated neuron radius in pixels.
    dims : (int, int)
        (d1, d2) image dimensions.
    method : str
        'gaussian' (smooth, weighted) or 'disk' (binary circle).

    Returns
    -------
    footprint : ndarray (d1, d2)
        Unnormalised spatial footprint.
    """
    d1, d2 = dims
    cy, cx = center
    yy, xx = np.ogrid[:d1, :d2]

    if method == 'gaussian':
        sigma = radius / 2
        footprint = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))
        footprint[footprint < 0.01] = 0
    elif method == 'disk':
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        footprint = (dist <= radius).astype(float)
    else:
        raise ValueError(f"Unknown method '{method}'; expected 'gaussian' or 'disk'")

    return footprint


def contours_to_spatial_footprints(
    seeds: ContourSeedResult,
    dims: Tuple[int, int],
    contour_fallback: bool = True,
    fallback_method: str = 'gaussian',
    normalize: bool = True,
    # Backwards compatibility
    fallback_type: Optional[str] = None,
) -> csc_matrix:
    """
    Convert contour seeds to spatial footprints for trace extraction.

    For seeds with successful contour extraction, uses the contour mask.
    For seeds where Otsu failed, behaviour depends on ``contour_fallback``:

    - ``True``: generate a circular footprint via
      :func:`generate_circular_footprint` (keeps all seeds).
    - ``False``: drop the seed entirely (only contour-verified neurons
      are kept).

    Parameters
    ----------
    seeds : ContourSeedResult
        Detection results with contours.
    dims : (int, int)
        (d1, d2) image dimensions.
    contour_fallback : bool
        Whether to generate circular footprints for failed contours.
        If False, seeds without contours are excluded.
    fallback_method : str
        'gaussian' or 'disk' — passed to ``generate_circular_footprint``.
    normalize : bool
        L1-normalise each footprint.

    Returns
    -------
    A_init : csc_matrix
        Sparse matrix (d1*d2, N) of spatial footprints.
        N equals ``seeds.n_seeds`` when ``contour_fallback=True``,
        or ``seeds.n_contours`` when ``contour_fallback=False``.
    """
    # Backwards compatibility: old callers may pass fallback_type='gaussian'
    if fallback_type is not None:
        contour_fallback = True
        fallback_method = fallback_type

    d1, d2 = dims
    n_pixels = d1 * d2
    n_seeds = seeds.n_seeds

    if n_seeds == 0:
        return csc_matrix((n_pixels, 0))

    n_failed = n_seeds - seeds.n_contours
    logger.info(f"Creating spatial footprints for {n_seeds} seeds...")
    logger.info(f"  With contours: {seeds.n_contours}")
    if contour_fallback and n_failed > 0:
        logger.info(f"  Circular fallback ({fallback_method}): {n_failed}")
    elif n_failed > 0:
        logger.info(f"  Dropping {n_failed} seeds without contours")

    footprints = []
    n_contour_used = 0
    n_fallback_used = 0
    n_dropped = 0

    for i in range(n_seeds):
        if seeds.contour_success[i] and seeds.contours[i] is not None:
            contour_info = seeds.contours[i]
            mask = contour_info.to_mask((d1, d2))
            footprint = mask.astype(float) / 255.0
            n_contour_used += 1
        elif contour_fallback:
            footprint = generate_circular_footprint(
                center=seeds.centers[i],
                radius=seeds.radii[i],
                dims=dims,
                method=fallback_method,
            )
            n_fallback_used += 1
        else:
            n_dropped += 1
            continue

        if normalize and footprint.sum() > 0:
            footprint /= footprint.sum()

        footprints.append(footprint.flatten())

    n_kept = len(footprints)
    A = lil_matrix((n_pixels, n_kept))
    for i, fp in enumerate(footprints):
        A[:, i] = fp[:, np.newaxis]

    logger.info(f"  Footprints: {n_contour_used} contour, "
                f"{n_fallback_used} fallback, {n_dropped} dropped "
                f"→ {n_kept} total")

    return csc_matrix(A)


# =============================================================================
# VISUALIZATION / DIAGNOSTICS
# =============================================================================

def visualize_contour_detection_detailed(
    projections: ProjectionSet,
    seeds: ContourSeedResult,
    output_dir: str,
    movie: Optional[np.ndarray] = None,
    n_zoom_regions: int = 4,
    zoom_size: int = 150,
    max_seeds_to_show: int = 500,
) -> None:
    """
    Create comprehensive multi-panel visualization for contour detection evaluation.
    
    Generates multiple output files:
    1. Overview figure (full FOV with different overlays)
    2. Zoom panel figure (multiple zoomed regions)
    3. Individual ROI gallery
    
    Parameters
    ----------
    projections : ProjectionSet
        Projection images
    seeds : ContourSeedResult
        Detection results
    output_dir : str
        Directory to save visualization files
    movie : np.ndarray, optional
        Original movie for additional visualizations
    n_zoom_regions : int
        Number of zoom regions to show
    zoom_size : int
        Size of each zoom region in pixels
    max_seeds_to_show : int
        Maximum number of seeds to display
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================================================
    # FIGURE 1: Full FOV Overview with Multiple Overlays
    # ==========================================================================
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
    fig1.suptitle(f'Contour Detection Overview - {seeds.n_seeds} seeds, {seeds.n_contours} contours', 
                  fontsize=14, fontweight='bold')
    
    # Normalize projections for display
    def normalize(img):
        vmin, vmax = np.percentile(img, [1, 99])
        return np.clip((img - vmin) / (vmax - vmin + 1e-10), 0, 1)
    
    max_norm = normalize(projections.max_proj)
    corr_norm = normalize(projections.correlation)
    std_norm = normalize(projections.std_proj)
    
    # Row 1: Base projections
    # Max projection with seed circles
    ax = axes1[0, 0]
    ax.imshow(max_norm, cmap='gray')
    for i in range(min(seeds.n_seeds, max_seeds_to_show)):
        y, x = seeds.centers[i]
        r = seeds.radii[i]
        color = 'lime' if seeds.contour_success[i] else 'red'
        circle = Circle((x, y), r, color=color, fill=False, linewidth=0.5, alpha=0.6)
        ax.add_patch(circle)
    ax.set_title(f'Max Projection + Seeds\n(green=contour, red=fallback)', fontsize=10)
    ax.axis('off')
    
    # Correlation with seed circles
    ax = axes1[0, 1]
    ax.imshow(corr_norm, cmap='gray')
    for i in range(min(seeds.n_seeds, max_seeds_to_show)):
        y, x = seeds.centers[i]
        r = seeds.radii[i]
        color = 'cyan' if seeds.contour_success[i] else 'red'
        circle = Circle((x, y), r, color=color, fill=False, linewidth=0.5, alpha=0.6)
        ax.add_patch(circle)
    ax.set_title('Correlation Image + Seeds', fontsize=10)
    ax.axis('off')
    
    # Std projection with seed circles
    ax = axes1[0, 2]
    ax.imshow(std_norm, cmap='gray')
    for i in range(min(seeds.n_seeds, max_seeds_to_show)):
        y, x = seeds.centers[i]
        r = seeds.radii[i]
        color = 'yellow' if seeds.contour_success[i] else 'red'
        circle = Circle((x, y), r, color=color, fill=False, linewidth=0.5, alpha=0.6)
        ax.add_patch(circle)
    ax.set_title('Std Projection + Seeds', fontsize=10)
    ax.axis('off')
    
    # Row 2: Contour overlays
    # Contours on max projection
    ax = axes1[1, 0]
    ax.imshow(max_norm, cmap='gray')
    for i in range(min(seeds.n_seeds, max_seeds_to_show)):
        if seeds.contour_success[i] and seeds.contours[i] is not None:
            contour = seeds.contours[i].contour
            pts = contour.squeeze()
            if len(pts.shape) == 2 and pts.shape[0] > 2:
                pts_closed = np.vstack([pts, pts[0]])
                ax.plot(pts_closed[:, 0], pts_closed[:, 1], 'c-', linewidth=0.5, alpha=0.7)
    ax.set_title(f'Contours on Max ({seeds.n_contours} shown)', fontsize=10)
    ax.axis('off')
    
    # Contours on correlation
    ax = axes1[1, 1]
    ax.imshow(corr_norm, cmap='gray')
    for i in range(min(seeds.n_seeds, max_seeds_to_show)):
        if seeds.contour_success[i] and seeds.contours[i] is not None:
            contour = seeds.contours[i].contour
            pts = contour.squeeze()
            if len(pts.shape) == 2 and pts.shape[0] > 2:
                pts_closed = np.vstack([pts, pts[0]])
                ax.plot(pts_closed[:, 0], pts_closed[:, 1], 'lime', linewidth=0.5, alpha=0.7)
    ax.set_title('Contours on Correlation', fontsize=10)
    ax.axis('off')
    
    # Contours on std projection
    ax = axes1[1, 2]
    ax.imshow(std_norm, cmap='gray')
    for i in range(min(seeds.n_seeds, max_seeds_to_show)):
        if seeds.contour_success[i] and seeds.contours[i] is not None:
            contour = seeds.contours[i].contour
            pts = contour.squeeze()
            if len(pts.shape) == 2 and pts.shape[0] > 2:
                pts_closed = np.vstack([pts, pts[0]])
                ax.plot(pts_closed[:, 0], pts_closed[:, 1], 'orange', linewidth=0.5, alpha=0.7)
    ax.set_title('Contours on Std', fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'detection_overview.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # ==========================================================================
    # FIGURE 2: Zoom Panels
    # ==========================================================================
    # Select zoom regions - spread across the image with good seed density
    d1, d2 = projections.max_proj.shape
    
    # Find regions with seeds
    zoom_centers = []
    if seeds.n_seeds > 0:
        # Divide image into grid and pick regions with seeds
        grid_size = max(d1, d2) // 3
        for gy in range(3):
            for gx in range(3):
                y_center = int((gy + 0.5) * grid_size)
                x_center = int((gx + 0.5) * grid_size)
                y_center = min(max(y_center, zoom_size//2), d1 - zoom_size//2)
                x_center = min(max(x_center, zoom_size//2), d2 - zoom_size//2)
                
                # Count seeds in this region
                y_min, y_max = y_center - zoom_size//2, y_center + zoom_size//2
                x_min, x_max = x_center - zoom_size//2, x_center + zoom_size//2
                
                n_seeds_in_region = sum(
                    1 for i in range(seeds.n_seeds)
                    if y_min <= seeds.centers[i, 0] < y_max and x_min <= seeds.centers[i, 1] < x_max
                )
                
                if n_seeds_in_region > 0:
                    zoom_centers.append((y_center, x_center, n_seeds_in_region))
        
        # Sort by seed count and take top n_zoom_regions
        zoom_centers.sort(key=lambda x: x[2], reverse=True)
        zoom_centers = zoom_centers[:n_zoom_regions]
    
    if len(zoom_centers) == 0:
        # Fallback: just use center regions
        zoom_centers = [
            (d1//4, d2//4, 0),
            (d1//4, 3*d2//4, 0),
            (3*d1//4, d2//4, 0),
            (3*d1//4, 3*d2//4, 0),
        ][:n_zoom_regions]
    
    # Create zoom figure
    n_cols = min(n_zoom_regions, 4)
    n_rows = (len(zoom_centers) + n_cols - 1) // n_cols
    n_rows = max(n_rows, 1) * 3  # 3 rows per zoom region (max, corr, contours)
    
    fig2, axes2 = plt.subplots(3, len(zoom_centers), figsize=(5*len(zoom_centers), 15))
    if len(zoom_centers) == 1:
        axes2 = axes2.reshape(-1, 1)
    
    fig2.suptitle('Zoom Regions - Detailed View', fontsize=14, fontweight='bold')
    
    for zi, (yc, xc, n_local) in enumerate(zoom_centers):
        y_min = max(0, yc - zoom_size//2)
        y_max = min(d1, yc + zoom_size//2)
        x_min = max(0, xc - zoom_size//2)
        x_max = min(d2, xc + zoom_size//2)
        
        # Zoom on max projection
        ax = axes2[0, zi]
        zoom_max = max_norm[y_min:y_max, x_min:x_max]
        ax.imshow(zoom_max, cmap='gray', extent=[x_min, x_max, y_max, y_min])
        
        # Draw seeds in this region
        for i in range(seeds.n_seeds):
            y, x = seeds.centers[i]
            if y_min <= y < y_max and x_min <= x < x_max:
                r = seeds.radii[i]
                color = 'lime' if seeds.contour_success[i] else 'red'
                circle = Circle((x, y), r, color=color, fill=False, linewidth=1.5, alpha=0.8)
                ax.add_patch(circle)
        ax.set_title(f'Region {zi+1}: Max + Seeds', fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        
        # Zoom on correlation
        ax = axes2[1, zi]
        zoom_corr = corr_norm[y_min:y_max, x_min:x_max]
        ax.imshow(zoom_corr, cmap='gray', extent=[x_min, x_max, y_max, y_min])
        
        for i in range(seeds.n_seeds):
            y, x = seeds.centers[i]
            if y_min <= y < y_max and x_min <= x < x_max:
                r = seeds.radii[i]
                color = 'cyan' if seeds.contour_success[i] else 'red'
                circle = Circle((x, y), r, color=color, fill=False, linewidth=1.5, alpha=0.8)
                ax.add_patch(circle)
        ax.set_title(f'Region {zi+1}: Correlation + Seeds', fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        
        # Zoom with contours
        ax = axes2[2, zi]
        ax.imshow(zoom_max, cmap='gray', extent=[x_min, x_max, y_max, y_min])
        
        n_contours_in_region = 0
        for i in range(seeds.n_seeds):
            y, x = seeds.centers[i]
            if y_min <= y < y_max and x_min <= x < x_max:
                if seeds.contour_success[i] and seeds.contours[i] is not None:
                    contour = seeds.contours[i].contour
                    pts = contour.squeeze()
                    if len(pts.shape) == 2 and pts.shape[0] > 2:
                        pts_closed = np.vstack([pts, pts[0]])
                        ax.plot(pts_closed[:, 0], pts_closed[:, 1], 'cyan', linewidth=1.5, alpha=0.9)
                        n_contours_in_region += 1
                else:
                    # Draw fallback circle
                    r = seeds.radii[i]
                    circle = Circle((x, y), r, color='red', fill=False, linewidth=1.5, 
                                   linestyle='--', alpha=0.8)
                    ax.add_patch(circle)
        
        ax.set_title(f'Region {zi+1}: Contours ({n_contours_in_region})', fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'detection_zoom_panels.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # ==========================================================================
    # FIGURE 3: Statistics Summary
    # ==========================================================================
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
    fig3.suptitle('Detection Statistics', fontsize=14, fontweight='bold')
    
    # Confidence histogram
    ax = axes3[0, 0]
    if seeds.n_seeds > 0:
        conf_success = seeds.confidence[seeds.contour_success]
        conf_fail = seeds.confidence[~seeds.contour_success]
        bins = np.linspace(0, 1, 21)
        if len(conf_success) > 0:
            ax.hist(conf_success, bins=bins, alpha=0.7, label=f'Contour ({len(conf_success)})',
                   color='forestgreen', edgecolor='black')
        if len(conf_fail) > 0:
            ax.hist(conf_fail, bins=bins, alpha=0.7, label=f'Fallback ({len(conf_fail)})',
                   color='coral', edgecolor='black')
        ax.axvline(np.median(seeds.confidence), color='black', linestyle='--',
                  label=f'Median: {np.median(seeds.confidence):.2f}')
        ax.legend(fontsize=9)
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    
    # Radius histogram
    ax = axes3[0, 1]
    if seeds.n_seeds > 0:
        radii_success = seeds.radii[seeds.contour_success]
        radii_fail = seeds.radii[~seeds.contour_success]
        r_min, r_max = seeds.radii.min(), seeds.radii.max()
        bins = np.linspace(r_min * 0.8, r_max * 1.2, 21)
        if len(radii_success) > 0:
            ax.hist(radii_success, bins=bins, alpha=0.7, label='Contour',
                   color='forestgreen', edgecolor='black')
        if len(radii_fail) > 0:
            ax.hist(radii_fail, bins=bins, alpha=0.7, label='Fallback',
                   color='coral', edgecolor='black')
        ax.axvline(np.median(seeds.radii), color='black', linestyle='--',
                  label=f'Median: {np.median(seeds.radii):.1f}px')
        ax.legend(fontsize=9)
    ax.set_xlabel('Radius (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Size Distribution')
    
    # Circularity histogram (for successful contours)
    ax = axes3[0, 2]
    circularities = []
    for i in range(seeds.n_seeds):
        if seeds.contour_success[i] and seeds.contours[i] is not None:
            circularities.append(seeds.contours[i].circularity)
    if len(circularities) > 0:
        ax.hist(circularities, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.median(circularities), color='black', linestyle='--',
                  label=f'Median: {np.median(circularities):.2f}')
        ax.legend(fontsize=9)
    ax.set_xlabel('Circularity (1=perfect circle)')
    ax.set_ylabel('Count')
    ax.set_title('Contour Circularity')
    
    # Spatial distribution
    ax = axes3[1, 0]
    if seeds.n_seeds > 0:
        sc = ax.scatter(seeds.centers[:, 1], seeds.centers[:, 0], 
                       c=seeds.confidence, cmap='RdYlGn', s=10, alpha=0.6)
        ax.set_xlim(0, d2)
        ax.set_ylim(d1, 0)
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label='Confidence')
    ax.set_title('Spatial Distribution (color=confidence)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Intensity vs radius
    ax = axes3[1, 1]
    if seeds.n_seeds > 0:
        colors = ['forestgreen' if s else 'coral' for s in seeds.contour_success]
        ax.scatter(seeds.radii, seeds.intensities, c=colors, alpha=0.5, s=20)
        ax.set_xlabel('Radius (pixels)')
        ax.set_ylabel('Intensity')
        ax.set_title('Radius vs Intensity')
    
    # Summary text
    ax = axes3[1, 2]
    ax.axis('off')
    summary_text = f"""
    Detection Summary
    ─────────────────
    Total seeds: {seeds.n_seeds}
    With contours: {seeds.n_contours} ({100*seeds.contour_success_rate:.1f}%)
    Fallback (circular): {seeds.n_seeds - seeds.n_contours}
    
    Radius range: {seeds.radii.min():.1f} - {seeds.radii.max():.1f} px
    Median radius: {np.median(seeds.radii):.1f} px
    
    Confidence range: {seeds.confidence.min():.2f} - {seeds.confidence.max():.2f}
    Median confidence: {np.median(seeds.confidence):.2f}
    """
    if len(circularities) > 0:
        summary_text += f"""
    Circularity range: {min(circularities):.2f} - {max(circularities):.2f}
    Median circularity: {np.median(circularities):.2f}
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'detection_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    logger.info(f"  Saved detailed visualizations to {output_dir}")


def visualize_contour_detection(
    projections: ProjectionSet,
    seeds: ContourSeedResult,
    output_path: str,
    max_seeds_to_show: int = 500,
):
    """
    Create comprehensive visualization of contour detection results.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.colors import Normalize
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Projections with detections
    # Max projection
    ax = axes[0, 0]
    ax.imshow(projections.max_norm, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Max Projection\n({seeds.n_seeds} seeds)', fontsize=12)
    ax.axis('off')
    
    # Add seed markers
    for i in range(min(seeds.n_seeds, max_seeds_to_show)):
        y, x = seeds.centers[i]
        r = seeds.radii[i]
        color = 'lime' if seeds.contour_success[i] else 'red'
        circle = Circle((x, y), r, color=color, fill=False, linewidth=0.8, alpha=0.7)
        ax.add_patch(circle)
    
    # Correlation image
    ax = axes[0, 1]
    ax.imshow(projections.correlation_norm, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Correlation Image', fontsize=12)
    ax.axis('off')
    
    # Std projection
    ax = axes[0, 2]
    std_norm = (projections.std_proj - projections.std_proj.min()) / \
               (projections.std_proj.max() - projections.std_proj.min() + 1e-10)
    ax.imshow(std_norm, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Std Projection', fontsize=12)
    ax.axis('off')
    
    # Row 2: Contour overlays and statistics
    # Contour overlay on max projection
    ax = axes[1, 0]
    ax.imshow(projections.max_norm, cmap='gray', vmin=0, vmax=1)
    
    # Draw contours
    n_drawn = 0
    for i in range(min(seeds.n_seeds, max_seeds_to_show)):
        if seeds.contour_success[i] and seeds.contours[i] is not None:
            contour = seeds.contours[i].contour
            # Convert contour to plottable format
            pts = contour.squeeze()
            if len(pts.shape) == 2 and pts.shape[0] > 2:
                # Close the contour
                pts_closed = np.vstack([pts, pts[0]])
                ax.plot(pts_closed[:, 0], pts_closed[:, 1], 'c-', linewidth=0.8, alpha=0.7)
                n_drawn += 1
    
    ax.set_title(f'Contour Overlay\n({n_drawn} contours shown)', fontsize=12)
    ax.axis('off')
    
    # Confidence histogram
    ax = axes[1, 1]
    if seeds.n_seeds > 0:
        # Split by contour success
        conf_success = seeds.confidence[seeds.contour_success]
        conf_fail = seeds.confidence[~seeds.contour_success]
        
        bins = np.linspace(0, 1, 21)
        if len(conf_success) > 0:
            ax.hist(conf_success, bins=bins, alpha=0.7, label=f'With contour ({len(conf_success)})',
                   color='forestgreen', edgecolor='black')
        if len(conf_fail) > 0:
            ax.hist(conf_fail, bins=bins, alpha=0.7, label=f'Fallback ({len(conf_fail)})',
                   color='coral', edgecolor='black')
        
        ax.axvline(np.median(seeds.confidence), color='black', linestyle='--',
                  label=f'Median: {np.median(seeds.confidence):.2f}')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution', fontsize=12)
        ax.legend(fontsize=9)
    
    # Size distribution
    ax = axes[1, 2]
    if seeds.n_seeds > 0:
        radii_success = seeds.radii[seeds.contour_success]
        radii_fail = seeds.radii[~seeds.contour_success]
        
        bins = np.linspace(seeds.radii.min() * 0.8, seeds.radii.max() * 1.2, 21)
        if len(radii_success) > 0:
            ax.hist(radii_success, bins=bins, alpha=0.7, label='With contour',
                   color='forestgreen', edgecolor='black')
        if len(radii_fail) > 0:
            ax.hist(radii_fail, bins=bins, alpha=0.7, label='Fallback',
                   color='coral', edgecolor='black')
        
        ax.axvline(np.median(seeds.radii), color='black', linestyle='--',
                  label=f'Median: {np.median(seeds.radii):.1f}px')
        ax.set_xlabel('Radius (pixels)')
        ax.set_ylabel('Count')
        ax.set_title('Size Distribution', fontsize=12)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {output_path}")


def save_projection_figures(
    projections: ProjectionSet,
    seeds: ContourSeedResult,
    output_dir: str,
    max_seeds_to_show: int = 500,
    dpi: int = 150,
    projections_corr: 'Optional[ProjectionSet]' = None,
):
    """
    Save projection panels and detection overlays into output_dir/figures/.

    All images use percentile-based intensity normalisation (1st–99th) for
    consistent contrast across datasets.

    Outputs
    -------
    figures/projection_summary.png    — 1×3 panel: Max, Correlation, Std
    figures/max_projection.png        — max-projection with seed circles
    figures/correlation_image.png     — local correlation image
    figures/std_projection.png        — standard-deviation projection
    figures/contour_overlay.png       — max-projection with contour outlines
    figures/confidence_distribution.png
    figures/size_distribution.png
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # Correlation image source: prefer the smoothed set if provided
    corr_src = projections_corr if projections_corr is not None else projections

    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Percentile normalisation for VISUAL OUTPUT only — clips extremes
    # for better contrast in figures.  Detection uses min-max via
    # ProjectionSet._normalize() which preserves the full range.
    def _pnorm(data):
        p1 = np.percentile(data, 1)
        p99 = np.percentile(data, 99)
        if p99 - p1 < 1e-10:
            return np.zeros_like(data, dtype=np.float32)
        return np.clip((data - p1) / (p99 - p1), 0, 1).astype(np.float32)

    max_norm = _pnorm(projections.max_proj)
    corr_norm = _pnorm(corr_src.correlation)
    std_norm = _pnorm(projections.std_proj)

    def _imshow_only(data, cmap='gray', vmin=0, vmax=1):
        """Return a tight figure containing only the image, no axes chrome."""
        h, w = data.shape
        fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')
        return fig, ax

    # ── 0. Projection Summary Panel (1×3) ────────────────────────────────
    d1, d2 = projections.max_proj.shape
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Projection Summary', fontsize=14, fontweight='bold', y=0.98)

    panels = [
        (max_norm, 'Max Projection'),
        (corr_norm, 'Local Correlation'),
        (std_norm, 'Std Projection'),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(figures_dir, 'projection_summary.png'),
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ── 1. Max Projection with seed circles ──────────────────────────────
    fig, ax = _imshow_only(max_norm)
    for i in range(min(seeds.n_seeds, max_seeds_to_show)):
        y, x = seeds.centers[i]
        r = seeds.radii[i]
        color = 'lime' if seeds.contour_success[i] else 'red'
        ax.add_patch(Circle((x, y), r, color=color, fill=False, linewidth=0.8, alpha=0.7))
    fig.savefig(os.path.join(figures_dir, 'max_projection.png'),
                dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # ── 2. Correlation Image ─────────────────────────────────────────────
    fig, ax = _imshow_only(corr_norm)
    fig.savefig(os.path.join(figures_dir, 'correlation_image.png'),
                dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # ── 3. Std Projection ────────────────────────────────────────────────
    fig, ax = _imshow_only(std_norm)
    fig.savefig(os.path.join(figures_dir, 'std_projection.png'),
                dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # ── 4. Contour Overlay ───────────────────────────────────────────────
    fig, ax = _imshow_only(max_norm)
    n_drawn = 0
    for i in range(min(seeds.n_seeds, max_seeds_to_show)):
        if seeds.contour_success[i] and seeds.contours[i] is not None:
            pts = seeds.contours[i].contour.squeeze()
            if len(pts.shape) == 2 and pts.shape[0] > 2:
                pts_closed = np.vstack([pts, pts[0]])
                ax.plot(pts_closed[:, 0], pts_closed[:, 1], 'c-', linewidth=0.8, alpha=0.7)
                n_drawn += 1
    fig.savefig(os.path.join(figures_dir, 'contour_overlay.png'),
                dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # ── 5. Confidence Distribution ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    if seeds.n_seeds > 0:
        conf_success = seeds.confidence[seeds.contour_success]
        conf_fail = seeds.confidence[~seeds.contour_success]
        bins = np.linspace(0, 1, 21)
        if len(conf_success) > 0:
            ax.hist(conf_success, bins=bins, alpha=0.7,
                    label=f'With contour ({len(conf_success)})',
                    color='forestgreen', edgecolor='black')
        if len(conf_fail) > 0:
            ax.hist(conf_fail, bins=bins, alpha=0.7,
                    label=f'Fallback ({len(conf_fail)})',
                    color='coral', edgecolor='black')
        ax.axvline(np.median(seeds.confidence), color='black', linestyle='--',
                   label=f'Median: {np.median(seeds.confidence):.2f}')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution')
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'confidence_distribution.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # ── 6. Size Distribution ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    if seeds.n_seeds > 0:
        radii_success = seeds.radii[seeds.contour_success]
        radii_fail = seeds.radii[~seeds.contour_success]
        bins = np.linspace(seeds.radii.min() * 0.8, seeds.radii.max() * 1.2, 21)
        if len(radii_success) > 0:
            ax.hist(radii_success, bins=bins, alpha=0.7, label='With contour',
                    color='forestgreen', edgecolor='black')
        if len(radii_fail) > 0:
            ax.hist(radii_fail, bins=bins, alpha=0.7, label='Fallback',
                    color='coral', edgecolor='black')
        ax.axvline(np.median(seeds.radii), color='black', linestyle='--',
                   label=f'Median: {np.median(seeds.radii):.1f}px')
        ax.set_xlabel('Radius (pixels)')
        ax.set_ylabel('Count')
        ax.set_title('Size Distribution')
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'size_distribution.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved projection summary + {n_drawn} contour overlays + "
                f"6 figures to {figures_dir}/")
    return figures_dir


