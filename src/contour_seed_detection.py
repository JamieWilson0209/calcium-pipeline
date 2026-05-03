"""
Contour-Based Seed Detection Module
=====================================

Detection pipeline:
  1. Multi-projection computation  (max, correlation, mean, std)
  2. LoG blob detection on each projection
  3. Duplicate merging across projections
  4. Per-blob Otsu contour extraction from peak-activity frames
  5. Confidence scoring and boundary filtering

Contour extraction strategy
----------------------------
For each blob candidate the module identifies the frames in which that
specific neuron is most active (top-N by mean intensity within the blob
radius) and builds a local max-projection from those frames only.  Otsu
thresholding is then applied to the locality-masked ROI.  This
activity-gated projection suppresses bright neighbouring cells and
calcium hotspots that would otherwise dominate a global max projection,
giving reliable contour boundaries even in dense organoid fields.

Spatial footprints
-------------------
Successful contours are converted to binary masks.  Blobs where Otsu
thresholding fails fall back to a Gaussian circular footprint centred on
the blob's detected centre.
"""

import json
import os
import time

import cv2
import numpy as np
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_log
from scipy.sparse import csc_matrix, lil_matrix
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ContourInfo:
    """Spatial properties of a single extracted contour."""
    contour: np.ndarray               # OpenCV contour array (N, 1, 2)
    center: Tuple[float, float]       # (row, col) centroid from moments
    area: float                       # Contour area in pixels
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) bounding box
    circularity: float                # 4π × area / perimeter²
    solidity: float                   # area / convex_hull_area
    mean_intensity: float             # Mean intensity within contour

    def to_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Return a binary uint8 mask with the contour interior filled."""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, 255, -1)
        return mask


@dataclass
class BlobDetection:
    """Single LoG blob detection result."""
    center: Tuple[float, float]  # (row, col)
    sigma: float
    radius: float                # sigma * sqrt(2)
    intensity: float
    source: str                  # 'max', 'correlation', 'mean', or 'std'


@dataclass
class ContourSeedResult:
    """Full detection result set returned by detect_seeds_with_contours."""

    # Core seed arrays — all length N
    centers: np.ndarray           # (N, 2)  (row, col)
    radii: np.ndarray             # (N,)
    intensities: np.ndarray       # (N,)
    confidence: np.ndarray        # (N,)  0–1
    source_projection: np.ndarray # (N,)  string codes

    # Contour data
    contours: List[Optional[ContourInfo]]  # None where extraction failed
    contour_success: np.ndarray            # (N,) bool

    # All boundary-touching contours are dropped before this result is
    # returned, so this array is always all-False.
    boundary_touching: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool)
    )

    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_seeds(self) -> int:
        return len(self.centers)

    @property
    def n_contours(self) -> int:
        return int(self.contour_success.sum())

    @property
    def contour_success_rate(self) -> float:
        return self.n_contours / self.n_seeds if self.n_seeds > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict (contour arrays excluded — too large)."""
        return {
            'centers': self.centers.tolist(),
            'radii': self.radii.tolist(),
            'intensities': self.intensities.tolist(),
            'confidence': self.confidence.tolist(),
            'n_seeds': self.n_seeds,
            'n_contours': self.n_contours,
            'contour_success_rate': self.contour_success_rate,
            'source_projection': self.source_projection.tolist(),
            'boundary_touching': (
                self.boundary_touching.tolist()
                if len(self.boundary_touching) else []
            ),
            'diagnostics': self.diagnostics,
        }


@dataclass
class ProjectionSet:
    """
    Collection of temporal projection images computed from a calcium movie.

    All projections are computed once and cached here for reuse by both
    blob detection and contour extraction.

    Normalised variants (``*_norm``) are min-max scaled to [0, 1] and
    used for LoG blob detection.  Figure output should use percentile
    clipping separately — see ``save_projection_figures()`` in
    ``contour_seed_detection_viz.py``.
    """
    max_proj: np.ndarray
    mean_proj: np.ndarray
    std_proj: np.ndarray
    correlation: np.ndarray

    # Normalised versions — set automatically in __post_init__
    max_norm: np.ndarray = field(init=False)
    mean_norm: np.ndarray = field(init=False)
    std_norm: np.ndarray = field(init=False)
    correlation_norm: np.ndarray = field(init=False)

    compute_time_seconds: float = 0.0
    movie_shape: Tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        self.max_norm = self._normalize(self.max_proj)
        self.mean_norm = self._normalize(self.mean_proj)
        self.std_norm = self._normalize(self.std_proj)
        self.correlation_norm = self._normalize(self.correlation)

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        """Min-max scale to [0, 1]."""
        lo, hi = img.min(), img.max()
        if hi > lo:
            return ((img - lo) / (hi - lo)).astype(np.float32)
        return np.zeros_like(img, dtype=np.float32)


# =============================================================================
# PROJECTIONS
# =============================================================================

def compute_projections(
    movie: np.ndarray,
    smooth_sigma: float = 0.0,
    compute_correlation: bool = True,
) -> ProjectionSet:
    """
    Compute temporal projections from a calcium imaging movie.

    Parameters
    ----------
    movie : ndarray (T, Y, X)
    smooth_sigma : float
        If > 0, apply per-frame Gaussian smoothing before computing
        projections.  This suppresses sub-cellular calcium hotspots
        (synaptic puncta, dendritic spines) that would otherwise seed
        spurious blob detections.  The same sigma is later reused inside
        ``extract_contour`` when building the per-blob local projection.
        Recommended range: 3–5 for widefield organoid data.
    compute_correlation : bool
        Whether to compute the local pixel-neighbour correlation image.
        This is the most expensive projection (~30 % of total compute)
        but substantially improves detection in low-SNR recordings.

    Returns
    -------
    ProjectionSet
        Call this once per recording and pass the result via
        ``precomputed_projections`` to avoid redundant computation.
    """
    start = time.time()
    T, d1, d2 = movie.shape
    logger.info(f"Computing projections for movie {movie.shape}...")

    if smooth_sigma > 0:
        logger.info(f"  Per-frame Gaussian smoothing (sigma={smooth_sigma})")
        movie_proc = np.stack(
            [gaussian_filter(movie[t].astype(np.float32), sigma=smooth_sigma)
             for t in range(T)]
        )
    else:
        movie_proc = movie

    max_proj  = np.max(movie_proc,  axis=0).astype(np.float32)
    mean_proj = np.mean(movie_proc, axis=0).astype(np.float32)
    std_proj  = np.std(movie_proc,  axis=0).astype(np.float32)
    logger.info("  Computed: max, mean, std")

    if compute_correlation:
        logger.info("  Computing local correlation image...")
        movie_zs = movie_proc.astype(np.float32, copy=True)
        tmean = movie_zs.mean(axis=0)
        movie_zs -= tmean
        tstd = movie_zs.std(axis=0)
        tstd[tstd == 0] = 1
        movie_zs /= tstd
        del tmean, tstd

        corr_img = np.zeros((d1, d2), dtype=np.float64)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                sy = slice(max(0, -di), d1 - max(0, di))
                sx = slice(max(0, -dj), d2 - max(0, dj))
                ny = slice(max(0,  di), d1 - max(0, -di))
                nx = slice(max(0,  dj), d2 - max(0, -dj))
                corr_img[sy, sx] += np.mean(
                    movie_zs[:, sy, sx] * movie_zs[:, ny, nx], axis=0
                )
        correlation = (corr_img / 8).astype(np.float32)
        del movie_zs
    else:
        correlation = np.zeros((d1, d2), dtype=np.float32)

    if smooth_sigma > 0:
        del movie_proc

    elapsed = time.time() - start
    logger.info(f"  Projections complete in {elapsed:.2f}s")

    return ProjectionSet(
        max_proj=max_proj,
        mean_proj=mean_proj,
        std_proj=std_proj,
        correlation=correlation,
        compute_time_seconds=elapsed,
        movie_shape=(T, d1, d2),
    )


# =============================================================================
# BLOB DETECTION
# =============================================================================

def _detect_blobs_on_projection(
    image: np.ndarray,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int = 10,
    log_threshold: float = 0.05,
    min_intensity: float = 0.2,
    border_margin: int = 20,
    local_contrast_threshold: float = 0.6,
    source_name: str = 'unknown',
) -> Tuple[List[BlobDetection], Dict[str, Any]]:
    """
    Run LoG blob detection on a single normalised projection image.

    Blobs are filtered by: border proximity, absolute intensity, local
    contrast, and local-maximum criterion.

    Returns (detections, diagnostics_dict).
    """
    diagnostics: Dict[str, Any] = {
        'source': source_name,
        'raw_detections': 0,
        'rejected': {'border': 0, 'intensity': 0,
                     'local_contrast': 0, 'not_local_max': 0},
        'final_detections': 0,
    }

    img_norm = image.astype(np.float32)
    lo, hi = img_norm.min(), img_norm.max()
    if hi <= lo:
        logger.warning(f"  [{source_name}] Image has no contrast — skipping")
        return [], diagnostics
    img_norm = (img_norm - lo) / (hi - lo)

    logger.info(
        f"  [{source_name}] LoG (sigma={min_sigma:.1f}–{max_sigma:.1f}, "
        f"thresh={log_threshold})"
    )
    try:
        raw_blobs = blob_log(
            img_norm,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=log_threshold,
        )
    except Exception as exc:
        logger.error(f"  [{source_name}] blob_log failed: {exc}")
        return [], diagnostics

    diagnostics['raw_detections'] = len(raw_blobs)
    logger.info(f"  [{source_name}] Raw detections: {len(raw_blobs)}")

    d1, d2 = image.shape
    detections: List[BlobDetection] = []

    for y, x, sigma in raw_blobs:
        radius = sigma * np.sqrt(2)

        if (y < border_margin or y >= d1 - border_margin or
                x < border_margin or x >= d2 - border_margin):
            diagnostics['rejected']['border'] += 1
            continue

        yi, xi = int(round(y)), int(round(x))
        local_intensity = img_norm[yi, xi]

        if local_intensity < min_intensity:
            diagnostics['rejected']['intensity'] += 1
            continue

        ns = max(10, int(radius * 1.5))
        patch = img_norm[
            max(0, yi - ns):min(d1, yi + ns + 1),
            max(0, xi - ns):min(d2, xi + ns + 1),
        ]
        p_lo, p_hi = patch.min(), patch.max()
        local_contrast = (
            (local_intensity - p_lo) / (p_hi - p_lo)
            if p_hi > p_lo else 0.5
        )
        if local_contrast < local_contrast_threshold:
            diagnostics['rejected']['local_contrast'] += 1
            continue

        if local_intensity < p_hi * 0.90:
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
    logger.info(
        f"  [{source_name}] After filtering: {len(detections)} blobs  "
        f"(rejected — border:{diagnostics['rejected']['border']}  "
        f"intensity:{diagnostics['rejected']['intensity']}  "
        f"contrast:{diagnostics['rejected']['local_contrast']}  "
        f"not_max:{diagnostics['rejected']['not_local_max']})"
    )
    return detections, diagnostics


def _merge_blob_detections(
    blobs_list: List[List[BlobDetection]],
    min_distance: float,
) -> Tuple[List[BlobDetection], Dict[str, Any]]:
    """
    Merge detections from multiple projections, removing spatial duplicates.

    When two blobs are within ``min_distance`` pixels the one with higher
    intensity is kept.  Sources are tracked so multi-projection hits can
    be logged.
    """
    diagnostics: Dict[str, Any] = {
        'total_input': sum(len(b) for b in blobs_list),
        'per_source': {f'source_{i}': len(b) for i, b in enumerate(blobs_list)},
        'duplicates_removed': 0,
        'multi_source_detections': 0,
        'final_count': 0,
    }

    all_blobs = [b for sublist in blobs_list for b in sublist]
    if not all_blobs:
        return [], diagnostics

    all_blobs.sort(key=lambda b: b.intensity, reverse=True)

    kept: List[BlobDetection] = []
    kept_centers: List[Tuple[float, float]] = []
    source_sets: Dict[int, set] = {}

    for blob in all_blobs:
        cy, cx = blob.center
        duplicate_idx = next(
            (i for i, (ky, kx) in enumerate(kept_centers)
             if np.hypot(cy - ky, cx - kx) < min_distance),
            None,
        )
        if duplicate_idx is not None:
            diagnostics['duplicates_removed'] += 1
            source_sets[duplicate_idx].add(blob.source)
        else:
            source_sets[len(kept)] = {blob.source}
            kept.append(blob)
            kept_centers.append(blob.center)

    diagnostics['multi_source_detections'] = sum(
        1 for s in source_sets.values() if len(s) > 1
    )
    diagnostics['final_count'] = len(kept)
    logger.info(
        f"  Merged {diagnostics['total_input']} blobs → {len(kept)} unique  "
        f"({diagnostics['duplicates_removed']} duplicates,  "
        f"{diagnostics['multi_source_detections']} multi-projection)"
    )
    return kept, diagnostics


# =============================================================================
# CONTOUR-LEVEL OVERLAP MERGE
# =============================================================================
#
# The blob-level NMS above does only sub-pixel duplicate removal — it
# compares centre points without knowing about contour shape. This stage
# runs *after* contour extraction and fuses contours that overlap heavily
# in mask space. It exists because auto-radius tends to underestimate
# large neurons; the LoG detector then fires on multiple sub-cellular
# hotspots inside a single cell, each growing into its own contour, and
# those contours largely overlap.
#
# Decision rule (per pair):
#     min-overlap = |A∩B| / min(|A|, |B|)
#     IoU         = |A∩B| / |A∪B|
#     edge if min-overlap ≥ τ_min  OR  IoU ≥ τ_iou
#
# Merge unit: connected components on the resulting graph, fused by
# convex hull of the union. Components whose hull is more than
# ``max_area_growth`` × the largest member's area trip the runaway guard
# and are kept as independent singletons (better to under-merge than to
# fuse two real cells via a chain).


def _bboxes_intersect(b1, b2) -> bool:
    """Cheap O(1) intersection test on (x, y, w, h) bboxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or
                y1 + h1 < y2 or y2 + h2 < y1)


def _build_overlap_graph(
    contours: List["ContourInfo"],
    dims: Tuple[int, int],
    min_overlap_threshold: float,
    iou_threshold: float,
) -> Tuple[List[List[int]], Dict[str, int]]:
    """
    Build adjacency list for contour-overlap graph.

    Bounding-box pre-filter prunes most pairs without rasterising masks.
    Masks are rasterised lazily on first use and cached for the duration
    of this call.
    """
    n = len(contours)
    adj: List[List[int]] = [[] for _ in range(n)]

    bboxes = [c.bbox for c in contours]
    areas = np.array([c.area for c in contours], dtype=np.float64)
    masks_cache: Dict[int, np.ndarray] = {}

    def get_mask(i: int) -> np.ndarray:
        if i not in masks_cache:
            masks_cache[i] = contours[i].to_mask(dims) > 0
        return masks_cache[i]

    n_mask_checks = 0
    n_edges = 0

    for i in range(n):
        for j in range(i + 1, n):
            if not _bboxes_intersect(bboxes[i], bboxes[j]):
                continue
            n_mask_checks += 1
            mi = get_mask(i)
            mj = get_mask(j)
            inter = int(np.logical_and(mi, mj).sum())
            if inter == 0:
                continue
            min_a = min(areas[i], areas[j])
            union = areas[i] + areas[j] - inter
            min_overlap = inter / min_a if min_a > 0 else 0.0
            iou = inter / union if union > 0 else 0.0
            if min_overlap >= min_overlap_threshold or iou >= iou_threshold:
                adj[i].append(j)
                adj[j].append(i)
                n_edges += 1

    return adj, {
        'n_pairs_total': n * (n - 1) // 2,
        'n_bbox_pass': n_mask_checks,
        'n_edges': n_edges,
        'n_masks_rasterised': len(masks_cache),
    }


def _connected_components(adj: List[List[int]]) -> List[List[int]]:
    """Return list of components (each a list of node indices) via BFS."""
    n = len(adj)
    seen = [False] * n
    components: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        comp: List[int] = []
        while stack:
            u = stack.pop()
            if seen[u]:
                continue
            seen[u] = True
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    stack.append(v)
        components.append(comp)
    return components


def _convex_hull_contour_info(
    members: List["ContourInfo"],
    member_intensities: np.ndarray,
    dims: Tuple[int, int],
) -> "ContourInfo":
    """
    Build a ContourInfo from the convex hull of member contours.

    Centre is the intensity-weighted average of member centres.
    Mean intensity is the area-weighted average of member intensities
    (cheap proxy — avoids re-reading the source image).
    """
    all_pts = np.vstack([c.contour for c in members])
    hull_cnt = cv2.convexHull(all_pts)

    area = cv2.contourArea(hull_cnt)
    perimeter = cv2.arcLength(hull_cnt, True)
    circularity = 4 * np.pi * area / perimeter ** 2 if perimeter > 0 else 0.0

    component_area = sum(c.area for c in members)
    solidity = component_area / area if area > 0 else 0.0

    weights = member_intensities.astype(np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    cy = float(np.average([c.center[0] for c in members], weights=weights))
    cx = float(np.average([c.center[1] for c in members], weights=weights))

    bbox = cv2.boundingRect(hull_cnt)

    member_areas = np.array([c.area for c in members], dtype=np.float64)
    member_means = np.array([c.mean_intensity for c in members], dtype=np.float64)
    if member_areas.sum() > 0:
        mean_intensity = float(np.average(member_means, weights=member_areas))
    else:
        mean_intensity = float(np.mean(member_means))

    return ContourInfo(
        contour=hull_cnt,
        center=(cy, cx),
        area=area,
        bbox=bbox,
        circularity=circularity,
        solidity=solidity,
        mean_intensity=mean_intensity,
    )


def _merge_overlapping_contours(
    contours: List[Optional["ContourInfo"]],
    contour_success: List[bool],
    intensities: np.ndarray,
    dims: Tuple[int, int],
    *,
    min_overlap_threshold: float = 0.4,
    iou_threshold: float = 0.2,
    max_area_growth: float = 4.0,
) -> Tuple[
    List[Optional["ContourInfo"]],
    List[bool],
    np.ndarray,
    List[List[int]],
    Dict[str, Any],
]:
    """
    Shape-aware merge of overlapping contours via convex hull of
    connected components.

    Successful contours that overlap above threshold are grouped and
    fused into a single contour (convex hull of the union). Failed-
    contour seeds pass through unchanged — they have no mask to compare.

    Returns
    -------
    new_contours, new_success, new_intensities, kept_indices, diagnostics
        ``kept_indices[i]`` lists the original-index members that map
        into output index ``i``. Singletons report ``[orig_i]``; merged
        groups report all members.
    """
    n = len(contours)
    if n == 0:
        return [], [], np.array([]), [], {
            'merge_groups': 0, 'rejected_runaway': 0, 'singletons_passed': 0,
            'n_input': 0, 'n_output': 0, 'n_input_success': 0,
            'n_pairs_total': 0, 'n_bbox_pass': 0, 'n_edges': 0,
            'n_masks_rasterised': 0,
        }

    # Successful-contour subset for graph construction
    success_idx = [i for i, ok in enumerate(contour_success)
                   if ok and contours[i] is not None]
    success_contours = [contours[i] for i in success_idx]

    if len(success_idx) < 2:
        return (
            list(contours), list(contour_success), intensities.copy(),
            [[i] for i in range(n)],
            {
                'merge_groups': 0, 'rejected_runaway': 0,
                'singletons_passed': n,
                'n_input': n, 'n_output': n,
                'n_input_success': len(success_idx),
                'n_pairs_total': 0, 'n_bbox_pass': 0, 'n_edges': 0,
                'n_masks_rasterised': 0,
            },
        )

    adj, graph_diag = _build_overlap_graph(
        success_contours, dims,
        min_overlap_threshold=min_overlap_threshold,
        iou_threshold=iou_threshold,
    )
    components = _connected_components(adj)

    new_contours: List[Optional[ContourInfo]] = []
    new_success: List[bool] = []
    new_intensities: List[float] = []
    kept_indices: List[List[int]] = []

    rejected_runaway = 0
    merged_groups = 0
    success_handled: set = set()

    for comp in components:
        member_orig_indices = [success_idx[k] for k in comp]
        success_handled.update(member_orig_indices)

        if len(comp) == 1:
            i = member_orig_indices[0]
            new_contours.append(contours[i])
            new_success.append(True)
            new_intensities.append(float(intensities[i]))
            kept_indices.append([i])
            continue

        members = [contours[i] for i in member_orig_indices]
        member_ints = intensities[member_orig_indices]
        max_member_area = max(m.area for m in members)

        merged_ci = _convex_hull_contour_info(members, member_ints, dims)

        if merged_ci.area > max_area_growth * max_member_area:
            # Runaway: keep all members independent
            rejected_runaway += 1
            for i in member_orig_indices:
                new_contours.append(contours[i])
                new_success.append(True)
                new_intensities.append(float(intensities[i]))
                kept_indices.append([i])
        else:
            merged_groups += 1
            new_contours.append(merged_ci)
            new_success.append(True)
            new_intensities.append(float(np.max(member_ints)))
            kept_indices.append(member_orig_indices)

    # Pass through failed-contour seeds
    singletons_passed = 0
    for i in range(n):
        if i in success_handled:
            continue
        new_contours.append(contours[i])
        new_success.append(bool(contour_success[i]))
        new_intensities.append(float(intensities[i]))
        kept_indices.append([i])
        singletons_passed += 1

    diag = {
        'merge_groups': merged_groups,
        'rejected_runaway': rejected_runaway,
        'singletons_passed': singletons_passed,
        'n_input': n,
        'n_output': len(new_contours),
        'n_input_success': len(success_idx),
        **graph_diag,
    }

    return (
        new_contours,
        new_success,
        np.array(new_intensities, dtype=np.float64),
        kept_indices,
        diag,
    )


# =============================================================================
# CONTOUR EXTRACTION
# =============================================================================

def _gaussian_locality_mask(
    roi_shape: Tuple[int, int],
    center_local: Tuple[int, int],
    blob_radius: float,
) -> np.ndarray:
    """
    Gaussian weight mask centred on a blob's local ROI coordinates.

    Pixels beyond ~2× the blob radius are progressively suppressed,
    preventing contour expansion into brighter neighbouring cells.
    """
    h, w = roi_shape
    cy, cx = center_local
    yy, xx = np.ogrid[:h, :w]
    sigma = blob_radius * 1.5
    return np.exp(-0.5 * ((yy - cy)**2 + (xx - cx)**2) / sigma**2)


def _select_best_contour(
    contours: list,
    center_roi: Tuple[int, int],
    blob_radius: float,
    roi_size: int,
    min_area: float,
    max_area: float,
) -> Optional[np.ndarray]:
    """
    From a list of OpenCV contours return the one closest to the expected
    blob centre and size.

    Scoring: distance_to_centre + log_area_ratio × radius.
    Returns None if no valid candidate exists.
    """
    cx_roi, cy_roi = center_roi
    best_contour = None
    best_score = float('inf')

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue

        mcx = int(M['m10'] / M['m00'])
        mcy = int(M['m01'] / M['m00'])
        dist = np.hypot(mcx - cx_roi, mcy - cy_roi)

        if dist > roi_size * 0.8:
            continue

        size_diff = abs(np.log(area / (np.pi * blob_radius**2 + 1)))
        score = dist + size_diff * blob_radius

        if score < best_score:
            best_score = score
            best_contour = cnt

    return best_contour


def _build_contour_info(
    contour_local: np.ndarray,
    offset: Tuple[int, int],
    image: np.ndarray,
    fallback_center: Tuple[float, float],
) -> ContourInfo:
    """
    Convert a contour in ROI-local coordinates to a ContourInfo in global
    image coordinates, computing shape descriptors and mean intensity.
    """
    x_min, y_min = offset

    cnt = contour_local.copy()
    cnt[:, 0, 0] += x_min
    cnt[:, 0, 1] += y_min

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / perimeter**2 if perimeter > 0 else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    M = cv2.moments(cnt)
    if M['m00'] > 0:
        cx_g = M['m10'] / M['m00']
        cy_g = M['m01'] / M['m00']
    else:
        cy_g, cx_g = fallback_center

    bbox = cv2.boundingRect(cnt)

    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mean_intensity = float(np.mean(image[mask > 0])) if mask.any() else 0.0

    return ContourInfo(
        contour=cnt,
        center=(cy_g, cx_g),
        area=area,
        bbox=bbox,
        circularity=circularity,
        solidity=solidity,
        mean_intensity=mean_intensity,
    )


def extract_contour(
    movie: np.ndarray,
    blob: BlobDetection,
    padding: int = 10,
    morphology_kernel_size: int = 3,
    min_contour_area_fraction: float = 0.5,
    max_contour_area_ratio: float = 6.0,
    n_peak_frames: int = 10,
    peak_percentile: float = 90.0,
    smooth_sigma: float = 0.0,
) -> Tuple[Optional[ContourInfo], Dict[str, Any]]:
    """
    Extract an Otsu-thresholded contour for a single blob candidate.

    Strategy
    --------
    1. Define a padded ROI around the blob (2× radius + padding on each side).
    2. Identify peak-activity frames: frames where mean intensity inside the
       blob radius exceeds ``peak_percentile``, or the top ``n_peak_frames``
       frames if fewer qualify.
    3. Build a local max-projection from those peak frames.
    4. Apply ``smooth_sigma`` Gaussian smoothing to suppress hotspots.
    5. Apply a Gaussian locality mask to down-weight distant pixels.
    6. Otsu-threshold → morphological opening → find contours.
    7. Select the contour closest in position and size to the expected blob.

    Parameters
    ----------
    movie : ndarray (T, Y, X)
    blob : BlobDetection
    padding : int
    morphology_kernel_size : int
    min_contour_area_fraction : float
    max_contour_area_ratio : float
    n_peak_frames : int
    peak_percentile : float
    smooth_sigma : float
        Gaussian smoothing applied to the local ROI projection.

    Returns
    -------
    contour_info : ContourInfo or None
    diagnostics : dict
    """
    T, d1, d2 = movie.shape
    y, x = blob.center
    r = blob.radius

    diagnostics: Dict[str, Any] = {
        'blob_center': blob.center,
        'blob_radius': r,
        'success': False,
        'failure_reason': None,
        'n_peak_frames_used': 0,
        'threshold_value': None,
        'n_contours_found': 0,
        'selected_contour_area': None,
    }

    # ── ROI bounds ───────────────────────────────────────────────────────────
    roi_half = int(r * 2) + padding
    y_min = max(0, int(y - roi_half))
    y_max = min(d1, int(y + roi_half))
    x_min = max(0, int(x - roi_half))
    x_max = min(d2, int(x + roi_half))
    diagnostics['roi_bounds'] = (y_min, y_max, x_min, x_max)

    # ── Peak-frame selection ─────────────────────────────────────────────────
    blob_trace = movie[:, y_min:y_max, x_min:x_max].mean(axis=(1, 2))
    activity_threshold = np.percentile(blob_trace, peak_percentile)
    active_mask = blob_trace >= activity_threshold

    if active_mask.sum() < n_peak_frames:
        top_idx = np.argsort(blob_trace)[-n_peak_frames:]
        active_mask = np.zeros(T, dtype=bool)
        active_mask[top_idx] = True

    diagnostics['n_peak_frames_used'] = int(active_mask.sum())

    # ── Local projection from peak frames ────────────────────────────────────
    roi = movie[active_mask, y_min:y_max, x_min:x_max].max(axis=0).astype(np.float32)

    if roi.size == 0:
        diagnostics['failure_reason'] = 'empty_roi'
        return None, diagnostics

    if smooth_sigma > 0:
        roi = gaussian_filter(roi, sigma=smooth_sigma)

    # ── Gaussian locality mask ───────────────────────────────────────────────
    center_local = (int(y - y_min), int(x - x_min))
    locality_mask = _gaussian_locality_mask(roi.shape, center_local, r)
    roi_masked = roi * locality_mask

    # ── Normalise to uint8 for OpenCV ────────────────────────────────────────
    lo, hi = roi_masked.min(), roi_masked.max()
    if hi <= lo:
        diagnostics['failure_reason'] = 'no_contrast_in_roi'
        return None, diagnostics
    roi_uint8 = ((roi_masked - lo) / (hi - lo) * 255).astype(np.uint8)

    # ── Otsu threshold ───────────────────────────────────────────────────────
    try:
        thresh_val, roi_binary = cv2.threshold(
            roi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    except Exception as exc:
        diagnostics['failure_reason'] = f'otsu_error: {exc}'
        return None, diagnostics
    diagnostics['threshold_value'] = float(thresh_val)

    # ── Otsu inter-class variance ────────────────────────────────────────────
    # Measures the separation between foreground (cell body) and background
    # pixel distributions: σ²_between = w_bg * w_fg * (μ_bg - μ_fg)²
    # A well-matched radius produces a clean bimodal histogram with high
    # inter-class variance.  Used by auto_radius to select the optimal radius.
    pixels = roi_uint8.flatten().astype(np.float64)
    bg = pixels[pixels <= thresh_val]
    fg = pixels[pixels >  thresh_val]
    if len(bg) > 0 and len(fg) > 0:
        w_bg = len(bg) / len(pixels)
        w_fg = len(fg) / len(pixels)
        otsu_variance = w_bg * w_fg * (bg.mean() - fg.mean()) ** 2
    else:
        otsu_variance = 0.0
    diagnostics['otsu_inter_class_variance'] = float(otsu_variance)

    # ── Morphological opening ────────────────────────────────────────────────
    kernel = np.ones((morphology_kernel_size, morphology_kernel_size), np.uint8)
    roi_clean = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, kernel)

    # ── Contour finding and selection ────────────────────────────────────────
    contours_found, _ = cv2.findContours(
        roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    diagnostics['n_contours_found'] = len(contours_found)

    if not contours_found:
        diagnostics['failure_reason'] = 'no_contours_found'
        return None, diagnostics

    expected_area = np.pi * r * r
    best = _select_best_contour(
        contours_found,
        center_roi=(int(x - x_min), int(y - y_min)),
        blob_radius=r,
        roi_size=roi_half,
        min_area=expected_area * min_contour_area_fraction,
        max_area=expected_area * max_contour_area_ratio,
    )

    if best is None:
        diagnostics['failure_reason'] = 'no_valid_contour_near_centre'
        return None, diagnostics

    # ── Build ContourInfo in global coordinates ──────────────────────────────
    peak_proj_global = movie[active_mask].max(axis=0).astype(np.float32)
    contour_info = _build_contour_info(
        contour_local=best,
        offset=(x_min, y_min),
        image=peak_proj_global,
        fallback_center=(y, x),
    )

    diagnostics['success'] = True
    diagnostics['selected_contour_area'] = contour_info.area
    diagnostics['circularity'] = contour_info.circularity
    diagnostics['solidity'] = contour_info.solidity

    return contour_info, diagnostics


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================

def detect_seeds_with_contours(
    movie: np.ndarray,
    min_radius: float = 10.0,
    max_radius: float = 30.0,
    intensity_threshold: float = 0.20,
    correlation_threshold: float = 0.15,
    border_margin: int = 20,
    max_seeds: int = 2000,
    use_correlation: bool = True,
    use_mean: bool = False,
    use_std: bool = False,
    smooth_sigma: float = 0.0,
    n_peak_frames: int = 10,
    peak_percentile: float = 90.0,
    contour_merge_min_overlap: float = 0.4,
    contour_merge_iou: float = 0.2,
    contour_merge_max_growth: float = 4.0,
    diagnostics_dir: Optional[str] = None,
    precomputed_projections: Optional[ProjectionSet] = None,
) -> ContourSeedResult:
    """
    Full contour-based seed detection pipeline.

    Parameters
    ----------
    movie : ndarray (T, Y, X)
    min_radius, max_radius : float
        Expected neuron radius range in pixels.
    intensity_threshold : float
        Minimum normalised intensity for LoG blobs on the max projection.
    correlation_threshold : float
        Minimum normalised intensity for LoG blobs on the correlation image.
    border_margin : int
        Pixels excluded from detection near the FOV edge.
    max_seeds : int
        Cap on total seeds (top-N by intensity after merging).
    use_correlation : bool
        Detect blobs on the local-correlation projection.
    use_mean : bool
        Detect blobs on the mean projection (persistently active neurons).
    use_std : bool
        Detect blobs on the std projection (high temporal-variance neurons).
    smooth_sigma : float
        Gaussian smoothing sigma applied per-frame before projections and
        inside each blob's local ROI during contour extraction.
    n_peak_frames : int
        Number of peak-activity frames per blob for local projection.
    peak_percentile : float
        Percentile threshold for selecting peak-activity frames.
    contour_merge_min_overlap : float
        Min-overlap-fraction threshold for contour merge — fraction of
        the *smaller* contour contained in the larger. Default 0.4.
        Catches the "small hotspot inside a larger cell" case.
    contour_merge_iou : float
        IoU threshold for contour merge. Default 0.2. Catches the
        symmetric mid-overlap case.
    contour_merge_max_growth : float
        Reject merges whose convex hull is more than this multiple of
        the largest member's area. Default 4.0. Guards against chains
        of barely-overlapping contours fusing into a runaway region.
    diagnostics_dir : str, optional
        If given, writes JSON diagnostics here.
    precomputed_projections : ProjectionSet, optional
        Skip projection computation if already done upstream.

    Returns
    -------
    ContourSeedResult
    """
    t0 = time.time()
    T, d1, d2 = movie.shape

    logger.info("=" * 70)
    logger.info("CONTOUR-BASED SEED DETECTION")
    logger.info("=" * 70)
    logger.info(f"Movie: {movie.shape}  radius: {min_radius:.0f}–{max_radius:.0f} px  "
                f"max_seeds: {max_seeds}")

    master_diag: Dict[str, Any] = {
        'movie_shape': movie.shape,
        'parameters': {
            'min_radius': min_radius, 'max_radius': max_radius,
            'intensity_threshold': intensity_threshold,
            'correlation_threshold': correlation_threshold,
            'border_margin': border_margin, 'max_seeds': max_seeds,
            'smooth_sigma': smooth_sigma,
            'n_peak_frames': n_peak_frames,
            'peak_percentile': peak_percentile,
        },
        'timing': {},
        'blob_detection': {},
        'contour_extraction': {
            'total_attempts': 0, 'successes': 0,
            'failures_by_reason': {},
        },
    }

    # =========================================================================
    # STEP 1: Projections
    # =========================================================================
    logger.info("\n--- STEP 1: Projections ---")
    t_step = time.time()

    if precomputed_projections is not None:
        logger.info("  Reusing precomputed projections")
        projections = precomputed_projections
    else:
        projections = compute_projections(
            movie,
            smooth_sigma=smooth_sigma,
            compute_correlation=use_correlation,
        )
    master_diag['timing']['projections'] = time.time() - t_step

    # =========================================================================
    # STEP 2: Blob detection
    # =========================================================================
    logger.info("\n--- STEP 2: Blob detection ---")
    t_step = time.time()

    min_sigma = min_radius / np.sqrt(2)
    max_sigma = max_radius / np.sqrt(2)

    blob_lists: List[List[BlobDetection]] = []

    blobs_max, diag_max = _detect_blobs_on_projection(
        projections.max_norm,
        min_sigma=min_sigma, max_sigma=max_sigma,
        log_threshold=0.05, min_intensity=intensity_threshold,
        border_margin=border_margin, local_contrast_threshold=0.6,
        source_name='max',
    )
    blob_lists.append(blobs_max)
    master_diag['blob_detection']['max'] = diag_max

    if use_correlation:
        blobs_corr, diag_corr = _detect_blobs_on_projection(
            projections.correlation_norm,
            min_sigma=min_sigma, max_sigma=max_sigma,
            log_threshold=0.03, min_intensity=correlation_threshold,
            border_margin=border_margin, local_contrast_threshold=0.5,
            source_name='correlation',
        )
        blob_lists.append(blobs_corr)
        master_diag['blob_detection']['correlation'] = diag_corr

    if use_mean:
        blobs_mean, diag_mean = _detect_blobs_on_projection(
            projections.mean_norm,
            min_sigma=min_sigma, max_sigma=max_sigma,
            log_threshold=0.05, min_intensity=intensity_threshold,
            border_margin=border_margin, local_contrast_threshold=0.6,
            source_name='mean',
        )
        blob_lists.append(blobs_mean)
        master_diag['blob_detection']['mean'] = diag_mean

    if use_std:
        blobs_std, diag_std = _detect_blobs_on_projection(
            projections.std_norm,
            min_sigma=min_sigma, max_sigma=max_sigma,
            log_threshold=0.05, min_intensity=intensity_threshold,
            border_margin=border_margin, local_contrast_threshold=0.6,
            source_name='std',
        )
        blob_lists.append(blobs_std)
        master_diag['blob_detection']['std'] = diag_std

    # Tight NMS: only collapse near-duplicate detections from the same
    # hotspot (sub-pixel offsets, multi-projection hits on identical
    # points). Anatomical fusion of multiple blobs inside one cell is
    # handled later at the contour level via _merge_overlapping_contours.
    merged_blobs, merge_diag = _merge_blob_detections(
        blob_lists, min_distance=min_radius * 0.5
    )
    master_diag['blob_detection']['merge'] = merge_diag
    master_diag['timing']['blob_detection'] = time.time() - t_step

    if not merged_blobs:
        logger.warning("No blobs detected — returning empty result")
        return ContourSeedResult(
            centers=np.empty((0, 2)), radii=np.array([]),
            intensities=np.array([]), confidence=np.array([]),
            contours=[], contour_success=np.array([], dtype=bool),
            source_projection=np.array([], dtype='U10'),
            diagnostics=master_diag,
        )

    if len(merged_blobs) > max_seeds:
        logger.info(f"Capping {len(merged_blobs)} → {max_seeds} blobs by intensity")
        merged_blobs.sort(key=lambda b: b.intensity, reverse=True)
        merged_blobs = merged_blobs[:max_seeds]

    # =========================================================================
    # STEP 3: Contour extraction
    # =========================================================================
    logger.info(f"\n--- STEP 3: Contour extraction ({len(merged_blobs)} blobs) ---")
    t_step = time.time()

    contours: List[Optional[ContourInfo]] = []
    contour_success: List[bool] = []
    per_blob_diagnostics: List[Dict] = []

    for i, blob in enumerate(merged_blobs):
        ci, blob_diag = extract_contour(
            movie, blob,
            padding=int(min_radius),
            min_contour_area_fraction=0.5,
            max_contour_area_ratio=6.0,
            n_peak_frames=n_peak_frames,
            peak_percentile=peak_percentile,
            smooth_sigma=smooth_sigma,
        )
        contours.append(ci)
        contour_success.append(ci is not None)
        per_blob_diagnostics.append(blob_diag)

        if ci is None:
            reason = blob_diag.get('failure_reason', 'unknown')
            fd = master_diag['contour_extraction']['failures_by_reason']
            fd[reason] = fd.get(reason, 0) + 1

        if (i + 1) % 100 == 0:
            logger.info(
                f"  {i+1}/{len(merged_blobs)} blobs processed  "
                f"({sum(contour_success)} contours so far)"
            )

    contour_success_arr = np.array(contour_success)
    master_diag['contour_extraction']['total_attempts'] = len(merged_blobs)
    master_diag['contour_extraction']['successes'] = int(contour_success_arr.sum())
    master_diag['timing']['contour_extraction'] = time.time() - t_step

    logger.info(
        f"  Contour extraction: {contour_success_arr.sum()}/{len(merged_blobs)} "
        f"successful ({100 * contour_success_arr.mean():.1f}%)"
    )

    # =========================================================================
    # STEP 3.5: Shape-aware contour merge
    # =========================================================================
    # Fuse contours that overlap heavily in mask space — typically multiple
    # sub-cellular detections inside one cell that grew into separate
    # contours. See _merge_overlapping_contours for the decision rule.
    logger.info("\n--- STEP 3.5: Contour overlap merge ---")
    t_step = time.time()

    blob_intensities = np.array([b.intensity for b in merged_blobs])
    blob_sources     = np.array([b.source for b in merged_blobs])
    blob_centers     = np.array([b.center for b in merged_blobs])
    blob_radii       = np.array([b.radius for b in merged_blobs])

    (
        contours,
        contour_success,
        intensities,
        kept_indices,
        contour_merge_diag,
    ) = _merge_overlapping_contours(
        contours,
        contour_success,
        blob_intensities,
        dims=(d1, d2),
        min_overlap_threshold=contour_merge_min_overlap,
        iou_threshold=contour_merge_iou,
        max_area_growth=contour_merge_max_growth,
    )
    contour_success_arr = np.array(contour_success)
    master_diag['contour_merge'] = contour_merge_diag
    master_diag['timing']['contour_merge'] = time.time() - t_step

    logger.info(
        f"  Merged contours: {contour_merge_diag['n_input']} → "
        f"{contour_merge_diag['n_output']} "
        f"({contour_merge_diag['merge_groups']} groups merged, "
        f"{contour_merge_diag['rejected_runaway']} rejected as runaway)"
    )

    # Build per-output centres / radii / sources from the merged groups.
    # For multi-member groups the representative source is taken from the
    # highest-intensity member; centre and radius come from the merged
    # contour itself (or fall back to the original blob centre on failures).
    n_out = len(contours)
    centers = np.zeros((n_out, 2), dtype=np.float64)
    radii = np.zeros(n_out, dtype=np.float64)
    sources = np.empty(n_out, dtype=object)

    for i, members in enumerate(kept_indices):
        ci = contours[i]
        if ci is not None and contour_success[i]:
            centers[i] = ci.center
            radii[i] = np.sqrt(ci.area / np.pi)
        else:
            # failed-contour seed: use original blob geometry
            j = members[0]
            centers[i] = blob_centers[j]
            radii[i] = blob_radii[j]

        # Representative source: highest-intensity member's projection
        member_ints = blob_intensities[members]
        sources[i] = blob_sources[members[int(np.argmax(member_ints))]]

    sources = sources.astype('U10')

    # Remap per-blob diagnostics across the merge. Each merged output
    # inherits the highest-intensity member's diagnostics record.
    remapped_diagnostics: List[Dict] = []
    for members in kept_indices:
        member_ints = blob_intensities[members]
        rep = members[int(np.argmax(member_ints))]
        remapped_diagnostics.append(per_blob_diagnostics[rep])
    per_blob_diagnostics = remapped_diagnostics

    # =========================================================================
    # STEP 4: Build output arrays
    # =========================================================================
    logger.info("\n--- STEP 4: Output ---")

    # ── Boundary-touching exclusion ──────────────────────────────────────────
    boundary_touching = np.zeros(n_out, dtype=bool)
    for i, (ci, ok) in enumerate(zip(contours, contour_success)):
        if ok and ci is not None:
            pts = ci.contour.squeeze()
            if pts.ndim == 2:
                if (pts[:, 0].min() == 0 or pts[:, 0].max() == d2 - 1 or
                        pts[:, 1].min() == 0 or pts[:, 1].max() == d1 - 1):
                    boundary_touching[i] = True

    n_boundary = int(boundary_touching.sum())
    if n_boundary:
        logger.info(f"  Excluding {n_boundary} boundary-touching contours")
    keep = ~boundary_touching
    centers             = centers[keep]
    radii               = radii[keep]
    intensities         = intensities[keep]
    sources             = sources[keep]
    contours            = [c for c, k in zip(contours, keep) if k]
    contour_success_arr = contour_success_arr[keep]
    per_blob_diagnostics = [d for d, k in zip(per_blob_diagnostics, keep) if k]
    n_seeds = int(keep.sum())

    # ── Confidence scores ────────────────────────────────────────────────────
    int_lo, int_hi = intensities.min(), intensities.max()
    intensity_score = (
        (intensities - int_lo) / (int_hi - int_lo)
        if int_hi > int_lo else np.full(n_seeds, 0.5)
    )
    contour_boost = np.where(contour_success_arr, 1.0, 0.7)
    source_boost  = np.where(sources == 'max', 1.0, 0.9)

    contour_quality = np.full(n_seeds, 0.5)
    for i, (ci, ok) in enumerate(zip(contours, contour_success_arr)):
        if ok and ci is not None:
            contour_quality[i] = (ci.circularity + ci.solidity) / 2

    confidence = np.clip(
        intensity_score * 0.4 + contour_quality * 0.3 +
        contour_boost * 0.2 + source_boost * 0.1,
        0, 1,
    )

    total_time = time.time() - t0
    master_diag['timing']['total'] = total_time
    master_diag['n_seeds_final'] = n_seeds
    master_diag['n_boundary_excluded'] = n_boundary
    master_diag['per_blob'] = per_blob_diagnostics

    logger.info(f"\nDetection complete: {n_seeds} seeds  "
                f"({contour_success_arr.sum()} contours, "
                f"{n_boundary} boundary excluded)  "
                f"[{total_time:.1f}s]")

    # ── Optional JSON diagnostics ────────────────────────────────────────────
    if diagnostics_dir is not None:
        os.makedirs(diagnostics_dir, exist_ok=True)
        with open(os.path.join(diagnostics_dir,
                               'contour_detection_diagnostics.json'), 'w') as f:
            json.dump(master_diag, f, indent=2, default=str)
        with open(os.path.join(diagnostics_dir,
                               'per_blob_diagnostics.json'), 'w') as f:
            json.dump(per_blob_diagnostics, f, indent=2, default=str)
        logger.info(f"  Diagnostics written to {diagnostics_dir}")

    return ContourSeedResult(
        centers=centers,
        radii=radii,
        intensities=intensities,
        confidence=confidence,
        contours=contours,
        contour_success=contour_success_arr,
        source_projection=sources,
        boundary_touching=np.zeros(n_seeds, dtype=bool),
        diagnostics=master_diag,
    )


# =============================================================================
# SPATIAL FOOTPRINTS
# =============================================================================

def generate_circular_footprint(
    center: Tuple[float, float],
    radius: float,
    dims: Tuple[int, int],
    method: str = 'gaussian',
) -> np.ndarray:
    """
    Generate a circular spatial footprint for a seed.

    Used as a fallback when Otsu contour extraction fails.

    Parameters
    ----------
    center : (row, col)
    radius : float
    dims : (d1, d2)
    method : 'gaussian' or 'disk'
    """
    d1, d2 = dims
    cy, cx = center
    yy, xx = np.ogrid[:d1, :d2]

    if method == 'gaussian':
        sigma = radius / 2
        fp = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))
        fp[fp < 0.01] = 0
    elif method == 'disk':
        fp = (np.hypot(yy - cy, xx - cx) <= radius).astype(float)
    else:
        raise ValueError(f"Unknown method '{method}'; expected 'gaussian' or 'disk'")

    return fp


def contours_to_spatial_footprints(
    seeds: ContourSeedResult,
    dims: Tuple[int, int],
    contour_fallback: bool = True,
    fallback_method: str = 'gaussian',
    normalize: bool = True,
) -> csc_matrix:
    """
    Convert ContourSeedResult to a sparse spatial footprint matrix.

    For seeds with a successful contour the binary contour mask is used.
    For seeds where contour extraction failed, behaviour depends on
    ``contour_fallback``:

    - ``True`` (default): circular footprint via
      :func:`generate_circular_footprint`.
    - ``False``: seed is dropped.

    Parameters
    ----------
    seeds : ContourSeedResult
    dims : (d1, d2)
    contour_fallback : bool
    fallback_method : 'gaussian' or 'disk'
    normalize : bool
        L1-normalise each footprint column.

    Returns
    -------
    csc_matrix (d1*d2, N)
    """
    d1, d2 = dims
    n_pixels = d1 * d2

    if seeds.n_seeds == 0:
        return csc_matrix((n_pixels, 0))

    n_failed = seeds.n_seeds - seeds.n_contours
    logger.info(f"Building spatial footprints for {seeds.n_seeds} seeds "
                f"({seeds.n_contours} contour, {n_failed} fallback/drop)...")

    footprints = []
    n_contour_used = n_fallback_used = n_dropped = 0

    for i in range(seeds.n_seeds):
        if seeds.contour_success[i] and seeds.contours[i] is not None:
            fp = seeds.contours[i].to_mask((d1, d2)).astype(float) / 255.0
            n_contour_used += 1
        elif contour_fallback:
            fp = generate_circular_footprint(
                center=seeds.centers[i],
                radius=seeds.radii[i],
                dims=dims,
                method=fallback_method,
            )
            n_fallback_used += 1
        else:
            n_dropped += 1
            continue

        if normalize and fp.sum() > 0:
            fp /= fp.sum()
        footprints.append(fp.flatten())

    n_kept = len(footprints)
    A = lil_matrix((n_pixels, n_kept))
    for i, fp in enumerate(footprints):
        A[:, i] = fp[:, np.newaxis]

    logger.info(f"  {n_contour_used} contour  {n_fallback_used} fallback  "
                f"{n_dropped} dropped  → {n_kept} total")

    return csc_matrix(A)
