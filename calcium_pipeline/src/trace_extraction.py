"""
Trace Extraction
================

Extracts temporal fluorescence traces from spatial ROI footprints.

Each ROI's trace is the spatial-weighted average of the movie within
its footprint:

    C_i(t) = Σ_x  A_i(x) · Y(x,t)  /  Σ_x A_i(x)

where A_i is the spatial footprint (weight map) and Y is the movie.

Processing is chunked along the time axis to keep memory bounded for
large movies.

Dependencies: numpy, scipy (sparse).
"""

import numpy as np
import logging
from typing import Tuple
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


def extract_traces(
    movie: np.ndarray,
    A,
    *,
    chunk_size: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fluorescence traces from a movie using spatial footprints.

    Parameters
    ----------
    movie : array (T, d1, d2)
        Motion-corrected movie.
    A : sparse or dense matrix (d1*d2, N)
        Spatial footprints — each column is one ROI.
    chunk_size : int
        Frames per processing chunk (default 500).

    Returns
    -------
    C : array (N, T)
        Weighted-average fluorescence traces.
    C_raw : array (N, T)
        Same as C (kept for API compatibility with downstream code
        that expects two return values).
    """
    T, d1, d2 = movie.shape
    n_pixels = d1 * d2
    n_components = A.shape[1]

    logger.info(f"Extracting traces for {n_components} ROIs from "
                f"{T}-frame movie ({d1}×{d2})")

    # Densify footprints for matrix multiply
    A_dense = A.toarray().astype(np.float32) if issparse(A) else np.asarray(A, dtype=np.float32)

    # ROI weights (denominators for weighted average)
    weights = A_dense.sum(axis=0)  # (N,)
    weights[weights == 0] = 1e-10

    # Chunked extraction
    C_raw = np.zeros((n_components, T), dtype=np.float32)

    n_chunks = (T + chunk_size - 1) // chunk_size
    logger.info(f"  Processing in {n_chunks} chunks of {chunk_size} frames...")

    for chunk_i in range(n_chunks):
        t0 = chunk_i * chunk_size
        t1 = min(t0 + chunk_size, T)
        chunk_T = t1 - t0

        # (chunk_T, d1, d2) → (n_pixels, chunk_T)
        Y = movie[t0:t1].reshape(chunk_T, -1).T

        # Weighted average within each ROI
        C_raw[:, t0:t1] = (A_dense.T @ Y) / weights[:, np.newaxis]

        if (chunk_i + 1) % max(1, n_chunks // 5) == 0:
            logger.info(f"    Chunk {chunk_i + 1}/{n_chunks}")

    logger.info(
        f"  Traces: min={C_raw.min():.2f}, max={C_raw.max():.2f}, "
        f"median={np.median(C_raw):.2f}"
    )

    return C_raw, C_raw
