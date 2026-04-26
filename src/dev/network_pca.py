"""
Network PCA — Detection-Free Spatial Decomposition
===================================================

Decomposes the raw calcium imaging movie into principal spatial modes
using PCA on the pixel-by-pixel temporal data.  This reveals population-
level spatial structure (which regions co-activate) without requiring
ROI detection.

Handles the empty-space problem by mean-centring without z-scoring,
so low-variance pixels (empty space) contribute negligibly to the
covariance matrix and receive near-zero PC loadings.

Key outputs:
    - Spatial loading maps for top PCs (automatic tissue segmentation)
    - Temporal weight traces for each PC (when each spatial mode is active)
    - Variance explained per PC
    - Spectral analysis of PC temporal weights

Author: Calcium Pipeline Project
"""

import logging
import os
import json

import numpy as np

logger = logging.getLogger(__name__)


# ─── Core computation ────────────────────────────────────────────────────

def run_movie_pca(movie, n_components=10, downsample_spatial=1):
    """
    PCA decomposition of the movie into spatial modes.

    The movie is reshaped to (T, N_pixels) and decomposed via SVD.
    Mean-centring is applied per pixel (removes static background),
    but z-scoring is NOT applied (would amplify noise in empty pixels).

    Parameters
    ----------
    movie : np.ndarray, shape (T, Y, X)
        Raw fluorescence movie.
    n_components : int
        Number of PCs to extract.
    downsample_spatial : int
        Spatial downsampling factor (2 = half resolution each axis).
        Reduces memory for large FOVs.

    Returns
    -------
    result : dict
        spatial_maps : (n_components, Y_ds, X_ds) — spatial loading per PC
        temporal_weights : (n_components, T) — temporal weight per PC
        variance_explained : (n_components,) — fraction of total variance
        singular_values : (n_components,) — raw singular values
        pixel_means : (Y_ds, X_ds) — mean image (removed before PCA)
        downsample_factor : int
    """
    T, Y, X = movie.shape

    # Optional spatial downsampling
    if downsample_spatial > 1:
        ds = downsample_spatial
        Y_ds = Y // ds
        X_ds = X // ds
        movie_ds = movie[:, :Y_ds * ds, :X_ds * ds].reshape(T, Y_ds, ds, X_ds, ds).mean(axis=(2, 4))
        logger.info(f"  Spatial downsampling {ds}x: {Y}x{X} -> {Y_ds}x{X_ds}")
    else:
        movie_ds = movie
        Y_ds, X_ds = Y, X

    N_pixels = Y_ds * X_ds
    logger.info(f"  PCA input: {T} frames x {N_pixels} pixels")

    # Reshape to (T, N_pixels)
    data = movie_ds.reshape(T, N_pixels).astype(np.float64)

    # Mean-centre per pixel (NOT z-scored — critical for empty space handling)
    pixel_means = data.mean(axis=0)
    data -= pixel_means

    # SVD (compute on the smaller dimension for efficiency)
    # If T < N_pixels, work on (T x T) temporal covariance
    n_components = min(n_components, T - 1, N_pixels)

    if T <= N_pixels:
        logger.info(f"  Computing temporal covariance ({T}x{T})...")
        C = data @ data.T / (N_pixels - 1)
        eigenvalues, V = np.linalg.eigh(C)
        # eigh returns ascending order, reverse
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        eigenvalues = eigenvalues[idx]
        V = V[:, idx]  # (T, n_components) — temporal weights

        # Recover spatial maps: U_i = X^T v_i / sigma_i
        singular_values = np.sqrt(np.maximum(eigenvalues * (N_pixels - 1), 0))
        spatial_flat = np.zeros((n_components, N_pixels), dtype=np.float64)
        for i in range(n_components):
            if singular_values[i] > 1e-10:
                spatial_flat[i] = (data.T @ V[:, i]) / singular_values[i]

        temporal_weights = V.T  # (n_components, T)
    else:
        logger.info(f"  Computing spatial covariance ({N_pixels}x{N_pixels})...")
        C = data.T @ data / (T - 1)
        eigenvalues, U = np.linalg.eigh(C)
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        eigenvalues = eigenvalues[idx]
        U = U[:, idx]  # (N_pixels, n_components)

        singular_values = np.sqrt(np.maximum(eigenvalues * (T - 1), 0))
        spatial_flat = U.T  # (n_components, N_pixels)

        temporal_weights = np.zeros((n_components, T), dtype=np.float64)
        for i in range(n_components):
            if singular_values[i] > 1e-10:
                temporal_weights[i] = (data @ U[:, i]) / singular_values[i]

    # Variance explained
    total_var = np.sum(data ** 2) / (T - 1)
    var_explained = (singular_values ** 2 / (T - 1)) / total_var if total_var > 0 else np.zeros(n_components)

    # Reshape spatial maps
    spatial_maps = spatial_flat.reshape(n_components, Y_ds, X_ds)

    logger.info(f"  Top {n_components} PCs explain {var_explained.sum() * 100:.1f}% of variance")
    for i in range(min(5, n_components)):
        logger.info(f"    PC{i + 1}: {var_explained[i] * 100:.1f}%")

    return {
        'spatial_maps': spatial_maps,
        'temporal_weights': temporal_weights,
        'variance_explained': var_explained,
        'singular_values': singular_values,
        'pixel_means': pixel_means.reshape(Y_ds, X_ds),
        'downsample_factor': downsample_spatial,
    }


def analyse_pc_temporal(temporal_weights, frame_rate, n_analyse=5):
    """
    Spectral analysis of each PC's temporal weight trace.

    Parameters
    ----------
    temporal_weights : np.ndarray, shape (n_components, T)
    frame_rate : float
    n_analyse : int
        Number of top PCs to analyse.

    Returns
    -------
    list of dict, one per PC:
        dominant_freq_hz, dominant_period_s, spectral_entropy, rms_amplitude
    """
    from scipy.signal import welch

    results = []
    for i in range(min(n_analyse, temporal_weights.shape[0])):
        trace = temporal_weights[i]
        T = len(trace)
        nperseg = min(T // 2, max(32, T // 4))

        freqs, psd = welch(trace, fs=frame_rate, nperseg=nperseg,
                           noverlap=nperseg // 2, detrend='linear')

        mask = freqs > 0.005
        if mask.sum() > 0:
            peak_idx = np.argmax(psd[mask])
            dom_freq = float(freqs[mask][peak_idx])
        else:
            dom_freq = 0.0

        psd_norm = psd[mask] / psd[mask].sum() if mask.sum() > 0 and psd[mask].sum() > 0 else np.array([1.0])
        psd_norm = psd_norm[psd_norm > 0]
        entropy = float(-np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd_norm))) if len(psd_norm) > 1 else 0.0

        results.append({
            'pc_index': i + 1,
            'dominant_freq_hz': dom_freq,
            'dominant_period_s': float(1 / dom_freq) if dom_freq > 0 else float('inf'),
            'spectral_entropy': entropy,
            'rms_amplitude': float(np.sqrt(np.mean(trace ** 2))),
            'freqs': freqs,
            'psd': psd,
        })

    return results


# ─── Figures ─────────────────────────────────────────────────────────────

def generate_pca_figures(pca_result, pc_spectra, frame_rate, output_dir,
                         dataset_name='', n_show=5):
    """Generate PCA diagnostic figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    os.makedirs(output_dir, exist_ok=True)
    n_show = min(n_show, pca_result['spatial_maps'].shape[0])
    T = pca_result['temporal_weights'].shape[1]
    time_axis = np.arange(T) / frame_rate

    # ── Figure 1: Spatial maps + temporal weights ─────────────────────────
    fig = plt.figure(figsize=(16, 3 * n_show + 1))
    gs = GridSpec(n_show, 3, width_ratios=[1, 2, 1], hspace=0.35, wspace=0.3)
    fig.suptitle(f'PCA spatial modes — {dataset_name}', fontsize=13, fontweight='bold')

    for i in range(n_show):
        smap = pca_result['spatial_maps'][i]
        tw = pca_result['temporal_weights'][i]
        ve = pca_result['variance_explained'][i]

        # Spatial map
        ax = fig.add_subplot(gs[i, 0])
        vlim = max(abs(smap.min()), abs(smap.max()))
        ax.imshow(smap, cmap='RdBu_r', vmin=-vlim, vmax=vlim, aspect='equal')
        ax.set_title(f'PC{i + 1}  ({ve * 100:.1f}%)', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # Temporal weight
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(time_axis, tw, color='#2B5797', linewidth=0.7)
        ax.set_ylabel('Weight')
        ax.set_xlim(0, time_axis[-1])
        if i == n_show - 1:
            ax.set_xlabel('Time (s)')

        # PSD of temporal weight
        if i < len(pc_spectra):
            ax = fig.add_subplot(gs[i, 2])
            ax.semilogy(pc_spectra[i]['freqs'], pc_spectra[i]['psd'],
                        color='#1D9E75', linewidth=1)
            ax.set_xlim(0, frame_rate / 2)
            ax.set_ylabel('PSD')
            if i == n_show - 1:
                ax.set_xlabel('Freq (Hz)')

    path = os.path.join(output_dir, 'network_pca_modes.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  PCA modes figure saved: {path}")

    # ── Figure 2: Variance explained (scree plot) ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'PCA variance structure — {dataset_name}', fontsize=13, fontweight='bold')

    ve = pca_result['variance_explained']
    n_total = len(ve)

    ax = axes[0]
    ax.bar(range(1, n_total + 1), ve * 100, color='#2B5797', alpha=0.8)
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Variance explained (%)')
    ax.set_title('Individual', fontsize=10)

    ax = axes[1]
    ax.plot(range(1, n_total + 1), np.cumsum(ve) * 100, 'o-', color='#1D9E75', markersize=4)
    ax.axhline(80, color='#E24B4A', linestyle='--', alpha=0.5, label='80%')
    ax.axhline(95, color='#E24B4A', linestyle=':', alpha=0.5, label='95%')
    ax.legend(fontsize=9)
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative variance (%)')
    ax.set_title('Cumulative', fontsize=10)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    path2 = os.path.join(output_dir, 'network_pca_scree.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  PCA scree plot saved: {path2}")

    return [path, path2]


# ─── Main entry point ────────────────────────────────────────────────────

def run_network_pca(movie, frame_rate, output_dir, dataset_name='',
                    n_components=10, downsample_spatial=1, n_show=5):
    """
    Full detection-free PCA spatial decomposition.

    Parameters
    ----------
    movie : np.ndarray, shape (T, Y, X)
        Raw (or motion-corrected) fluorescence movie.
    frame_rate : float
    output_dir : str
    dataset_name : str
    n_components : int
        Number of PCs to extract.
    downsample_spatial : int
        Spatial downsampling factor (2 = half each axis).
    n_show : int
        Number of PCs to visualise in figures.

    Returns
    -------
    result : dict
        pca_result, pc_spectra, summary, figure_paths
    """
    logger.info("=" * 60)
    logger.info("DEV: Network PCA Decomposition (detection-free)")
    logger.info("=" * 60)

    T, Y, X = movie.shape
    logger.info(f"  Movie: {T} frames, {Y}x{X} px, {T / frame_rate:.1f}s at {frame_rate} Hz")

    # Step 1: PCA
    logger.info("  Running PCA decomposition...")
    pca_result = run_movie_pca(movie, n_components, downsample_spatial)

    # Step 2: Spectral analysis of PC temporal weights
    logger.info("  Analysing PC temporal dynamics...")
    pc_spectra = analyse_pc_temporal(pca_result['temporal_weights'], frame_rate, n_show)

    # Step 3: Summary
    summary = {
        'n_components': n_components,
        'downsample_factor': downsample_spatial,
        'variance_explained_top5': [float(v) for v in pca_result['variance_explained'][:5]],
        'variance_explained_cumulative': float(pca_result['variance_explained'].sum()),
        'pc1_var_explained': float(pca_result['variance_explained'][0]),
        'pc_spectra': [
            {k: v for k, v in s.items() if k not in ('freqs', 'psd')}
            for s in pc_spectra
        ],
    }

    # Tissue uniformity indicator
    # If PC1 explains >60%, the tissue behaves as a near-uniform oscillator
    # If PC1 explains <30%, there is significant spatial substructure
    pc1_ve = summary['pc1_var_explained']
    if pc1_ve > 0.6:
        summary['spatial_uniformity'] = 'high'
        summary['interpretation'] = (
            f'PC1 explains {pc1_ve * 100:.0f}% of variance, suggesting the tissue '
            f'behaves as a near-uniform oscillator (most of the FOV activates together).'
        )
    elif pc1_ve > 0.3:
        summary['spatial_uniformity'] = 'moderate'
        summary['interpretation'] = (
            f'PC1 explains {pc1_ve * 100:.0f}% of variance. There is both a global '
            f'component and significant spatial substructure — check PC2+ spatial maps '
            f'for regional differences.'
        )
    else:
        summary['spatial_uniformity'] = 'low'
        summary['interpretation'] = (
            f'PC1 explains only {pc1_ve * 100:.0f}% of variance, indicating substantial '
            f'spatial heterogeneity — different tissue regions have largely independent dynamics.'
        )
    logger.info(f"  Spatial uniformity: {summary['spatial_uniformity']} — {summary['interpretation']}")

    # Step 4: Figures
    logger.info("  Generating figures...")
    dev_dir = os.path.join(output_dir, 'dev_network')
    fig_paths = generate_pca_figures(pca_result, pc_spectra, frame_rate,
                                     dev_dir, dataset_name, n_show)

    # Step 5: Save data
    np.save(os.path.join(dev_dir, 'pca_spatial_maps.npy'),
            pca_result['spatial_maps'].astype(np.float32))
    np.save(os.path.join(dev_dir, 'pca_temporal_weights.npy'),
            pca_result['temporal_weights'].astype(np.float32))
    np.save(os.path.join(dev_dir, 'pca_variance_explained.npy'),
            pca_result['variance_explained'].astype(np.float32))

    with open(os.path.join(dev_dir, 'pca_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  Network PCA complete. Outputs in {dev_dir}/")

    return {
        'pca_result': pca_result,
        'pc_spectra': pc_spectra,
        'summary': summary,
        'figure_paths': fig_paths,
    }
