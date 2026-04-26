"""
Calcium Pipeline — Development / Experimental Modules
=====================================================

Detection-free network analysis tools for characterising
population-level calcium dynamics without relying on
individual neuron ROI detection.

These modules are activated via environment variables:
    DEV_NETWORK_ANALYSIS=true  bash calcium_pipeline/run.sh single --movie ...

Modules:
    network_spectral    Variance-weighted global trace, Welch PSD,
                        population-level spectral analysis
    network_pca         PCA decomposition of raw movie pixels,
                        spatial mode extraction, tissue-adaptive
"""

__all__ = [
    'network_spectral',
    'network_pca',
]
