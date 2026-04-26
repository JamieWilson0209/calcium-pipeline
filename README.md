# Calcium Imaging Pipeline

A neuron detection and trace extraction pipeline for calcium imaging data,
designed for organoid preparations using 1-photon confocal microscopy.
Currently functional and under active development.

## Features

- **Contour-based ROI detection** — LoG blob detection on multi-projection fusion
  (max + correlation + std), per-blob Otsu contour extraction, circular
  footprint fallback for failed contours
- **Auto-radius estimation** — sweeps candidate radii and selects the one
  producing the most high-SNR traces
- **Motion correction** — CaImAn NoRMCorre rigid or piecewise-rigid
  registration (optional; requires `caiman` package)
- **Flexible baseline correction** — per-trace rolling percentile
  (`global_dff`), tissue-masked local background (`local_background`) for
  organoids, or pass raw traces directly to OASIS (`direct`)
- **OASIS deconvolution** with noise-gated spike inference
- **Multi-format support** — ND2 (Nikon), TIFF, NPY, TIFF image folders
- **Group analysis** — cross-dataset statistics, quality filtering, longitudinal
  developmental trajectory, genotype comparison
- **Single YAML config** — all parameters in `config/default.yaml`, CLI overrides
  supported

## Installation

```bash
git clone https://github.com/JamieWilson0209/calcium-pipeline.git
cd calcium_pipeline

# Core dependencies
pip install numpy scipy pyyaml tifffile scikit-image opencv-python matplotlib

# Optional: motion correction (requires CaImAn)
pip install caiman

# Optional: Nikon ND2 file support
pip install nd2
```

## Quick Start

```bash
# Single recording (submits SGE job on HPC, or runs locally)
bash run.sh single --movie /path/to/file.nd2

# Batch all .nd2 files in a directory (submits SGE array)
bash run.sh batch --data-dir /path/to/data

# Group analysis after batch completes
bash run.sh analyse --results-dir /path/to/batch_results

# Full pipeline (batch + analysis with job dependency)
bash run.sh full --data-dir /path/to/data
```

## Configuration

All parameters live in `config/default.yaml`, organised by pipeline stage:

```yaml
imaging:
  frame_rate: 2.0
  indicator: fluo4              # auto-resolves decay time

motion:
  enabled: true
  mode: rigid                   # rigid | piecewise_rigid | auto
  max_shift: 20

detection:
  min_radius: 10.0
  max_radius: 25.0
  smooth_sigma: 4.0             # hotspot suppression
  auto_radius:
    enabled: true               # sweep and auto-select radii
  contour_fallback: true        # circular footprint when Otsu fails

baseline:
  method: global_dff            # direct | global_dff | local_dff | local_background
  percentile: 8.0

deconvolution:
  enabled: true
  method: oasis
  s_min: 0.1
  noise_gate: 3.5
```

Override per-run with `--config custom.yaml` or individual flags:

```bash
bash run.sh single --movie recording.nd2 --frame-rate 30 --indicator gcamp6f
```

## Pipeline Stages

1. **Load Movie** — ND2, TIFF, NPY, or TIFF image folder
2. **Motion Correction** — CaImAn NoRMCorre rigid/piecewise-rigid registration
3. **Contour-Based Seed Detection** — multi-projection fusion, LoG blob detection,
   per-blob Otsu contour extraction, circular fallback
4. **Trace Extraction** — weighted-average extraction within contour ROIs
5. **Baseline Correction (ΔF/F₀)** — per-trace rolling percentile or local
   tissue-masked background
6. **Deconvolution (OASIS)** — spike inference with noise-gated filtering
7. **Diagnostics** — per-neuron statistics and quality assessment
8. **Save Results** — all outputs saved as NumPy/JSON files
9. **Visual Outputs** — diagnostic figures, optional interactive HTML gallery

## Outputs

```
output/
├── spatial_footprints.npz       # Sparse spatial components (d1*d2 × N)
├── temporal_traces.npy          # ΔF/F₀ traces (N × T)
├── temporal_traces_raw.npy      # Raw fluorescence traces (N × T)
├── traces_denoised.npy          # OASIS denoised traces (N × T)
├── spike_trains.npy             # Inferred spike trains (N × T)
├── confidence_scores.npy        # Per-neuron confidence (0–1)
├── max_projection.npy           # Smoothed max projection
├── max_projection_raw.npy       # Unsmoothed max projection
├── std_projection.npy           # Unsmoothed std projection
├── correlation_image.npy        # Local correlation image
├── mean_projection.npy          # Smoothed mean projection
├── motion_shifts.npy            # Per-frame [dy, dx] motion shifts
├── diagnostics.npz              # Per-neuron statistics
├── run_info.json                # Config, timing, and result summary
├── figures/                     # Projection images, contour overlays
└── gallery.html                 # Interactive HTML report (if --gallery)
```

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| ND2 | .nd2 | Nikon NIS-Elements (multi-channel/Z handled automatically) |
| TIFF | .tif, .tiff | Single or multi-page |
| NumPy | .npy | Pre-loaded arrays |
| Image folder | directory | Folder containing `images/` subfolder of TIFFs |

## Supported Calcium Indicators

Decay times are auto-resolved from the indicator name (full table in
`src/config_loader.py`):

| Indicator | Decay (s) | Notes |
|-----------|-----------|-------|
| GCaMP6f | 0.4 | Fast |
| GCaMP6s | 2.0 | Slow |
| jGCaMP7f | 0.5 | Fast |
| jGCaMP8f | 0.3 | Fastest |
| Fluo-4 | 0.4 | Synthetic dye |
| OGB-1 | 0.7 | Synthetic dye |
| jRGECO1a | 0.7 | Red |

## Programmatic Usage

```python
import sys; sys.path.insert(0, 'src')
from config_loader import load_config
from contour_seed_detection import (
    detect_seeds_with_contours,
    contours_to_spatial_footprints,
    compute_projections_extended,
)
from trace_extraction import extract_traces
from preprocessing import compute_dff_traces

cfg = load_config('config/default.yaml')

# Compute projections once — shared across detection stages
projections = compute_projections_extended(movie, smooth_sigma=cfg.detection.smooth_sigma)

# Detect neurons
seeds = detect_seeds_with_contours(
    movie,
    min_radius=cfg.detection.min_radius,
    max_radius=cfg.detection.max_radius,
    precomputed_projections=projections,
)
A = contours_to_spatial_footprints(seeds, dims=(d1, d2), contour_fallback=True)

# Extract + baseline-correct
C_raw, _ = extract_traces(movie, A)
C_dff, _, info = compute_dff_traces(C_raw, frame_rate=cfg.imaging.frame_rate)
```

## References

- **CaImAn NoRMCorre**: Pnevmatikakis & Giovannucci, J. Neurosci. Methods 2017
- **OASIS**: Friedrich et al., PLoS Comp Biol 2017
- **LoG blob detection**: Lindeberg, IJCV 1998
- **Otsu contour extraction**: Otsu, IEEE 1979
## License

