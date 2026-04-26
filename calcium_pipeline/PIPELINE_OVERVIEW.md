# Calcium Imaging Pipeline — How It Works

A step-by-step guide to what happens when you submit a batch job, written for anyone working with the data regardless of programming background.

---

## The Big Picture

The pipeline takes raw microscopy videos of brain organoids and extracts meaningful information about which neurons are active, when they fire, and how they relate to each other. It does this in two phases:

1. **Per-recording processing** — each video is analysed independently to find neurons and measure their activity
2. **Group comparison** — results from all recordings are brought together to compare datasets side by side

---

## Phase 1: Per-Recording Processing

When you submit `bash run.sh batch --data-dir /path/to/data`, the HPC cluster launches one independent job per recording. Each job runs through the following stages:

### Stage 1 — Load the Video

The pipeline reads the raw microscopy file and converts it into a NumPy array of pixel intensities over time. Supported formats include ND2 (Nikon), TIFF, NPY, and folders of TIFF images. It logs basic properties like the number of frames and image dimensions.

### Stage 2 — Motion Correction

Living tissue moves during recording — the organoid drifts, pulses, or shifts slightly between frames. If left uncorrected, a neuron's signal would be smeared across different pixel locations. Motion correction aligns every frame to a reference so that the same neuron sits in the same place throughout.

The pipeline uses **rigid registration** (shifting the whole image, no warping) and records how much each frame had to move. Recordings with excessive motion (>15 pixels max shift or >2 pixels residual jitter) are flagged as poor quality and excluded from group analysis later.

### Stage 3 — Find the Neurons (Seed Detection)

The pipeline locates neurons by looking for round, bright regions in summary projection images.

First, it computes projections from the video: max projection (brightest moment per pixel), std projection (temporal variability), and local correlation (how much neighbouring pixels co-fluctuate). These are optionally smoothed to suppress small bright artefacts (hotspots) while preserving cell-body-sized structures.

Laplacian-of-Gaussian (LoG) blob detection then finds circular features across multiple spatial scales. Each candidate blob is filtered by border distance, intensity, local contrast, and whether it's a local maximum. Surviving seeds are passed to a contour extraction step — for each seed, a Gaussian-weighted locality mask isolates the region around that cell, and Otsu thresholding separates cell from background to produce an irregular contour. Seeds where contour extraction fails fall back to a circular footprint.

### Stage 4 — Extract Activity Traces & Baseline Correction

For each detected neuron, the pipeline extracts a fluorescence time series by computing the weighted average of pixel intensities within its contour across all frames.

The raw traces are then baseline-corrected to produce ΔF/F₀ values. The default method (`global_dff`) estimates each trace's baseline as a rolling low percentile and divides it out. Alternative methods include tissue-masked local background subtraction (for organoid data where neuropil assumptions don't apply) or passing raw traces directly to the deconvolution stage.

After baseline correction, a population drift removal step subtracts slow recording-wide artefacts (focus shifts, illumination changes) that affect all ROIs simultaneously. The population median trace is smoothed with a ~10-second rolling window and subtracted from each individual trace, preserving fast synchronous calcium events while removing slow drift.

### Stage 5 — Spike Detection (Deconvolution)

Calcium indicators are slow — when a neuron fires an action potential, the fluorescence rises quickly but decays over hundreds of milliseconds. The raw trace is a blurred version of the true spiking activity.

**OASIS deconvolution** (the default method) works backwards from the blurred calcium signal to estimate when the neuron actually fired. It models the expected shape of a calcium transient and finds the most likely set of spike times that would produce the observed trace.

Post-processing then cleans up the results: merging events that are unrealistically close together, removing tiny events that are likely noise, snapping spike times to the actual peak of each transient, and suppressing artefacts at recording boundaries.

The output is a **spike train** for each neuron — a sparse signal that is zero most of the time, with non-zero values at the moments the neuron fired.

### Stage 6 — Diagnostics

The pipeline computes per-neuron diagnostics including transient counts, signal-to-noise ratio, and baseline stability. These are saved alongside the main outputs for inspection.

### Stage 7 — Save Everything

All results are saved as standard NumPy files:

- `temporal_traces.npy` — ΔF/F₀ traces (neurons × time)
- `traces_denoised.npy` — OASIS denoised traces
- `spike_trains.npy` — inferred spike events
- `spatial_footprints.npz` — where each neuron is located in the image
- `confidence_scores.npy` — detection confidence per neuron
- `max_projection_raw.npy` — unsmoothed max projection
- `std_projection.npy` — std projection
- `correlation_image.npy` — local correlation image
- `diagnostics.npz` — per-neuron statistics
- `run_info.json` — recording parameters and processing metadata

An interactive HTML gallery can also be generated for manual inspection (if `--gallery` is enabled).

---

## Phase 2: Group Comparison

After all recordings are processed, you run `bash run.sh analyse --results-dir /path/to/results`. This brings everything together.

### Load and Filter Datasets

The analysis scans all per-recording output directories and applies quality gating. Recordings are excluded if they exceed any of three thresholds: maximum motion shift (>15 pixels), residual motion jitter (>2 pixels std), or excessive baseline drift (>1.0 drift ratio). Remaining datasets proceed to comparison.

### Neuron Selection

For each dataset, the pipeline selects neurons in two steps:

1. **Deconvolution gating** — only ROIs with valid deconvolved events are included. The OASIS deconvolution stage already filters by minimum spike size (`s_min`), noise gate (3.5σ), and transient duration. ROIs that fail deconvolution have their spike trains zeroed and are excluded here.

2. **Distance deduplication** — ROIs with centroids closer than 15 pixels apart are likely detecting the same cell twice. The higher-SNR duplicate is kept, the other is removed.

All surviving neurons are used for analysis — there is no arbitrary top-N cutoff or composite quality score.

### Compute Population Metrics

For each dataset's selected neurons, the pipeline calculates:

- **Spike rate** — how often neurons fire (events per 10 seconds)
- **Spike amplitude** — mean transient size (ΔF/F₀)
- **Active fraction** — proportion of detected neurons with at least one validated transient
- **Pairwise correlation** — mean correlation between neuron pairs' denoised calcium traces. Requires at least 5 selected neurons; datasets with fewer are excluded from this metric.
- **Synchrony index** — composite measure of population coordination, combining population coupling with co-activation frequency
- **Inter-event interval (IEI)** — mean time between spike events and its coefficient of variation (regularity of firing)
- **Network bursts** — periods where a threshold fraction of neurons are simultaneously active, with burst rate and mean participation

### Generate Figures

All figures are saved in an organised structure:

**`1 - Main Results/`** — feature heatmap across all datasets

**`1b - Metrics/`** — individual metric bar charts (spike rate, amplitude, correlation, synchrony, etc.)

**`2 - Genotype Comparison/`** — if genotype information is available: raincloud plots, longitudinal trajectories, within-day comparisons, meta-analysis forest plots, and scatter diagnostics

**`Correlation Graphs/`** — neuron-by-neuron correlation matrix for each dataset

**`Temporal Visualisations/`** — spike raster plots for each dataset

**`Full Overview/`** — population activity summary, neuron distributions, combined bar charts

**`Results by Dataset/`** — per-dataset detail: raster, correlation matrix, and traces

### Save Summary Data

- `dataset_features.csv` — the full feature matrix (every metric for every dataset) in spreadsheet-compatible format
- `analysis_results.json` — complete results including per-dataset metrics, neuron selection details, and quality criteria
- `motion_quality.json` — which datasets were included/excluded and why

---

## Key Parameters You Can Change

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `DECONV_METHOD` | oasis | Spike detection algorithm |
| `FRAME_RATE` | 2.0 | Recording frame rate in Hz |
| `MIN_RADIUS` / `MAX_RADIUS` | 10 / 25 | Expected neuron size range in pixels |
| `MIN_ROI_DISTANCE` | 15 | Minimum distance (pixels) between ROI centroids for deduplication |

All parameters can be set in `config/default.yaml` or as CLI flags.
