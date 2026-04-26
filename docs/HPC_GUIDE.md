# Running Calcium Pipeline on a Linux HPC Cluster

## Overview

This guide covers running the Calcium Pipeline on a Linux HPC cluster with
SGE (Sun Grid Engine) job scheduling.  Paths and module commands may need
adapting for your specific cluster.

## Quick Start

```bash
# 1. Upload and extract the pipeline
cd /path/to/calcium_pipeline_workspace
tar -xzf calcium_pipeline.tar.gz

# 2. Process a single recording (submits SGE job)
bash calcium_pipeline/run.sh single --movie /path/to/file.nd2

# 3. Batch process all .nd2 files (submits SGE array)
bash calcium_pipeline/run.sh batch --data-dir /path/to/data

# 4. Group analysis after batch completes
bash calcium_pipeline/run.sh analyse --results-dir /path/to/results
```

## Directory Structure

```
/path/to/calcium_pipeline_workspace/
├── calcium_pipeline/           # Pipeline code
│   ├── src/                    # Python modules
│   ├── config/default.yaml     # Pipeline configuration
│   ├── run.sh                  # Unified entry point
│   ├── run_seeded_v3.sh         # Per-recording pipeline script
│   └── ...
├── data/                       # Your data files
│   ├── recording1.nd2
│   └── ...
├── conda/envs/caiman/          # Conda environment
└── results_*/                  # Output directories
```

## Setup

### 1. Conda Environment

Create or activate a conda environment with the required dependencies:

```bash
# Load anaconda (cluster-specific — check your module system)
module load anaconda
source /path/to/anaconda/etc/profile.d/conda.sh

# Create environment (first time only)
conda create -n caiman python=3.10 numpy scipy scikit-image matplotlib pyyaml tifffile opencv -y
conda activate caiman

# Or activate existing environment
conda activate caiman
```

### 2. Upload Pipeline

```bash
# From your local machine
scp calcium_pipeline.tar.gz <username>@<cluster>:/path/to/calcium_pipeline_workspace/

# On the cluster
cd /path/to/calcium_pipeline_workspace
tar -xzf calcium_pipeline.tar.gz
```

### 3. Upload Data

```bash
# Example: upload ND2 file
scp mydata.nd2 <username>@<cluster>:/path/to/calcium_pipeline_workspace/data/
```

## Running Jobs

### Option 1: Single Recording

```bash
# Submit a single recording as an SGE job
bash calcium_pipeline/run.sh single --movie /path/to/movie.nd2

# With custom config
bash calcium_pipeline/run.sh single --movie /path/to/movie.nd2 --config my_config.yaml
```

### Option 2: Batch Processing

```bash
# Submit SGE job array for all .nd2 files in a directory
bash calcium_pipeline/run.sh batch --data-dir /path/to/data

# Full pipeline: batch + group analysis (analysis held until batch completes)
bash calcium_pipeline/run.sh full --data-dir /path/to/data
```

### Option 3: Interactive Testing

For debugging or quick tests:

```bash
# Start interactive session
qlogin -l h_vmem=32G -pe sharedmem 4

# Run script directly
cd /path/to/calcium_pipeline_workspace
bash calcium_pipeline/run.sh single --movie /path/to/movie.nd2
```

### Key Parameters

All parameters are defined in `config/default.yaml` and can be overridden
with CLI flags.  See `bash run.sh help` for the full list.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--indicator` | fluo4 | Calcium indicator type |
| `--frame-rate` | 2.0 | Acquisition rate (Hz) |
| `--min-radius` | 10 | Minimum neuron radius (pixels) |
| `--max-radius` | 25 | Maximum neuron radius (pixels) |
| `--amplitude-method` | global_dff | Baseline correction method |
| `--deconv-method` | oasis | Spike detection method |

## Resource Recommendations

| Dataset Size | Memory | Cores | Time |
|--------------|--------|-------|------|
| Small (<1000 frames, <512px) | 16G | 4 | 1h |
| Medium (1000-5000 frames) | 32G | 8 | 2-4h |
| Large (>5000 frames) | 48G | 16 | 4-8h |

Modify job resources in the script header or via SGE flags:
```bash
#$ -l h_vmem=48G      # Memory per core
#$ -pe sharedmem 16   # Number of cores
#$ -l h_rt=06:00:00   # Wall time
```

## Output Files

After running, find results in the output directory:

```
<output_dir>/
├── spatial_footprints.npz         # Final neuron footprints (sparse, d1*d2 × N)
├── temporal_traces.npy            # ΔF/F₀ traces (N × T)
├── temporal_traces_raw.npy        # Raw fluorescence traces (N × T)
├── spike_trains.npy               # Inferred spike trains (N × T)
├── confidence_scores.npy          # Per-neuron confidence (0–1)
├── motion_shifts.npy              # Per-frame motion shifts
├── run_info.json                  # Full config, timing, result summary
├── seed_detection_v3.png          # Seed visualisation
├── gallery.html                   # Interactive HTML report
├── figures/                       # Projection images, overlays
└── diagnostics/                   # Detection and processing diagnostics
```

## Monitoring Jobs

```bash
# Check job status
qstat

# Check all your jobs
qstat -u $USER

# View job output in real-time
tail -f logs/calcium_*.o<job_id>

# Check job details
qstat -j <job_id>
```

## Troubleshooting

### Job fails immediately
- Check error log: `cat logs/calcium_*.<job_id>`
- Verify paths are correct
- Ensure conda environment exists

### Out of memory
- Increase `h_vmem`: `#$ -l h_vmem=64G`
- Reduce `max_seeds` in config
- Use chunked processing for very large movies

### Module not found
```bash
# Ensure environment is activated in script
module load anaconda
conda activate caiman
```

### Import errors
```bash
# Reinstall dependencies
conda activate caiman
pip install numpy scipy scikit-image opencv-python matplotlib pyyaml tifffile
```

## Support

- Pipeline issues: Check the diagnostics JSON files in `diagnostics/`
- Cluster issues: Consult your cluster documentation
- Pipeline bugs: <repository>/issues
