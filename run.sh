#!/bin/bash
# =============================================================================
# Calcium Pipeline v2.1 — Unified Entry Point
# =============================================================================
#
# Single entry point for all pipeline operations on Linux HPC.
#
# COMMANDS:
#   single    Process a single recording
#   batch     Submit SGE job array for all recordings
#   analyse   Run group analysis on completed results
#   full      Submit batch + analysis (analysis held until batch completes)
#
# Each command works both interactively and via qsub. When submitted with
# qsub, appropriate SGE resources are embedded automatically.
#
# USAGE:
#   # Interactive (runs immediately on current node):
#   bash calcium_pipeline/run.sh single --movie /path/to/file.nd2
#   bash calcium_pipeline/run.sh analyse --results-dir /path/to/results
#
#   # Submit to SGE (queued on compute node):
#   bash calcium_pipeline/run.sh batch --data-dir /path/to/data
#   bash calcium_pipeline/run.sh full --data-dir /path/to/data
#
# All parameters can be set as environment variables or --flags.
# =============================================================================

set +e  # Don't exit on error during setup

# ─────────────────────────────────────────────────────────────────────────────
# RESOLVE PATHS
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${PIPELINE_DIR:-${SCRIPT_DIR}}"
SCRATCH_DIR="${SCRATCH_DIR:-$(pwd)}"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Redirect caches to scratch (home quota is limited on HPC clusters)
export MPLCONFIGDIR="${SCRATCH_DIR}/.cache/matplotlib"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
mkdir -p "${MPLCONFIGDIR}" 2>/dev/null || true

# Paths
DATA_DIR="${DATA_DIR:-${SCRATCH_DIR}/data}"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRATCH_DIR}/results}"
CONDA_ENV="${CONDA_ENV:-caiman}"

# YAML configuration — single source of truth for all pipeline parameters.
# Override per-run with --config /path/to/custom.yaml
CONFIG_PATH="${CONFIG_PATH:-${PIPELINE_DIR}/config/default.yaml}"

# Batch file pattern (shell-level concern — not a pipeline parameter)
FILE_PATTERN="${FILE_PATTERN:-*.nd2}"

# ─────────────────────────────────────────────────────────────────────────────
# PARSE COMMAND AND FLAGS
# ─────────────────────────────────────────────────────────────────────────────

COMMAND="${1:-}"
shift 2>/dev/null || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --movie)             MOVIE="$2";              shift 2 ;;
        --output)            OUTPUT_DIR="$2";         shift 2 ;;
        --config)            CONFIG_PATH="$2";        shift 2 ;;
        --results-dir)       RESULTS_DIR="$2";        shift 2 ;;
        --data-dir)          DATA_DIR="$2";           shift 2 ;;
        --output-base)       OUTPUT_BASE="$2";        shift 2 ;;
        --inactive-file)     INACTIVE_FILE="$2";      shift 2 ;;
        --help|-h)           COMMAND="help";          shift   ;;
        *)                   echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Resolve relative paths to absolute
[ -n "${DATA_DIR:-}" ]    && DATA_DIR="$(cd "${DATA_DIR}" 2>/dev/null && pwd || echo "${DATA_DIR}")"
[ -n "${MOVIE:-}" ]       && MOVIE="$(cd "$(dirname "${MOVIE}")" 2>/dev/null && pwd)/$(basename "${MOVIE}")" 2>/dev/null || true
[ -n "${OUTPUT_BASE:-}" ] && OUTPUT_BASE="$(mkdir -p "${OUTPUT_BASE}" 2>/dev/null; cd "${OUTPUT_BASE}" 2>/dev/null && pwd || echo "${OUTPUT_BASE}")"
[ -n "${CONFIG_PATH:-}" ] && CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" 2>/dev/null && pwd)/$(basename "${CONFIG_PATH}")" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# HELP
# ─────────────────────────────────────────────────────────────────────────────

show_help() {
    cat << 'EOF'
Calcium Pipeline v2.1

COMMANDS:
  single    Submit a single .nd2 recording as an SGE job
  batch     Submit SGE job array for all recordings in DATA_DIR
  analyse   Run group analysis on completed batch results
  full      Submit batch + analysis (analysis auto-runs after batch)

EXAMPLES:
  # Single file:
  bash calcium_pipeline/run.sh single --movie /path/to/file.nd2

  # Batch all .nd2 files (submits SGE array):
  bash calcium_pipeline/run.sh batch --data-dir /path/to/data

  # Group analysis (submits SGE job):
  bash calcium_pipeline/run.sh analyse --results-dir /path/to/results

  # Full pipeline (batch + analysis):
  bash calcium_pipeline/run.sh full --data-dir /path/to/data

  # Custom YAML config:
  bash calcium_pipeline/run.sh single --movie file.nd2 --config my_config.yaml

FLAGS:
  --movie PATH        Single .nd2 file (required for 'single')
  --data-dir DIR      Directory with .nd2 files (required for 'batch'/'full')
  --results-dir DIR   Batch results directory (required for 'analyse')
  --output DIR        Output directory (default: $OUTPUT_BASE/<filestem>)
  --config PATH       YAML config (default: config/default.yaml)
  --inactive-file F   List of inactive dataset names (for group analysis)

All pipeline parameters are defined in config/default.yaml.  Override
per-run by copying the YAML and editing, or by passing individual CLI
flags to the pipeline.

ENVIRONMENT VARIABLES (shell-level only):
  PIPELINE_DIR        Path to the calcium_pipeline/ directory
  SCRATCH_DIR         Scratch filesystem for results + temp files
  DATA_DIR            Input data directory
  OUTPUT_BASE         Output directory base
  CONDA_ENV           Conda environment path
  CONFIG_PATH         YAML config path (alternative to --config)
  FILE_PATTERN        Glob pattern for batch mode (default: *.nd2)
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# SHARED: conda environment setup
# ─────────────────────────────────────────────────────────────────────────────

setup_conda() {
    set +e
    source /etc/profile.d/modules.sh 2>/dev/null || true
    if command -v module &>/dev/null; then
        module load anaconda 2>/dev/null || \
        module load anaconda/2024.02 2>/dev/null || true
        if [ -f ${CONDA_PREFIX:-/opt/conda}/etc/profile.d/conda.sh ]; then
            source ${CONDA_PREFIX:-/opt/conda}/etc/profile.d/conda.sh
        fi
    fi
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    fi
    conda activate "${CONDA_ENV}" 2>/dev/null || \
        conda activate caiman 2>/dev/null || true
    set -e
}

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND: single
# ─────────────────────────────────────────────────────────────────────────────

cmd_single() {
    MOVIE="${MOVIE:-}"
    if [ -z "${MOVIE}" ]; then
        echo "ERROR: No movie specified. Use --movie /path/to/file.nd2"
        exit 1
    fi
    if [ ! -f "${MOVIE}" ]; then
        echo "ERROR: File not found: ${MOVIE}"
        exit 1
    fi
    if [ ! -f "${CONFIG_PATH}" ]; then
        echo "ERROR: Config file not found: ${CONFIG_PATH}"
        echo "  Use --config /path/to/config.yaml or place config at default location"
        exit 1
    fi

    FILESTEM="$(basename "${MOVIE}" .nd2)"
    OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${FILESTEM}}"

    echo "================================================================"
    echo "Calcium Pipeline — Single Recording"
    echo "================================================================"
    echo "  Movie:  ${MOVIE}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "  Config: ${CONFIG_PATH}"
    echo "================================================================"

    # Generate SGE job script with baked-in absolute paths
    LOG_DIR="${SCRATCH_DIR}/logs"
    mkdir -p "${LOG_DIR}"
    JOB_SCRIPT_DIR="${SCRATCH_DIR}/.job_scripts"
    mkdir -p "${JOB_SCRIPT_DIR}"
    JOB_SCRIPT="${JOB_SCRIPT_DIR}/single_$(date +%Y%m%d_%H%M%S)_$$.sh"
    cat > "${JOB_SCRIPT}" << SINGLE_EOF
#!/bin/bash
#\$ -N calcium_single
#\$ -l h_rt=06:00:00
#\$ -l h_vmem=32G
#\$ -j y
#\$ -o ${LOG_DIR}
#\$ -V

echo "================================================================"
echo "Calcium Pipeline — Single Recording"
echo "================================================================"
echo "Job: \${JOB_ID}  Host: \$(hostname)  Time: \$(date)"
echo "Movie:  ${MOVIE}"
echo "Output: ${OUTPUT_DIR}"
echo "Config: ${CONFIG_PATH}"
echo "================================================================"

# ── Paths (resolved at submit time) ──
PIPELINE_DIR="${PIPELINE_DIR}"
CONDA_ENV="${CONDA_ENV}"
MOVIE="${MOVIE}"
OUTPUT_DIR="${OUTPUT_DIR}"
CONFIG_PATH="${CONFIG_PATH}"

# ── Conda ──
set +e
source /etc/profile.d/modules.sh 2>/dev/null || true
module load anaconda 2>/dev/null || true
[ -f ${CONDA_PREFIX:-/opt/conda}/etc/profile.d/conda.sh ] && \\
    source ${CONDA_PREFIX:-/opt/conda}/etc/profile.d/conda.sh
[ -f ~/miniconda3/etc/profile.d/conda.sh ] && \\
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate "\${CONDA_ENV}" 2>/dev/null || conda activate caiman 2>/dev/null || true
set -e

echo "Python: \$(which python)"
mkdir -p "\${OUTPUT_DIR}"

# ── Run pipeline ──
cd "\${PIPELINE_DIR}"
python -m src.run_full_pipeline \\
    --movie "\${MOVIE}" \\
    --output "\${OUTPUT_DIR}" \\
    --config "\${CONFIG_PATH}"
EXIT_CODE=\$?

echo ""
if [ -f "\${OUTPUT_DIR}/spatial_footprints.npz" ]; then
    N_NEURONS=\$(python -c "from scipy.sparse import load_npz; print(load_npz('\${OUTPUT_DIR}/spatial_footprints.npz').shape[1])" 2>/dev/null || echo "?")
    echo "DONE: \${MOVIE}  (\${N_NEURONS} neurons)"
else
    echo "FAILED: \${MOVIE}"
fi
echo "Finished: \$(date)"
exit \${EXIT_CODE}
SINGLE_EOF

    chmod +x "${JOB_SCRIPT}"

    SINGLE_JOB_ID=$(qsub -terse "${JOB_SCRIPT}")
    echo ""
    echo "Submitted: job ${SINGLE_JOB_ID}"
    echo "  Log: ${LOG_DIR}/calcium_single.o${SINGLE_JOB_ID}"
    echo "  Monitor: qstat -j ${SINGLE_JOB_ID}"
}

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND: batch — generates and submits a proper SGE array job
# ─────────────────────────────────────────────────────────────────────────────

cmd_batch() {
    if [ ! -d "${DATA_DIR}" ]; then
        echo "ERROR: DATA_DIR not found: ${DATA_DIR}"
        echo "  Use --data-dir /path/to/data or set DATA_DIR"
        exit 1
    fi

    mapfile -t FILES < <(find "${DATA_DIR}" -name "${FILE_PATTERN}" -type f | sort)
    N_FILES=${#FILES[@]}

    if [ "${N_FILES}" -eq 0 ]; then
        echo "ERROR: No files matching '${FILE_PATTERN}' in ${DATA_DIR}"
        exit 1
    fi

    echo "================================================================"
    echo "Calcium Pipeline — Batch Submit"
    echo "================================================================"
    echo "  DATA_DIR:    ${DATA_DIR}"
    echo "  FILES:       ${N_FILES}"
    echo "  CONFIG:      ${CONFIG_PATH}"
    echo "  OUTPUT_BASE: ${OUTPUT_BASE}"
    echo "================================================================"
    echo ""

    # Show breakdown
    for dir in "${DATA_DIR}"/*/; do
        [ -d "${dir}" ] || continue
        n=$(find "${dir}" -name "${FILE_PATTERN}" -type f | wc -l)
        [ "${n}" -gt 0 ] && echo "  $(basename "${dir}"): ${n} recordings"
    done
    echo "  Total: ${N_FILES}"
    echo ""

    # Generate the SGE array job script on shared filesystem
    LOG_DIR="${SCRATCH_DIR}/logs"
    mkdir -p "${LOG_DIR}"
    JOB_SCRIPT_DIR="${SCRATCH_DIR}/.job_scripts"
    mkdir -p "${JOB_SCRIPT_DIR}"
    JOB_SCRIPT="${JOB_SCRIPT_DIR}/batch_$(date +%Y%m%d_%H%M%S)_$$.sh"
    cat > "${JOB_SCRIPT}" << BATCH_HEADER
#!/bin/bash
#\$ -N calcium_batch
#\$ -l h_rt=06:00:00
#\$ -l h_vmem=32G
#\$ -j y
#\$ -o ${LOG_DIR}
#\$ -V
BATCH_HEADER

    # Embed the resolved configuration (not escaped — values baked in)
    cat >> "${JOB_SCRIPT}" << BATCH_CONFIG

# ── Paths (resolved at submit time) ──
PIPELINE_DIR="${PIPELINE_DIR}"
DATA_DIR="${DATA_DIR}"
OUTPUT_BASE="${OUTPUT_BASE}"
FILE_PATTERN="${FILE_PATTERN}"
CONDA_ENV="${CONDA_ENV}"
CONFIG_PATH="${CONFIG_PATH}"
BATCH_CONFIG

    # Embed the task execution logic
    cat >> "${JOB_SCRIPT}" << 'BATCH_BODY'

# ── Resolve file for this task ──
TASK_ID="${SGE_TASK_ID:-1}"
mapfile -t FILES < <(find "${DATA_DIR}" -name "${FILE_PATTERN}" -type f | sort)
N_FILES=${#FILES[@]}

if [ "${TASK_ID}" -gt "${N_FILES}" ]; then
    echo "Task ${TASK_ID} exceeds file count (${N_FILES}) — skipping"
    exit 0
fi

FILE="${FILES[$((TASK_ID - 1))]}"
FILESTEM="$(basename "${FILE}" .nd2)"
JOB_TAG="${JOB_ID:-local}"
FILE_OUTPUT="${OUTPUT_BASE}/run_${JOB_TAG}/${FILESTEM}"

echo "================================================================"
echo "Calcium Pipeline — Task ${TASK_ID}/${N_FILES}"
echo "================================================================"
echo "  File:   ${FILE}"
echo "  Output: ${FILE_OUTPUT}"
echo "  Config: ${CONFIG_PATH}"
echo "  Host:   $(hostname)"
echo "  Time:   $(date)"
echo "================================================================"

# ── Conda ──
set +e
source /etc/profile.d/modules.sh 2>/dev/null || true
module load anaconda 2>/dev/null || true
[ -f ${CONDA_PREFIX:-/opt/conda}/etc/profile.d/conda.sh ] && \
    source ${CONDA_PREFIX:-/opt/conda}/etc/profile.d/conda.sh
[ -f ~/miniconda3/etc/profile.d/conda.sh ] && \
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate "${CONDA_ENV}" 2>/dev/null || conda activate caiman 2>/dev/null || true
set -e

echo "Python: $(which python)"
mkdir -p "${FILE_OUTPUT}"

# ── Run pipeline ──
cd "${PIPELINE_DIR}"
python -m src.run_full_pipeline \
    --movie "${FILE}" \
    --output "${FILE_OUTPUT}" \
    --config "${CONFIG_PATH}"
EXIT_CODE=$?

echo ""
if [ -f "${FILE_OUTPUT}/spatial_footprints.npz" ]; then
    N_NEURONS=$(python -c "from scipy.sparse import load_npz; print(load_npz('${FILE_OUTPUT}/spatial_footprints.npz').shape[1])" 2>/dev/null || echo "?")
    echo "Task ${TASK_ID} DONE: ${FILESTEM}  (${N_NEURONS} neurons)"
else
    echo "Task ${TASK_ID} FAILED: ${FILESTEM}"
fi
echo "Finished: $(date)"
exit ${EXIT_CODE}
BATCH_BODY

    chmod +x "${JOB_SCRIPT}"

    # Submit
    BATCH_JOB_ID=$(qsub -t 1-${N_FILES} -terse "${JOB_SCRIPT}")
    BATCH_JOB_ID="${BATCH_JOB_ID%%.*}"

    echo "Submitted: job ${BATCH_JOB_ID} (${N_FILES} tasks)"
    echo "Results:   ${OUTPUT_BASE}/run_${BATCH_JOB_ID}/"
    echo "Logs:      ${LOG_DIR}/"
    echo ""
    echo "Monitor:   qstat -j ${BATCH_JOB_ID}"
    echo "Progress:  ls ${OUTPUT_BASE}/run_${BATCH_JOB_ID}/*/temporal_traces.npy 2>/dev/null | wc -l"

    # Export for full command
    export _BATCH_JOB_ID="${BATCH_JOB_ID}"
    export _BATCH_RESULTS="${OUTPUT_BASE}/run_${BATCH_JOB_ID}"
}

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND: analyse — generates and submits a proper SGE job
# ─────────────────────────────────────────────────────────────────────────────

cmd_analyse() {
    RESULTS_DIR="${RESULTS_DIR:-}"
    if [ -z "${RESULTS_DIR}" ]; then
        echo "ERROR: No results directory. Use --results-dir /path/to/results"
        exit 1
    fi
    if [ ! -d "${RESULTS_DIR}" ]; then
        echo "ERROR: Not found: ${RESULTS_DIR}"
        exit 1
    fi
    # Ensure absolute paths — relative paths break on compute nodes
    RESULTS_DIR="$(cd "${RESULTS_DIR}" && pwd)"

    ANALYSIS_OUTPUT="${OUTPUT_DIR:-${RESULTS_DIR}/analysis}"
    # Ensure absolute path — relative paths break on compute nodes
    ANALYSIS_OUTPUT="$(cd "$(dirname "${ANALYSIS_OUTPUT}")" 2>/dev/null && pwd)/$(basename "${ANALYSIS_OUTPUT}")" \
        || ANALYSIS_OUTPUT="$(pwd)/${ANALYSIS_OUTPUT}"

    N_DATASETS=$(find "${RESULTS_DIR}" -maxdepth 2 -name "temporal_traces.npy" 2>/dev/null | wc -l)
    echo "================================================================"
    echo "Calcium Pipeline — Group Analysis"
    echo "================================================================"
    echo "  Results:  ${RESULTS_DIR}"
    echo "  Output:   ${ANALYSIS_OUTPUT}"
    echo "  Config:   ${CONFIG_PATH}"
    echo "  Datasets: ${N_DATASETS}"
    echo "================================================================"

    if [ "${N_DATASETS}" -eq 0 ] && [ -z "${_HOLD_JID:-}" ]; then
        echo "ERROR: No temporal_traces.npy found in ${RESULTS_DIR}"
        exit 1
    elif [ "${N_DATASETS}" -eq 0 ]; then
        echo "  (batch not yet complete — analysis will run after batch finishes)"
    fi

    # If already on a compute node (inside SGE), run directly
    if [ -n "${JOB_ID:-}" ] && [[ "$(hostname)" != login* ]]; then
        echo "Running on compute node — executing directly"
        setup_conda
        mkdir -p "${ANALYSIS_OUTPUT}"
        cd "${PIPELINE_DIR}"
        python -m src.group_analysis \
            --results-dir "${RESULTS_DIR}" \
            --output "${ANALYSIS_OUTPUT}" \
            --config "${CONFIG_PATH}" \
            ${INACTIVE_FILE:+--inactive-file ${INACTIVE_FILE}}

        # Dev network aggregation (if enabled in config)
        DEV_NET=$(python -c "import sys; sys.path.insert(0,'src'); from config_loader import load_config; print(load_config('${CONFIG_PATH}').dev.network_analysis)")
        if [ "${DEV_NET}" = "True" ]; then
            echo ""
            echo "Running dev network analysis aggregation..."
            python -m src.dev.network_aggregate \
                --results-dir "${RESULTS_DIR}" \
                --output "${ANALYSIS_OUTPUT}/dev_network" \
                --config "${CONFIG_PATH}" || \
                echo "  (dev network aggregation skipped — no dev_network/ results found)"
        fi
        echo ""
        echo "Analysis complete. Output: ${ANALYSIS_OUTPUT}"
        return
    fi

    # On login node — generate and submit SGE job on shared filesystem
    echo "Submitting SGE job..."

    JOB_SCRIPT_DIR="${SCRATCH_DIR}/.job_scripts"
    mkdir -p "${JOB_SCRIPT_DIR}"
    LOG_DIR="${SCRATCH_DIR}/logs"
    mkdir -p "${LOG_DIR}"
    JOB_SCRIPT="${JOB_SCRIPT_DIR}/analyse_$(date +%Y%m%d_%H%M%S)_$$.sh"
    cat > "${JOB_SCRIPT}" << ANALYSE_EOF
#!/bin/bash
#\$ -N calcium_analysis
#\$ -l h_rt=02:00:00
#\$ -l h_vmem=32G
#\$ -j y
#\$ -o ${LOG_DIR}
#\$ -V

echo "================================================================"
echo "Calcium Pipeline — Group Analysis"
echo "================================================================"
echo "Job: \${JOB_ID}  Host: \$(hostname)  Time: \$(date)"
echo "Results: ${RESULTS_DIR}"
echo "Output:  ${ANALYSIS_OUTPUT}"
echo "Config:  ${CONFIG_PATH}"
echo "================================================================"

# Conda
set +e
source /etc/profile.d/modules.sh 2>/dev/null || true
module load anaconda 2>/dev/null || true
[ -f ${CONDA_PREFIX:-/opt/conda}/etc/profile.d/conda.sh ] && \\
    source ${CONDA_PREFIX:-/opt/conda}/etc/profile.d/conda.sh
[ -f ~/miniconda3/etc/profile.d/conda.sh ] && \\
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
# Try activate by path first, then by name
conda activate "${CONDA_ENV}" 2>/dev/null || \\
    conda activate caiman 2>/dev/null || \\
    echo "WARNING: conda activation failed — using system Python"
set -e

echo "Python: \$(which python)"
echo "Conda env: \${CONDA_DEFAULT_ENV:-not set}"
mkdir -p "${ANALYSIS_OUTPUT}"

cd "${PIPELINE_DIR}"
python -m src.group_analysis \\
    --results-dir "${RESULTS_DIR}" \\
    --output "${ANALYSIS_OUTPUT}" \\
    --config "${CONFIG_PATH}" \\
    ${INACTIVE_FILE:+--inactive-file ${INACTIVE_FILE}}

# Dev network aggregation (if enabled in config)
DEV_NET=\$(python -c "import sys; sys.path.insert(0,'src'); from config_loader import load_config; print(load_config('${CONFIG_PATH}').dev.network_analysis)")
if [ "\${DEV_NET}" = "True" ]; then
    echo ""
    echo "Running dev network analysis aggregation..."
    python -m src.dev.network_aggregate \\
        --results-dir "${RESULTS_DIR}" \\
        --output "${ANALYSIS_OUTPUT}/dev_network" \\
        --config "${CONFIG_PATH}" || \\
        echo "  (dev network aggregation skipped)"
fi

echo ""
echo "================================================================"
echo "Analysis complete"
echo "================================================================"
echo "  Output:  ${ANALYSIS_OUTPUT}"
echo "  Figures: ${ANALYSIS_OUTPUT}/figures/"
echo "  Time:    \$(date)"
echo "================================================================"
ANALYSE_EOF

    chmod +x "${JOB_SCRIPT}"

    HOLD_ARG=""
    if [ -n "${_HOLD_JID:-}" ]; then
        HOLD_ARG="-hold_jid ${_HOLD_JID}"
    fi

    ANALYSIS_JOB_ID=$(qsub ${HOLD_ARG} -terse "${JOB_SCRIPT}")
    echo ""
    echo "Submitted: job ${ANALYSIS_JOB_ID}"
    if [ -n "${_HOLD_JID:-}" ]; then
        echo "  Held on: ${_HOLD_JID}"
    fi
    echo "  Log: ${LOG_DIR}/calcium_analysis.o${ANALYSIS_JOB_ID}"
    echo "  Monitor: qstat -j ${ANALYSIS_JOB_ID}"

    export _ANALYSIS_JOB_ID="${ANALYSIS_JOB_ID}"
}

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND: full — batch + analysis
# ─────────────────────────────────────────────────────────────────────────────

cmd_full() {
    if [ ! -d "${DATA_DIR}" ]; then
        echo "ERROR: DATA_DIR not found: ${DATA_DIR}"
        echo "  Use --data-dir /path/to/data or set DATA_DIR"
        exit 1
    fi

    echo "================================================================"
    echo "Calcium Pipeline v2.1 — Full Pipeline"
    echo "================================================================"
    echo ""

    # Step 1: submit batch
    cmd_batch

    # Step 2: submit analysis held on batch
    RESULTS_DIR="${_BATCH_RESULTS}"
    _HOLD_JID="${_BATCH_JOB_ID}"
    OUTPUT_DIR="${_BATCH_RESULTS}/analysis"
    ANALYSIS_OUTPUT="${OUTPUT_DIR}"

    # Pre-create the results directory so cmd_analyse doesn't fail the
    # existence check — the batch tasks will populate it once they run.
    mkdir -p "${RESULTS_DIR}"

    echo ""
    echo "================================================================"
    echo "Step 2: Submitting group analysis (held on batch)"
    echo "================================================================"

    cmd_analyse

    echo ""
    echo "================================================================"
    echo "All jobs submitted"
    echo "================================================================"
    echo "  Batch:    ${_BATCH_JOB_ID}"
    echo "  Analysis: ${_ANALYSIS_JOB_ID}"
    echo "  Results:  ${_BATCH_RESULTS}/"
    echo "================================================================"
}

# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────

case "${COMMAND}" in
    single)        cmd_single   ;;
    batch)         cmd_batch    ;;
    analyse|analyze) cmd_analyse  ;;
    full)          cmd_full     ;;
    help|-h)       show_help    ;;
    "")
        echo "ERROR: No command specified."
        echo ""
        show_help
        exit 1
        ;;
    *)
        echo "ERROR: Unknown command '${COMMAND}'"
        echo ""
        show_help
        exit 1
        ;;
esac
