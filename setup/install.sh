#!/usr/bin/env bash
# =============================================================================
# install.sh — one-step environment setup for the calcium pipeline
# =============================================================================
#
# Creates a fresh conda env named `calpipe` from environment.yml.  CaImAn
# is installed via conda-forge as part of the yml (the PyPI package
# called "caiman" is a totally unrelated MicroPython build tool — never
# `pip install caiman`).
#
# Usage:
#   bash install.sh                  # create env, install everything
#   bash install.sh --force          # delete and recreate existing env
#
# Detects mamba and uses it if available (much faster than conda for
# the initial create step).
#
# After this finishes:
#   conda activate calpipe
#   bash run.sh single --movie /path/to/file.nd2
# =============================================================================

set -euo pipefail

ENV_NAME="calpipe"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YML_PATH="${SCRIPT_DIR}/environment.yml"
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --force)  FORCE=1 ;;
        -h|--help)
            sed -n '3,18p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)  echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# --- pick a solver ------------------------------------------------------------
if command -v mamba >/dev/null 2>&1; then
    SOLVER="mamba"
elif command -v conda >/dev/null 2>&1; then
    SOLVER="conda"
else
    echo "ERROR: neither mamba nor conda found on PATH." >&2
    echo "Install Miniconda/Miniforge first: https://github.com/conda-forge/miniforge" >&2
    exit 1
fi
echo "[install] using solver: ${SOLVER}"

# --- check / remove existing env ---------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [ "${FORCE}" -eq 1 ]; then
        echo "[install] removing existing '${ENV_NAME}' env (--force)..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "ERROR: env '${ENV_NAME}' already exists." >&2
        echo "Re-run with --force to delete and recreate, or activate it directly:" >&2
        echo "    conda activate ${ENV_NAME}" >&2
        exit 1
    fi
fi

# --- create env from yml ------------------------------------------------------
echo "[install] creating env '${ENV_NAME}' from ${YML_PATH}..."
"${SOLVER}" env create -f "${YML_PATH}" -n "${ENV_NAME}"

# --- smoke test ---------------------------------------------------------------
ENV_PYTHON="$(conda env list | awk -v n="${ENV_NAME}" '$1==n {print $NF}')/bin/python"
if [ ! -x "${ENV_PYTHON}" ]; then
    echo "ERROR: cannot locate python for new env at ${ENV_PYTHON}" >&2
    exit 1
fi
echo "[install] verifying installed packages..."
"${ENV_PYTHON}" - <<'PY'
import sys
import importlib.util
required = ['numpy', 'scipy', 'skimage', 'sklearn', 'cv2',
            'matplotlib', 'tifffile', 'yaml', 'PIL',
            'pandas', 'statsmodels', 'joblib', 'nd2', 'caiman']
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print(f"ERROR: missing modules after install: {missing}", file=sys.stderr)
    sys.exit(1)
# Sanity-check that we got the right caiman (calcium imaging, not the
# MicroPython build tool that shares the name on PyPI).
import caiman
if not hasattr(caiman, 'motion_correction'):
    print(f"ERROR: wrong 'caiman' package installed at {caiman.__file__}", file=sys.stderr)
    print("       (looks like the MicroPython tool, not the calcium imaging one)", file=sys.stderr)
    sys.exit(1)
print(f"[install] caiman OK: {caiman.__file__}")
print("[install] all required packages present.")
PY

cat <<EOF

✓ Environment '${ENV_NAME}' is ready.

Activate it with:
    conda activate ${ENV_NAME}

Then run the pipeline:
    bash run.sh single --movie /path/to/file.nd2

EOF
