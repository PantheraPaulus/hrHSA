#!/bin/bash
# Shared setup for hrHSA CoolMUC-4 benchmark jobs.

set -euo pipefail

# Optional persistent site/user overrides. This is preferable to relying on the
# submitting shell because LRZ batch scripts intentionally use --export=NONE.
HRHSA_COOLMUC_CONFIG="${HRHSA_COOLMUC_CONFIG:-$HOME/.config/hrhsa/coolmuc4.env}"
if [[ -f "$HRHSA_COOLMUC_CONFIG" ]]; then
    # shellcheck disable=SC1090
    source "$HRHSA_COOLMUC_CONFIG"
fi

# Submission helpers inject the source commit explicitly. For direct batch runs,
# retain a best-effort fallback when git is available on the compute node.
if [[ -z "${HRHSA_GIT_COMMIT:-}" ]] && command -v git >/dev/null 2>&1; then
    HRHSA_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
fi
export HRHSA_GIT_COMMIT="${HRHSA_GIT_COMMIT:-}"

activate_hrhsa() {
    local env_name="${HRHSA_CONDA_ENV:-hsa}"
    local repo_root

    if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
        conda activate "$env_name"
    elif [[ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/mambaforge/etc/profile.d/conda.sh"
        conda activate "$env_name"
    else
        # LRZ provides micromamba through an environment module. Its module setup
        # and shell hook may inspect variables such as MAMBA_ROOT_PREFIX before
        # defining/defaulting them. Bash `set -u` therefore causes a false startup
        # failure. Suspend nounset for the entire LRZ micromamba bootstrap -- from
        # module loading through activation -- and restore it immediately after.
        # `set -e` and `pipefail` remain active throughout.
        set +u

        if ! command -v micromamba >/dev/null 2>&1; then
            if type module >/dev/null 2>&1; then
                module load micromamba
            fi
        fi

        if command -v micromamba >/dev/null 2>&1; then
            eval "$(micromamba shell hook -s bash)"
            micromamba activate "$env_name"
        else
            echo "Could not find Miniforge/Mambaforge or load LRZ micromamba." >&2
            echo "Expected environment: $env_name" >&2
            set -u
            exit 2
        fi

        set -u
    fi

    # Benchmark jobs must execute the source tree associated with the submitted
    # checkout, not an older hrHSA installation that happens to live in the
    # activated environment.  Derive the repository root from this versioned
    # helper rather than from the caller's working directory, prepend its src/
    # directory to PYTHONPATH, and verify the resolved import below.
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    export HRHSA_REPO_ROOT="$repo_root"
    export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

    python - <<'PY'
from pathlib import Path
import hsa
import dask
import distributed
import geopandas
import os
import xarray

repo_root = Path(os.environ["HRHSA_REPO_ROOT"]).resolve()
repo_src = (repo_root / "src").resolve()
hsa_path = Path(hsa.__file__).resolve()
try:
    hsa_path.relative_to(repo_src)
except ValueError as exc:
    raise SystemExit(
        "hrHSA benchmark environment resolved hsa from the wrong source tree: "
        f"expected beneath {repo_src}, observed {hsa_path}"
    ) from exc

print("hrHSA benchmark environment OK")
print("repo_root:", repo_root)
print("hsa:", hsa_path)
print("dask:", dask.__version__)
print("distributed:", distributed.__version__)
print("xarray:", xarray.__version__)
print("geopandas:", geopandas.__version__)
print("git_commit:", os.environ.get("HRHSA_GIT_COMMIT") or "unavailable")
PY
}

# Dask provides task-level concurrency; native numerical libraries should not
# start their own nested thread teams inside each Dask task.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -z "${SCRATCH_DSS:-}" ]]; then
    echo "SCRATCH_DSS is not defined; CoolMUC-4 benchmark data need shared scratch." >&2
    exit 2
fi

# Set HRHSA_DASK_INTERFACE in coolmuc4.env after running probe_coolmuc4.py if a
# dedicated InfiniBand/IPoIB interface should be selected explicitly.
export HRHSA_DASK_INTERFACE="${HRHSA_DASK_INTERFACE:-}"
export HRHSA_DATA_ROOT="${HRHSA_DATA_ROOT:-$SCRATCH_DSS/hrhsa-benchmark-medium}"
export HRHSA_LARGE_DATA_ROOT="${HRHSA_LARGE_DATA_ROOT:-$SCRATCH_DSS/hrhsa-benchmark-large}"
export HRHSA_RESULT_ROOT="${HRHSA_RESULT_ROOT:-$SCRATCH_DSS/hrhsa-benchmark-results}"
mkdir -p "$HRHSA_RESULT_ROOT"
