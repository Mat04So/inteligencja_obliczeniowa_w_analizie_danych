#!/usr/bin/env bash
# Uruchom trening tak, żeby zmienne OpenMP/KMP były widoczne ZANIM załaduje się PyTorch
# (częsty powód „mutex lock failed” na macOS przy mieszance torch/numpy/MKL).
#
# Użycie:
#   chmod +x run_train_macos.sh   # raz
#   ./run_train_macos.sh --env v2
#
# Z aktywnym venv (zalecane):
#   source .venv/bin/activate
#   ./run_train_macos.sh --env v2

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

exec "$PYTHON_BIN" "$ROOT/train.py" "$@"
