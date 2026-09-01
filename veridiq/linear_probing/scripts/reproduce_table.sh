#!/usr/bin/env bash
# Reproduce the cross-dataset linear-probing table (eval-only).
#
# From repo root (veridiq_2026/veridiq):
#   bash veridiq/linear_probing/scripts/reproduce_table.sh
#
# Or from linear_probing/:
#   bash scripts/reproduce_table.sh
#
# Uses the py312 conda env when available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LP_DIR}/../.." && pwd)"
CFG_DIR="${LP_DIR}/configs/table"

if [[ -x /root/.conda/envs/py312/bin/python ]]; then
  PYTHON=/root/.conda/envs/py312/bin/python
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  PYTHON="$(command -v python3)"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${LP_DIR}"

echo "Repo:    ${REPO_ROOT}"
echo "LP dir:  ${LP_DIR}"
echo "Python:  ${PYTHON}"
echo "Configs: ${CFG_DIR}"

shopt -s nullglob
TEST_CFGS=("${CFG_DIR}"/test_*.yaml)
LATE_CFGS=("${CFG_DIR}"/late_*.yaml)

if [[ ${#TEST_CFGS[@]} -eq 0 ]]; then
  echo "No test configs found in ${CFG_DIR}" >&2
  exit 1
fi

echo "Running ${#TEST_CFGS[@]} test configs..."
for cfg in "${TEST_CFGS[@]}"; do
  echo "=== TEST $(basename "${cfg}") ==="
  "${PYTHON}" -m veridiq.linear_probing.train_test --config_path "${cfg}" --test
done

echo "Running ${#LATE_CFGS[@]} late-fusion configs..."
for cfg in "${LATE_CFGS[@]}"; do
  echo "=== LATE $(basename "${cfg}") ==="
  "${PYTHON}" -m veridiq.linear_probing.utils.late_fusion --config_path "${cfg}"
done

echo "Done. Printing table:"
"${PYTHON}" "${SCRIPT_DIR}/print_table.py"
