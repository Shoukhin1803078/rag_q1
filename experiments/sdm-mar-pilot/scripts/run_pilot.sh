#!/usr/bin/env bash
# End-to-end SDM-MAR pilot: run the agent over the synthetic corpus, then analyse.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

echo "== Phase A: agent runs =="
$PY -m src.run_runs "$@"

echo "== Phase B: analysis =="
$PY -m src.analyze

echo "Done. See results/PRELIMINARY_RESULTS.md"
