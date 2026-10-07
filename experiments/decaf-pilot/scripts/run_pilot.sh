#!/usr/bin/env bash
# End-to-end DECAF pilot: scaling + compression experiments, then analysis.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

echo "== Phase A: experiments =="
$PY -m src.run_experiments "$@"

echo "== Phase B: analysis =="
$PY -m src.analyze

echo "Done. See results/PRELIMINARY_RESULTS.md"
