#!/usr/bin/env bash
# End-to-end CHAB-RAG pilot: build the (q,k) grid, then analyse it.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

echo "== Phase A: generation grid =="
$PY -m src.run_grid "$@"

echo "== Phase B: analysis =="
$PY -m src.analyze

echo "Done. See results/PRELIMINARY_RESULTS.md"
