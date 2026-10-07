#!/usr/bin/env bash
# End-to-end PACER pilot: run base systems, then analyse.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

echo "== Phase A: base-system runs =="
$PY -m src.run_harness "$@"

echo "== Phase B: analysis =="
$PY -m src.analyze

echo "Done. See results/PRELIMINARY_RESULTS.md"
