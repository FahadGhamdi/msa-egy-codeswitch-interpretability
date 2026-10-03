#!/usr/bin/env bash
# Regenerates every table and figure of the paper from the saved files in results/ (no model runs, ~1 min).
#     bash scripts/make_paper_outputs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python paper/review_analyses.py --results results
python paper/make_tables.py  --results results --out paper/tables
python paper/make_figures.py --results results --out paper/figures
