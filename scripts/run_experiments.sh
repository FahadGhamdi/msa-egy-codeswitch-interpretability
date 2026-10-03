#!/usr/bin/env bash
# Re-runs every model experiment of the paper, in the order used for the reported results.
# Run from the repository root with the Python environment active:
#     bash scripts/run_experiments.sh
# Models are downloaded from Hugging Face into .cache/huggingface (about 70 GB for the four models).
# Total running time on one Apple-silicon workstation: several days (exp06 and exp09 dominate).
# Set DRAWS to fewer random head sets for a quick check (the paper uses 100).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
DRAWS="${DRAWS:-100}"
MODELS="Qwen/Qwen3-8B-Base ALLaM-AI/ALLaM-7B-Instruct-preview google/gemma-2-9b QCRI/Fanar-1-9B"
E=experiments
mkdir -p experiments/logs
LOG="experiments/logs/run_$(date +%Y%m%d_%H%M).log"
{
  python $E/fetch_lince.py                                   # LinCE lid_msaea -> data/raw/lince_msaea
  for m in $MODELS; do
    echo "=== $m: probes, layer sweep, patching, natural vs controlled ==="
    python $E/exp01_probe_switch.py --model "$m"
    python $E/exp01b_lexical_controls.py --model "$m"
    python $E/exp02_causal_patching.py --model "$m"
    python $E/exp03_natural_vs_controlled.py --model "$m"
    python $E/exp04_probe_readout.py --model "$m"
  done
  for m in Qwen/Qwen3-8B-Base ALLaM-AI/ALLaM-7B-Instruct-preview; do
    python $E/exp02_causal_patching.py --model "$m" --reverse   # reverse patching (MSA run as clean run)
  done
  python $E/compare_models.py
  for m in $MODELS; do
    echo "=== $m: audit, lexical controls, ablation and predictions ==="
    python $E/verify_ablation.py --model "$m"
    python $E/exp07_lexical_controls.py --model "$m"
    python $E/exp06_behavioral.py --model "$m" --random_draws "$DRAWS" --later_draws "$DRAWS" --span_draws "$DRAWS" --batch 16
  done
  python $E/exp06_behavioral.py --model QCRI/Fanar-1-9B --heads_model gemma-2-9b \
         --random_draws "$DRAWS" --later_draws "$DRAWS" --span_draws "$DRAWS" --batch 16
  python $E/exp06_behavioral.py --model Qwen/Qwen3-8B-Base --purity_source all --min_next 0   # sensitivity run
  python $E/exp07b_matching_sensitivity.py
  python $E/exp08_final_analyses.py
  python $E/exp12_review_stats.py
  for m in $MODELS; do
    python $E/exp10_heads_egy.py --model "$m"
    python $E/exp11_decodability.py --model "$m"
    python $E/exp09_sites.py --model "$m" --random_draws "$DRAWS" --later_draws "$DRAWS" --batch 16
  done
  python $E/exp09_sites.py --model QCRI/Fanar-1-9B --heads_model gemma-2-9b --random_draws "$DRAWS" --later_draws "$DRAWS" --batch 16
  python paper/review_analyses.py --results results
  bash scripts/record_environment.sh
} 2>&1 | tee "$LOG"
