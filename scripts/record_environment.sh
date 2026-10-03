#!/usr/bin/env bash
# Writes software versions and the exact Hugging Face commit of every model checkpoint found in the
# local caches to results/review/environment.txt.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/review/environment.txt
mkdir -p results/review
python -c "import platform,torch,transformers;print('python',platform.python_version(),'torch',torch.__version__,'transformers',transformers.__version__)" > "$OUT"
for m in Qwen/Qwen3-8B-Base ALLaM-AI/ALLaM-7B-Instruct-preview google/gemma-2-9b QCRI/Fanar-1-9B; do
  for c in ".cache/huggingface/hub" "${HF_HOME:-$HOME/.cache/huggingface}/hub" "$HOME/.cache/huggingface/hub"; do
    f="$c/models--${m//\//--}/refs/main"
    if [ -f "$f" ]; then echo "$m revision $(cat "$f")" >> "$OUT"; break; fi
  done
done
cat "$OUT"
