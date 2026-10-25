# Absorbed or Anchored? — reproducibility package

Code and saved results for the article

> F. AlGhamdi. *Absorbed or Anchored? How Large Language Models Represent Embedded Words in Modern Standard Arabic–Egyptian Arabic Code-Switching.* Manuscript under review, 2026.

The root-level pipeline and saved summaries correspond to the original submitted manuscript. The revised analyses are in [`revision_20261025/`](revision_20261025/README.md). Start with the [revision reproducibility guide](revision_20261025/REPRODUCIBILITY.md) for input requirements, supported recomputation paths and limitations. The original pipeline below regenerates its saved-summary outputs without model inference; model experiments require separately obtained data and checkpoints.

## Contents

| Folder | What it contains |
|---|---|
| `experiments/` | All experiment scripts (`exp01`–`exp12`), the shared read-out utilities (`readout.py`), the ablation audit (`verify_ablation.py`) and the data download script (`fetch_lince.py`). |
| `paper/` | `make_tables.py` and `make_figures.py`, which build every table and figure of the article from `results/`, and `review_analyses.py` (local insertions, Holm correction across runs). |
| `results/` | Saved summary results of every experiment used in the article (CSV/JSON). |
| `scripts/` | `make_paper_outputs.sh` (tables and figures), `run_experiments.sh` (all model runs), `record_environment.sh` (software versions and checkpoint commits). |
| `data/` | Empty; filled by `fetch_lince.py`. |

## Quick start: regenerate the tables and figures

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/make_paper_outputs.sh      # writes paper/tables/*.tex and paper/figures/*.pdf
```

The figures with Arabic script need the font **Noto Naskh Arabic** to be installed.

## Re-running the experiments

```bash
bash scripts/run_experiments.sh         # all models, in the order used for the article
DRAWS=10 bash scripts/run_experiments.sh   # quick check with fewer random head sets (article: 100)
```

- **Hardware:** the reported runs used a single Apple-silicon workstation (PyTorch MPS backend, bfloat16). The scripts fall back to CPU when MPS is not available. Running on CUDA requires changing the `device` line of a script. Small numerical differences across hardware are expected.
- **Downloads:** the four checkpoints (about 70 GB in total) are downloaded into `.cache/huggingface`.
- **Running time:** the full pipeline takes several days; `exp06` and `exp09` take the longest.

**Models:**
- `Qwen/Qwen3-8B-Base`
- `ALLaM-AI/ALLaM-7B-Instruct-preview`
- `google/gemma-2-9b`
- `QCRI/Fanar-1-9B`

Software versions and checkpoint commits are listed in `results/review/environment.txt`.

**Fixed settings:**
- Read-out layer: `R = round(0.8 L)`.
- Seed: 13 for item sampling, probe training and random head draws.
- All bootstrap procedures use fixed seeds and resample whole tweets.

## Data

The experiments use the MSA–Egyptian Arabic language-identification set (`lid_msaea`) of the LinCE benchmark (Aguilar et al., 2020).

- `experiments/fetch_lince.py` downloads it from the public Hugging Face copy `lince-benchmark/lince` into `data/raw/lince_msaea`.
- No tweets are redistributed in this repository.
- `results/exp07/*/L_items.csv` lists the analysed embedded words, each with its LinCE sentence identifier (e.g. `train-53`), so that every item can be traced back to the benchmark.

## Original submitted manuscript: code map

| Article | Script(s) | Results |
|---|---|---|
| Layer sweep, probe accuracy (Figure 3) | `exp01_probe_switch.py`, `exp01b_lexical_controls.py` | `results/exp01` |
| Flip rates and representation (Table 5) | `exp04_probe_readout.py`, `exp07_lexical_controls.py` | `results/exp04`, `results/exp07` |
| Regression and matching | `exp07_lexical_controls.py`, `exp07b_matching_sensitivity.py`, `exp08_final_analyses.py` | `results/exp07`, `results/exp07b_matching_sensitivity.json`, `results/exp08_final_analyses.json` |
| Natural vs. controlled contexts, equivalence tests | `exp03_natural_vs_controlled.py`, `exp12_review_stats.py` | `results/exp03`, `results/review/review_long.json` |
| Minimal pairs and activation patching | `exp02_causal_patching.py` (`--reverse` for reverse runs), `exp10_heads_egy.py` | `results/exp02`, `results/exp02_reverse`, `results/exp10_heads_EGY` |
| Head ablation vs. matched random heads, predictions | `exp06_behavioral.py` | `results/exp06` |
| Recoverability of the word's own variety | `exp11_decodability.py` | `results/exp11_decodability` |
| Intervention sites and distance profile | `exp09_sites.py` | `results/exp09_sites` |
| Audit of the intervention code (Appendix A) | `verify_ablation.py` | `results/verify_ablation` |
| Local insertions, Holm correction across runs | `paper/review_analyses.py` | `results/review/review_saved.json` |

Head labels such as `L29H12` use 0-based indices for the transformer block (`model.layers[29]`) and the head. The read-out layer `R` indexes `hidden_states[R]`, the output of `model.layers[R-1]`.

The per-item long files written by the experiments (`*_long.csv`) and the saved probes (`*.pt`) are not included. They are regenerated by the experiment scripts, and the summaries derived from them are in `results/`.

## Citation

See `CITATION.cff`. Please cite the article and the LinCE benchmark.

## License

The code is released under the MIT License (see `LICENSE`). LinCE data remain subject to the terms of their maintainers.

## Revision analyses — 25 October 2026

The revision code, aggregate results and documentation are available in
[`revision_20261025/`](revision_20261025/README.md).

The size-matched random-control extension includes size-matched random
controls at k = 1, 10 and 20 for all four models. The complete selected-head
curves and the limited random-control comparisons are distinct analyses.
Intervals are pointwise and conditional on the frozen random draws.

The original submitted version remains identifiable by commit:
`39757498a7dc76a5a634e93f1f9c0f865d1f7bab`.

Restricted corpus texts, tokenized inputs and model weights are not
redistributed. Some analyses require separately obtained inputs; see the
revision documentation for requirements and limitations.
