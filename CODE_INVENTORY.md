# Code inventory

Every script used to produce the results, tables and figures of the article. All local imports resolve inside `experiments/`.

## Data

| File | Role |
|---|---|
| `experiments/fetch_lince.py` | Downloads LinCE `lid_msaea` (Hugging Face copy) into `data/raw/lince_msaea`. |

## Shared modules

These are imported by the experiments; they are not run directly.

| File | Role |
|---|---|
| `experiments/exp02_causal_patching.py` | Also defines data loading, word/sub-token alignment and the bootstrap. |
| `experiments/exp03_natural_vs_controlled.py` | Also defines code-switched item extraction and the random-prefix pools. |
| `experiments/exp04_probe_readout.py` | Also defines layer-wise probe training. |
| `experiments/readout.py` | Saved read-out probe, monolingual type purity, head hooks, mean ablation, direct attribution, bootstrap helpers. |

## Experiments

Run in this order; see `scripts/run_experiments.sh`.

| File | Article section | Output |
|---|---|---|
| `exp01_probe_switch.py` | Layer sweep, probe accuracy (Fig. 3) | `results/exp01` |
| `exp01b_lexical_controls.py` | Lexical baselines for the layer sweep | `results/exp01` |
| `exp02_causal_patching.py` | Minimal pairs, activation patching, head ranking (`--reverse`: reverse runs) | `results/exp02`, `results/exp02_reverse` |
| `exp03_natural_vs_controlled.py` | Natural vs. random-prefix contexts (pull) | `results/exp03` |
| `exp04_probe_readout.py` | Flip rates at the read-out layer | `results/exp04` |
| `compare_models.py` | Cross-model summary | `results/comparison` |
| `verify_ablation.py` | Audit of the intervention code (Appendix A) | `results/verify_ablation` |
| `exp07_lexical_controls.py` | Item table with covariates; regression and matching | `results/exp07` |
| `exp06_behavioral.py` | Ablation vs. 100 matched random head sets; next-token preference D; continuation; Gemma heads in Fanar | `results/exp06` |
| `exp07b_matching_sensitivity.py` | Matching sensitivity (calipers, re-draws) | `results/exp07b_matching_sensitivity.json` |
| `exp08_final_analyses.py` | Final regression/matching specifications, word class | `results/exp08_final_analyses.json` |
| `exp12_review_stats.py` | Cluster-bootstrap Spearman, TOST margin sensitivity, tweet counts | `results/review/review_long.json` |
| `exp10_heads_egy.py` | Layers and heads re-ranked on Egyptian targets | `results/exp10_heads_EGY` |
| `exp11_decodability.py` | Recoverability of the word's own variety (balanced probe, control task) | `results/exp11_decodability` |
| `exp09_sites.py` | Equal-count intervention sites and distance profile | `results/exp09_sites` |

## Paper outputs

These need no model.

| File | Role |
|---|---|
| `paper/review_analyses.py` | Local insertions, Holm correction across runs → `results/review/review_saved.json`. |
| `paper/make_tables.py` | Every results table (LaTeX) of the article. |
| `paper/make_figures.py` | Every data figure of the article, including the Arabic-script examples. Figure 2 is drawn in TikZ inside the manuscript. |

## Shell scripts

| File | Role |
|---|---|
| `scripts/run_experiments.sh` | Full pipeline. |
| `scripts/make_paper_outputs.sh` | Regenerates tables and figures from `results/`. |
| `scripts/record_environment.sh` | Records versions and checkpoint commits. |

## Not included

These played no part in the reported results:
- Environment checks for the author's machine.
- A downloader for datasets that were not used.
- An early pilot on Qwen2.5-0.5B.
- `exp05_head_ablation.py`, an earlier ablation script superseded by `exp06_behavioral.py` (the late-binding error described in Section 3.7 occurred in it; every reported ablation number comes from `exp06`/`exp09` and the shared `readout.AblationHooks`).
