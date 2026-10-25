# Reproducing the revision analyses

This guide describes the scripts and inputs for the revised manuscript. It distinguishes recomputation from saved arrays from new model inference. The commands below perform statistical recomputation only; they do not load language models.

## Requirements and input availability

Use Python 3 with NumPy and SciPy. The recorded CPU analysis environment is in [PORTABLE_ENVIRONMENT.json](PORTABLE_ENVIRONMENT.json); it is not the model-inference environment.

The public repository contains scripts and aggregate results. The saved-array archives required by the commands below are **not included**. Their filenames and SHA-256 checksums are recorded in:

- [REQUIRED_CURVE_ARCHIVES.json](reproducibility_extension/REQUIRED_CURVE_ARCHIVES.json) for selected-head curves.
- [STAGE1_INPUT_INDEX.json](reproducibility_extension/STAGE1_INPUT_INDEX.json) and [REQUIRED_STAGE1_INPUTS.json](REQUIRED_STAGE1_INPUTS.json) for Stage1 archive preparation.

Contact the corresponding author about access to saved-array archives, subject to the applicable data terms; the inventories are not download links. Raw archival corpus texts must be requested from the original data providers. Model weights must be obtained from their respective providers. The repository's MIT licence applies to code, not to third-party corpora or model weights.

Replace all example absolute paths below with paths on your system. Use a new output directory for each run.

## Selected-head curve recomputation

From the repository root, after obtaining the three archives listed in the curve inventory:

```bash
python3 revision_20261025/reproducibility_extension/run_curve_audit.py \
  --archives /absolute/path/curve_archives \
  --out /absolute/path/new_curve_audit
```

The wrapper checks archive hashes, rejects an existing output directory and conflicting archive members, and runs the saved-array analysis with 10,000 bootstrap resamples. Results are written under the output directory's `analysis/` subdirectory; the validation report is `analysis/INDEPENDENT_AUDIT.json`.

The previously recorded validation covered 2,601 items, 21 conditions per item, 2,016 result rows and 10,080 numerical comparisons, with a maximum absolute difference of 4.440892098500626e-16. These figures describe the earlier validation run, not a new experiment or a rerun performed for this documentation update. This analysis does not generate new random-head controls.

## Stage1 statistical recomputation

[STAGE1_COMMANDS.txt](STAGE1_COMMANDS.txt) provides archive preparation and model-specific commands. Each model analysis requires frozen item arrays and six reference CSV files. The reference CSVs are available under [aggregate_results/verified_stage1_analysis/](aggregate_results/verified_stage1_analysis/).

The analysis recomputes continuation means, next-token variety preference, three replacement modes, paired contrasts and 10,000-resample tweet-cluster intervals. The earlier validation reported 627,120 numerical field comparisons across the four models, with zero difference in that environment. The implemented comparison tolerance is absolute 1e-10 with zero relative tolerance. These are numerical comparisons, not independent hypothesis tests.

## Original submitted analyses and manuscript source

The root-level [README](../README.md) describes the original submitted analysis pipeline. Its table and figure numbers refer to that version. Revision analyses and aggregate results are documented separately here.

The former reproduction note referred to a local delivery bundle containing `Reproduction/` and `Overleaf/`. Those directories are not part of this public repository. Commands that depend on that bundle are not repository quick-start commands. Build the manuscript using the instructions in its separately supplied Overleaf source package.

## Scope and limitations

- Saved-array recomputation does not rerun model inference or reconstruct the FP32 reference means.
- Historical BF16 replay remains unresolved; exact agreement with that runtime is not established by these checks.
- Some archived source scripts retain research-workspace paths and are not standalone entry points.
- The size-matched random controls at k = 1, 10 and 20 are a separate analysis, documented in [r14_limited_code/PROTOCOL.txt](r14_limited_code/PROTOCOL.txt) and [r14_limited_results/README.txt](r14_limited_results/README.txt). Their intervals are pointwise and conditional on frozen draws.
- Code and aggregate-result availability do not establish unrestricted access to all required inputs or end-to-end reproduction from a fresh clone.

## Version provenance

The original submitted code is pinned at `39757498a7dc76a5a634e93f1f9c0f865d1f7bab`. The revised analysis snapshot cited in the manuscript is `cc5d37c389f1bc1d9da42f551fa112861d98a521`.

This documentation update leaves analysis scripts, input inventories and aggregate results unchanged. Use the cited snapshot to identify the scientific code version, and this guide for corrected public-repository instructions. File hashes identify content; they do not grant data access or replace a versioned repository citation.
