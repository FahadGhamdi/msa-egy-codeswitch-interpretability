# Revision analyses — 25 October 2026

This directory accompanies the revised Electronics manuscript. The original submitted code remains pinned at `39757498a7dc76a5a634e93f1f9c0f865d1f7bab` and is retained outside this directory.

`source/` preserves revision analysis scripts and their source-relative paths. `SOURCE_PROVENANCE.json` records the exact copies. Portable Stage1 and selected-curve replay tools are in `reproducibility_extension/`; saved aggregate outcomes are in `aggregate_results/`. Consult `STAGE1_COMMANDS.txt` and the input archive inventory before execution. Source paths from the original research workspace may require configuration. These scripts are not all standalone entrypoints.

The limited R1.4 scripts are provided under `r14_limited_code/` for provenance; the independently audited random-control aggregates at k=1,10,20 are available under `r14_limited_results/`. That subdirectory alone is not a runnable inference bundle: its frozen models/inputs/manifest are deliberately not redistributed here. The numerical audit accompanies these results; reviewer acceptance and all-integer-k random controls are not claimed.

Raw restricted corpora, tokenized texts, model weights and private checkpoints are excluded. Obtain upstream data under its providers' conditions. The external archival files were supplied to the researcher; the author confirms that access is by request to the original data providers. Diab et al. (2016), https://aclanthology.org/L16-1669/, describes the data and annotations; its footnote points to annotation guidelines and an example repository, not an explicit blanket redistribution licence for the complete corpus. Required saved-array archives must be made accessible under appropriate terms or regenerated before describing this release as fully independently reproducible.

The existing MIT code licence does not license third-party corpora or model weights. The manuscript Data Availability Statement should identify the verified
release tag and commit. The original submitted version and this revision
are separate, traceable versions.
