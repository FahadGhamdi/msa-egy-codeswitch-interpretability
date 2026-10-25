# Revision analyses

This directory contains analysis scripts, aggregate results and reproducibility documentation for the revised Electronics manuscript.

Start with [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for requirements, input access, commands, expected outputs and limitations.

| Location | Contents |
|---|---|
| `source/` | Archived revision analysis scripts, preserving their source-relative paths. Some require workspace-specific configuration. |
| `SOURCE_PROVENANCE.json` and `SCRIPT_INVENTORY.json` | Source-file identities and inventory. |
| `reproducibility_extension/` | Saved-array statistical recomputation scripts and input inventories. |
| `aggregate_results/` | Saved aggregate outcomes, including Stage1 reference CSVs. |
| `r14_limited_code/` | Code and protocol for size-matched random controls at k = 1, 10 and 20. |
| `r14_limited_results/` | Aggregate random-control outcomes and statistical interpretation. |
| `TABLE_SOURCE_MAP.json` | Archived table-to-source mapping; numbering reflects its source version. |

## Data access and reproducibility

The public repository does not include the saved-array archives, restricted raw corpora, tokenized inputs, model weights or private checkpoints required for every analysis. Input inventories identify required files but do not provide downloads. Contact the corresponding author about saved-array access under applicable terms.

External archival tweet and comment texts are available by request to the original resource authors, subject to their access and use conditions. [Diab et al. (2016)](https://aclanthology.org/L16-1669/) describes that resource; links to guidelines and examples do not establish a blanket redistribution licence for the full corpus. The MIT code licence does not license third-party corpora or model weights.

The random-control code directory is not a standalone inference bundle. Its frozen inputs, models and execution manifest must be supplied separately. Saved-array checks and aggregate results do not establish end-to-end reproduction from a fresh clone.

## Versions

- Original submitted code: `39757498a7dc76a5a634e93f1f9c0f865d1f7bab`.
- Revised analysis snapshot cited in the manuscript: `cc5d37c389f1bc1d9da42f551fa112861d98a521`.

The English documentation update changes instructions and presentation only; analysis scripts and numerical results remain unchanged.
