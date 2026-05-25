---
name: copairs
description: Work with copairs and copairs-runner for phenotypic profiling analysis. Use when discussing mAP metrics, activity/consistency analysis, configuring copairs-runner YAML configs, reviewing copairs results, or strategizing Cell Painting data analysis.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# copairs & copairs-runner Skill

This skill provides expertise for working with the **copairs** library (phenotypic profiling metrics) and **copairs-runner** (YAML-driven orchestration).

## Quick Reference

For detailed API documentation, see [API.md](API.md).
For repo-specific configuration patterns, see [CONFIG.md](CONFIG.md).
For the scientific background, see [SCIENCE.md](SCIENCE.md).

## Overview

**copairs** is an information retrieval framework for evaluating phenotypic profile strength and similarity using mean Average Precision (mAP). Published in Nature Communications 2025.

**copairs-runner** is a YAML-driven CLI that orchestrates copairs analyses with Hydra configuration management.

## Core Concepts

The mAP framework answers three related questions by treating profile evaluation as an information retrieval problem:

| Concept | Question | Query Group | Reference Group |
|---------|----------|-------------|-----------------|
| **Phenotypic Activity** | Does this perturbation cause a detectable change vs controls? | Perturbation replicates | Negative controls |
| **Phenotypic Consistency** | Do perturbations with the same annotation cluster together? | Perturbations sharing a label (target/MOA) | Perturbations with different labels |
| **Phenotypic Distinctiveness** | Is this perturbation distinguishable from all other perturbations? | Perturbation replicates | All other perturbations |

All three use the same underlying algorithm - they differ only in how positive and negative pairs are defined.

### The Algorithm (General Form)

```
For each profile i in the query group:
  1. Compute distances to all other profiles (query group + reference group)
  2. Rank by similarity (closest = rank 1)
  3. Create binary relevance list: 1 = same group, 0 = different group
  4. Compute Average Precision:
     AP = Σ (Precision@k × ΔRecall@k) for k where relevance = 1

mAP = mean(AP scores for all profiles in query group)
p-value = fraction of null permutations with mAP ≥ observed
```

**For Activity:** "same group" = replicates of the same perturbation; "different group" = controls
**For Consistency:** "same group" = perturbations sharing a label; "different group" = perturbations with different labels
**For Distinctiveness:** "same group" = replicates of the perturbation; "different group" = all other perturbations

### Key Metrics

**mAP (mean Average Precision)**
- Range: 0 to 1
- Higher = better retrieval of relevant profiles
- mAP = 1 means perfect retrieval (all positives ranked before all negatives)
- Random performance depends on the ratio of positive to negative pairs

**Normalized mAP**
- Formula: `(AP - μ₀) / (1 - μ₀)` where μ₀ is expected AP under random ranking
- Range: can be negative (worse than random) to 1 (perfect)
- Normalized mAP = 0 means performance equals random chance
- Scale-independent: allows comparison across different group sizes

**p-value**
- Computed via permutation test (typically 100,000 permutations)
- Corrected for multiple testing using Benjamini-Hochberg FDR
- Use `corrected_p_value` and `below_corrected_p` columns for significance

### Key Parameters

| Parameter | Purpose | Notes |
|-----------|---------|-------|
| `pos_sameby` | Columns that must match for positive pairs | Defines what "same group" means |
| `pos_diffby` | Columns that must differ for positive pairs | Optional constraint (e.g., cross-source) |
| `neg_sameby` | Columns that must match for negative pairs | For blocking (e.g., same plate) |
| `neg_diffby` | Columns that must differ for negative pairs | Defines what "different group" means |
| `distance` | Similarity metric | Options: `cosine` (default), `abs_cosine`, `correlation`, `euclidean`, `manhattan` |
| `null_size` | Permutations for p-value | Typically 100,000 |
| `threshold` | p-value significance threshold | For `below_p` / `below_corrected_p` flags |

## Running copairs-runner

```bash
# General form
copairs-runner --config-dir <config_dir> --config-name <config_name> [overrides...]

# Override any parameter
copairs-runner ... distance=abs_cosine mean_average_precision.params.null_size=50000
```

The `distance` parameter can be set for any analysis type (activity, consistency, or distinctiveness).

## Interpreting Results

**Output columns:**
- `mean_average_precision`: mAP score for the group
- `mean_normalized_average_precision`: Normalized mAP (scale-independent)
- `p_value`: Raw p-value from permutation test
- `corrected_p_value`: FDR-corrected p-value (use this for significance)
- `below_corrected_p`: Boolean flag for significance

**Guidance:**
- Use `corrected_p_value` (not raw `p_value`) for determining significance
- Compare normalized mAP when groups have different sizes
- Low percent retrieved across the dataset may indicate batch effects or weak signal

## Troubleshooting

### "No positive pairs found"
- Check that `pos_sameby` columns exist and have matching values
- Verify data has replicates (same identifier appearing multiple times)

### "No negative pairs found"
- Check that reference assignment worked (controls have reference index = -1)
- Verify `neg_diffby` column has variation

### Low percent retrieved
- May indicate batch effects - try different preprocessing or batch correction
- Consider feature selection to improve signal-to-noise ratio
- Check that profiles are properly normalized

### Slow runs or hanging
- Usually caused by too many positive pairs (scales quadratically with replicates)
- Use `aggregate_replicates` preprocessing to reduce dense data to plate-level
- See [CONFIG.md](CONFIG.md#performance-troubleshooting) for pair count estimation and aggregation strategies

---

## Repository-Specific: jpx

The following sections describe patterns specific to this repository. For the current configuration, inspect the actual files rather than relying on this documentation (configs may change).

### Inspect Current Configs

```bash
# List all config groups
ls configs/copairs/metrics/

# View specific config
cat configs/copairs/metrics/activity.yaml
cat configs/copairs/metrics/activity_params/default.yaml

# List all preprocessing options
ls configs/copairs/metrics/preprocessing_02_core/

# List all target columns for consistency
ls configs/copairs/metrics/target_column/

# View the run matrix
cat configs/copairs/copairs_runs.yaml
```

### Running in This Repo

```bash
# Activity analysis
pixi run copairs-runner \
  --config-dir configs/copairs/metrics --config-name activity \
  dataset=compound_no_source7 \
  columns=feat_all \
  preprocessing_02_core=activity_no_target2 \
  filter=all_sources \
  activity_params=default

# Consistency analysis
pixi run copairs-runner \
  --config-dir configs/copairs/metrics --config-name consistency \
  dataset=compound_no_source7 \
  columns=feat_all \
  preprocessing_02_core=consistency_no_target2 \
  filter=all_sources \
  target_column=uniprot \
  distance=cosine
```

### Results Location

Results are written to structured directories based on config parameters:
```
data/processed/copairs/runs/
├── activity/{run_name}/results/
│   ├── activity_map_results.csv
│   ├── activity_ap_scores.csv
│   └── .hydra/config.yaml
└── consistency/{run_name}/results/
    ├── consistency_map_results.csv
    ├── consistency_ap_scores.csv
    └── .hydra/config.yaml
```

### Aggregated Results Database

This repo aggregates all copairs results into a DuckDB for unified querying:

```python
import duckdb
con = duckdb.connect("data/processed/copairs_results.duckdb")

# See available tables
con.sql("SHOW TABLES")

# Query with config metadata
con.sql("SELECT * FROM activity_results LIMIT 5")
con.sql("SELECT * FROM consistency_results LIMIT 5")
```

### Key Files

```bash
# Main configs
configs/copairs/metrics/activity.yaml
configs/copairs/metrics/consistency.yaml

# Preprocessing pipelines
configs/copairs/metrics/preprocessing_02_core/

# Run matrix
configs/copairs/copairs_runs.yaml

# Pipeline orchestration
workflow.py

# Results aggregation
notebooks/nb41_ss_copairs_db.py
```
