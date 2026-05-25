# copairs Configuration in jpx

This document describes how to explore and work with the copairs-runner configuration in this repository. **Since configs evolve, always inspect the actual files rather than relying on static documentation.**

## Key Concepts

### Datasets = Different Embeddings

A "dataset" is a profile identifier representing a specific embedding of the underlying compound data. The same compounds can have multiple embeddings (feature extraction methods):

| Dataset | Description |
|---------|-------------|
| `compound_no_source7` | CellProfiler features (source 7 excluded) |
| `compound_DL_CPCNN_no_source7` | Deep Learning CPCNN embeddings (source 7 excluded) |
| `compound_no_source7_rsc` | CellProfiler, robust scaled variant |
| `compound_no_source7_active_union` | CellProfiler, pre-filtered to union-active compounds |
| `compound_DL_CPCNN_no_source7_active_union` | DL, pre-filtered to union-active compounds |

### Config Keys → DuckDB Columns

Config keys map to Hydra configs in `configs/copairs/` and become `_*` metadata columns in `copairs_results.duckdb`:

| Config Key | DuckDB Column | Description |
|------------|---------------|-------------|
| `dataset` | `_dataset` | Profile identifier (see above) |
| `columns` | `_columns` | Feature subset (e.g., `feat_all`) |
| `preprocessing_02_core` | `_preprocessing` | Activity/consistency config name |
| `filter` | `_filter` | Source filtering (`all_sources`, `source_2`, `no_source9`) |
| `activity_params` | `_activity_params` | Replicate pairing (`default`, `withinsource`, `crosssource`) |
| `target_column` | `_group_type` | Consistency grouping (`repurposing`, `uniprot`, `moa`, etc.) |
| `distance` | `_distance` | Similarity metric (`cosine`, `abs_cosine`) |

When querying `copairs_results.duckdb`, always filter by these `_*` columns to get the exact configuration you need.

## Exploring Configs

### List Available Config Groups

```bash
# See all config group directories
ls configs/copairs/metrics/

# Output: activity.yaml, consistency.yaml, activity_params/, columns/,
#         dataset/, filter/, preprocessing_01_metadata/,
#         preprocessing_02_core/, target_column/
```

### View Main Configs

```bash
# Activity analysis config
cat configs/copairs/metrics/activity.yaml

# Consistency analysis config
cat configs/copairs/metrics/consistency.yaml
```

### Explore Config Groups

```bash
# List datasets
ls configs/copairs/metrics/dataset/

# View a specific dataset config
cat configs/copairs/metrics/dataset/compound_no_source7.yaml

# List preprocessing options
ls configs/copairs/metrics/preprocessing_02_core/

# View a preprocessing pipeline
cat configs/copairs/metrics/preprocessing_02_core/activity_no_target2.yaml

# List activity parameter variants
ls configs/copairs/metrics/activity_params/

# View cross-source reproducibility config
cat configs/copairs/metrics/activity_params/crosssource.yaml

# List target columns for consistency
ls configs/copairs/metrics/target_column/

# View a target column config
cat configs/copairs/metrics/target_column/uniprot.yaml
```

### View the Run Matrix

The run matrix defines all combinations of configs to run:

```bash
cat configs/copairs/copairs_runs.yaml
```

This shows the complete list of activity and consistency runs with their parameter combinations.

## Config Architecture

This repository uses Hydra's config groups for modular composition:

```
configs/copairs/metrics/
├── activity.yaml          # Main config (uses defaults from groups)
├── consistency.yaml       # Main config for consistency
└── <group_name>/          # Config groups
    └── <option>.yaml      # Individual options
```

### How Composition Works

When running copairs-runner:
```bash
pixi run copairs-runner \
  --config-dir configs/copairs/metrics --config-name activity \
  dataset=compound_no_source7 \
  columns=feat_all \
  preprocessing_02_core=activity_no_target2 \
  filter=all_sources \
  activity_params=default
```

Hydra:
1. Loads `activity.yaml` as the base config
2. Merges in `dataset/compound_no_source7.yaml`
3. Merges in `columns/feat_all.yaml`
4. Merges in `preprocessing_02_core/activity_no_target2.yaml`
5. Merges in `filter/all_sources.yaml`
6. Merges in `activity_params/default.yaml` (uses `@package _global_` to inject at root)
7. Resolves all interpolations (e.g., `${dataset.path}`)

### Output Path Convention

Output paths are constructed from config names:
```
data/processed/copairs/runs/activity/{dataset}__{columns}__{preprocessing}__{filter}__{activity_params}/results/
```

This creates unique, traceable output directories for each parameter combination.

## Adding New Configurations

### New Dataset

1. Create `configs/copairs/metrics/dataset/new_dataset.yaml`:
```yaml
name: new_dataset
path: "data/raw/profiles/new_dataset.parquet"
description: "Description of the dataset"
```

2. Add entries to `configs/copairs/copairs_runs.yaml`

### New Target Column (for Consistency)

1. Ensure the column exists in the augmented database
2. Add merge step to `preprocessing_01_metadata/jump_with_annotations.yaml`
3. Create `configs/copairs/metrics/target_column/new_target.yaml`:
```yaml
name: new_target
column: Metadata_new_column_name
description: "Description of the annotation source"
```

4. Add entries to `configs/copairs/copairs_runs.yaml`

### New Preprocessing Variant

Create `configs/copairs/metrics/preprocessing_02_core/new_preprocessing.yaml` with the desired sequence of preprocessing steps.

### New Activity Params Variant

Create `configs/copairs/metrics/activity_params/new_variant.yaml`:
```yaml
# @package _global_
# This directive injects the config at the root level

activity_params:
  name: "new_variant"

average_precision:
  params:
    pos_sameby: [...]
    pos_diffby: [...]
    neg_sameby: [...]
    neg_diffby: [...]
    batch_size: 1000

mean_average_precision:
  params:
    sameby: [...]
    null_size: 100000
    threshold: 0.1
    seed: 12527
```

## Inspecting Results

### Find Completed Runs

```bash
# List all activity runs
ls data/processed/copairs/runs/activity/

# List all consistency runs
ls data/processed/copairs/runs/consistency/

# View a specific run's config
cat data/processed/copairs/runs/activity/compound_no_source7__feat_all__activity_no_target2__all_sources__default/results/.hydra/config.yaml
```

### Query the Results Database

```python
import duckdb

con = duckdb.connect("data/processed/copairs_results.duckdb")

# See available tables
con.sql("SHOW TABLES").show()

# See table schemas
con.sql("DESCRIBE activity_results").show()
con.sql("DESCRIBE consistency_results").show()

# See column comments (documentation)
con.sql("""
    SELECT column_name, comment
    FROM duckdb_columns()
    WHERE table_name = 'activity_results'
""").show()
```

## Performance Troubleshooting

### Diagnosing Slow Runs

If a copairs run is taking too long or hanging, the issue is usually **too many positive pairs**. The number of pairs scales quadratically with replicates.

**Check pair counts before running:**

```python
import pandas as pd

# Load your data after preprocessing filters
df = pd.read_parquet("data/raw/profiles/compound_no_source7.parquet")
# ... apply your filters ...

# Count replicates per group
poscons = df[df['Metadata_pert_type'] == 'poscon']
by_compound_source = poscons.groupby(['Metadata_JCP2022', 'Metadata_Source']).size()

# Estimate cross-source pairs (for crosssource activity_params)
# Formula: sum of n_i * n_j for all source pairs i != j
for compound in poscons['Metadata_JCP2022'].unique():
    counts = by_compound_source[compound].values
    cross_pairs = sum(counts[i] * counts[j]
                      for i in range(len(counts))
                      for j in range(len(counts)) if i != j)
    print(f"{compound}: {cross_pairs:,} cross-source pairs")
```

**Rule of thumb:**
- < 1M pairs: runs in seconds
- 1-10M pairs: runs in minutes
- 10-100M pairs: runs in tens of minutes
- > 100M pairs: consider aggregation

### Aggregation Strategy

When you have dense replication (e.g., poscons with ~7,000 profiles per compound across sources), use **plate-level aggregation** to reduce pairs:

**Before:** 55,180 poscon profiles → ~333M cross-source pairs → ~100 min runtime
**After:** 11,340 plate-level profiles → ~14M pairs → ~20 sec runtime

The preprocessing step `aggregate_replicates` computes the median of features per group:

```yaml
# In preprocessing_02_core config
- type: aggregate_replicates
  params:
    groupby:
      - Metadata_JCP2022
      - Metadata_Source
      - Metadata_Plate
      - Metadata_pert_type
```

This aggregates all wells of the same compound on the same plate to a single plate-level profile.

**When to aggregate:**
- Poscon activity analysis (high replication per plate)
- Cross-source comparisons with densely replicated compounds
- Any analysis where pair counts exceed ~100M

**When NOT to aggregate:**
- Standard compound activity (most compounds have ~1 replicate per plate anyway)
- When you need well-level resolution

### Per-Source Diagnostics

To identify problematic sources, ensure mAP groups by both compound AND source:

```yaml
# In activity_params config
mean_average_precision:
  params:
    sameby: ["Metadata_JCP2022", "Metadata_Source"]  # Per-source results
```

Then compare within-source vs cross-source mAP per source to find outliers:

```python
import pandas as pd

within = pd.read_csv(".../withinsource/results/activity_map_results.csv")
cross = pd.read_csv(".../crosssource/results/activity_map_results.csv")

merged = within.merge(cross, on=['Metadata_JCP2022', 'Metadata_Source'],
                      suffixes=('_within', '_cross'))
merged['mAP_diff'] = merged['mean_average_precision_within'] - merged['mean_average_precision_cross']

# Sources with negative mAP_diff have higher within-source variability
by_source = merged.groupby('Metadata_Source')['mAP_diff'].mean().sort_values()
print(by_source)
```

## Redun Integration

Copairs runs are orchestrated by `workflow.py` via redun tasks:

```bash
# Run copairs pipeline
just redun core

# Preview what will run
just redun core --dry-run
```

The workflow:
1. Reads the run matrix from `configs/copairs/copairs_runs.yaml`
2. Generates all activity/consistency runs via `@task` functions
3. Aggregates results into `data/processed/copairs_results.duckdb`
