# Copairs Metrics Configuration

Hydra config groups for modular copairs analysis. Each subdirectory is a config group that can be composed at runtime.

## Adding New Configs

### Add a new dataset
Create `dataset/your_dataset.yaml`:
```yaml
name: your_dataset
path: "data/raw/profiles/your_dataset.parquet"
description: "Description of your dataset"
```

### Add a new filter
Create `filter/your_filter.yaml`:
```yaml
name: your_filter
query: "Metadata_Source IN ('source_3', 'source_4')"
description: "What this filter does"
```

### Add a new preprocessing pipeline
Create `preprocessing/your_preprocessing.yaml`:
```yaml
name: your_preprocessing
steps:
  - type: merge_metadata
    params:
      source: "data/external/jump_metadata.duckdb"
      table: "plate"
      on_columns: "Metadata_Plate"
  - type: filter
    params:
      query: "Metadata_PlateType != 'TARGET2'"
```

### Add a new target column (consistency only)
1. Create `target_column/your_target.yaml`:
```yaml
name: your_target
column: Metadata_your_target_column
description: "Your target annotation"
```

2. Update preprocessing pipeline to merge your target data.
In `preprocessing/consistency_*.yaml`, add merge step:
```yaml
  - type: merge_metadata
    params:
      source: "data/interim/jump_metadata_augmented.duckdb"
      table: "your_target_table"
      on_columns: "Metadata_JCP2022"
      how: "left"
```

## Usage

Compose configs at runtime:
```bash
pixi run copairs-runner --config-dir configs/copairs/metrics \
  --config-name activity \
  dataset=your_dataset \
  filter=your_filter \
  preprocessing=your_preprocessing \
  columns=feat_all
```
