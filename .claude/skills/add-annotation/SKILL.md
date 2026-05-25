---
name: add-annotation
description: Add a new annotation source to the JUMP pipeline. Use when integrating external databases like ToxCast, BindingDB, or other compound annotation sources. Guides through notebook creation, workflow.py task wiring, and SQL schema.
allowed-tools: Read, Write, Edit, Glob, Grep, Bash(pixi run python:*), Bash(pixi run ruff:*), WebFetch, WebSearch
---

# Add Annotation Source

Guide for integrating a new external annotation source into the JUMP pipeline.
The pipeline uses marimo notebooks for processing and redun for orchestration.

## Overview

Adding a new annotation source requires modifying 3 files and creating 1:

| File | Purpose |
|---|---|
| `notebooks/nbNN_ss_<source>.py` (new) | Marimo notebook with `@app.function` processing helper |
| `workflow.py` | Add `@task()` that calls the notebook helper |
| `scripts/augment_metadata_db.sql` | SQL schema, import, indexes, verification |
| `configs/copairs/copairs_runs.yaml` | Add target column if this is a consistency source |

## Step 1: Research the Data Source

Before coding, gather this information:

1. **Download URL**: Direct link to data file (CSV, TSV, Parquet, Excel)
2. **Data format**: Column names, especially compound identifiers
3. **Compound matching**: Does it have InChIKey, SMILES, or other identifiers?
4. **License**: Ensure data can be redistributed or used
5. **Fields of interest**: Which columns are valuable for the pipeline?

Ask the user if any of this information is missing.

## Step 2: Create Processing Notebook

Create `notebooks/nbNN_ss_<source>.py` as a marimo notebook.
Use nb35_ss_toxcast.py as the reference template - it shows the current pattern.

Key elements:
- PEP 723 inline script metadata for `--sandbox` launches
- `with app.setup` block importing from `nb00_ss_config`
- `@app.function` for the processing helper (this is what `workflow.py` imports)
- The function should download (or read from `data/external/`), process, match to JUMP compounds, and write to `data/interim/`

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "python-dotenv",
#     "loguru",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    import pandas as pd
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR


@app.function
def process_<source>():
    """Process <Source> annotations and match to JUMP compounds."""
    input_file = EXTERNAL_DATA_DIR / "<source>_annotations.csv"
    output_file = INTERIM_DATA_DIR / "<source>_processed.csv"

    logger.info(f"Loading {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} records")

    # Load JUMP compounds for InChIKey14 matching
    compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    jump = pd.read_csv(compound_file, usecols=["Metadata_JCP2022", "Metadata_InChIKey"])
    jump["inchikey_prefix"] = jump["Metadata_InChIKey"].str[:14]

    # Match via InChIKey14 (stereoisomer-tolerant)
    df["inchikey_prefix"] = df["<inchikey_column>"].str[:14]
    matched = df.merge(jump[["Metadata_JCP2022", "inchikey_prefix"]], on="inchikey_prefix", how="inner")
    logger.info(f"Matched {matched['Metadata_JCP2022'].nunique()} JUMP compounds")

    # Select and rename columns with Metadata_<prefix>_ convention
    result = matched[["Metadata_JCP2022", ...]].rename(columns={...})

    result.to_csv(output_file, index=False)
    logger.success(f"Saved {len(result)} records to {output_file}")
    return output_file
```

### Column Naming Convention

Use prefix `Metadata_<abbrev>_<field>`:

- `Metadata_txcst_*` for ToxCast
- `Metadata_bdb_*` for BindingDB
- Keep abbreviations short (4-6 chars)

### Matching Approaches

| Source has | Use | Notes |
|---|---|---|
| InChIKey | **InChIKey14** (preferred) | First 14 chars = connectivity layer, stereoisomer-tolerant |
| SMILES only | **SMILES standardization** | Via `jump_smiles` for max matches |
| Only DB IDs | **unichem_pointers.csv** | Fallback when no structure info |

## Step 3: Add Redun Task to workflow.py

Add a `@task()` function in Section 2 (external data processing) of `workflow.py`.
Follow the pattern of existing annotation tasks:

```python
@task()
def process_<source>() -> File:
    from nbNN_ss_<source> import process_<source>

    output = process_<source>()
    return File(str(output))
```

If the notebook needs conda packages (rdkit, cheminformatics), use `_run_in_env` instead:

```python
@task()
def process_<source>() -> File:
    _run_in_env("cheminformatics", "nbNN_ss_<source>", "process_<source>")
    return File("data/interim/<source>_processed.csv")
```

Then add it as an input to `augment_metadata_db`:

1. Add parameter to `augment_metadata_db()` function signature
2. Add the call in the `core_pipeline()` or `main_pipeline()` function where other annotation tasks are wired

## Step 4: Add SQL Schema

Add to `scripts/augment_metadata_db.sql` in the appropriate sections:

### 4a. Table Definition (Section 1)

```sql
-- <Source Name> annotations
CREATE TABLE <source>_annotations (
    Metadata_JCP2022 VARCHAR,
    Metadata_<prefix>_<field1> VARCHAR,
    FOREIGN KEY (Metadata_JCP2022) REFERENCES compound(Metadata_JCP2022)
);

COMMENT ON TABLE <source>_annotations IS '<Description>';
COMMENT ON COLUMN <source>_annotations.Metadata_JCP2022 IS 'JUMP compound ID';
```

### 4b. Data Import (Section 2)

```sql
INSERT INTO <source>_annotations
SELECT * FROM read_csv_auto('data/interim/<source>_processed.csv')
WHERE Metadata_JCP2022 IN (SELECT Metadata_JCP2022 FROM compound);
```

### 4c. Indexes (Section 4)

```sql
CREATE INDEX idx_<source>_jcp ON <source>_annotations(Metadata_JCP2022);
```

### 4d. Verification Queries (Section 5)

Add to the row counts UNION:

```sql
UNION ALL SELECT '<source>_annotations', COUNT(*) FROM <source>_annotations
```

Add to compound coverage UNION:

```sql
UNION ALL SELECT '<source>_annotations', COUNT(DISTINCT Metadata_JCP2022) FROM <source>_annotations
```

## Step 5: Add to Copairs (if consistency source)

If this annotation provides target groupings for consistency analysis:

1. Add a `target_column` config: `configs/copairs/metrics/target_column/<source>.yaml`
2. Add the column to `preprocessing_01_metadata/jump_with_annotations.yaml` merge step
3. Add consistency runs to `configs/copairs/copairs_runs.yaml`

## Step 6: Test the Integration

```bash
# 1. Download the data (manual or scripted into the notebook)
# Place raw file in data/external/

# 2. Run just the processing notebook
pixi run python -c "
import sys; sys.path.insert(0, 'notebooks')
from nbNN_ss_<source> import process_<source>
process_<source>()
"

# 3. Rebuild the augmented database
just redun augment_metadata_db

# 4. Verify in DuckDB
duckdb data/interim/jump_metadata_augmented.duckdb -c "SELECT COUNT(*) FROM <source>_annotations"
```

## Reference: Existing Sources

| Source | Notebook | Prefix | Matching |
|---|---|---|---|
| ChEMBL | nb33 | (none) | Direct JCP2022 |
| Repurposing Hub | nb33 | `repurposing_` | InChIKey14 |
| Chemical Probes | nb34 | `chmprb_` | SMILES standardization |
| MOTIVE | nb33 | `motive_` | InChIKey14 |
| ToxCast | nb35 | `txcst_` | InChIKey14 |
| Toxicity/PK | nb36 | various | SMILES standardization |

## Checklist

Before marking complete, verify:

- [ ] Marimo notebook created with `@app.function` processing helper
- [ ] Proper column naming (`Metadata_<prefix>_<field>`)
- [ ] `workflow.py`: `@task()` function added
- [ ] `workflow.py`: wired into `augment_metadata_db` inputs
- [ ] SQL: CREATE TABLE with FOREIGN KEY and COMMENTs
- [ ] SQL: INSERT statement with JCP2022 filter
- [ ] SQL: CREATE INDEX for Metadata_JCP2022
- [ ] SQL: Verification queries updated
- [ ] Test: Processing function runs without errors
- [ ] Test: Row counts look reasonable
