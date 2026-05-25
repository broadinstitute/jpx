---
name: compose-notebook
description: Compose a new marimo notebook in jpx by reusing @app.function helpers from the existing catalog (notebooks/nb00_*.py onward) to answer a JUMP Cell Painting question - e.g. "compare these distance metrics", "show consistency for these targets", "plot activity across datasets", "visualize UMAPs colored by X". Trigger whenever the user asks for a notebook, analysis, figure, or vignette that touches JUMP profiles, copairs results, mAP values, target consistency, compound metadata, or morphological activity - even if they don't say "marimo". Use this instead of writing standalone scripts. Also trigger when the user says "write me a notebook that..." or asks to build on existing catalog notebooks.
---

# Compose a new marimo notebook from the jpx catalog

## What this skill is for

jpx has an all-marimo catalog of reactive notebooks.
The catalog holds reactive notebooks whose `@app.function` helpers are importable by other notebooks.
When the user asks an analytical question, compose from the catalog rather than writing standalone scripts.

## The catalog

Every catalog file uses `with app.setup` for shared imports and `@app.function` for reusable helpers.
The functions are the contract; UI cells are illustrative.

### Foundation notebooks (always import from these)

| Module | Reusable exports | What they provide |
|---|---|---|
| `nb00_ss_config` | `PROJ_ROOT`, `DATA_DIR`, `RAW_DATA_DIR`, `INTERIM_DATA_DIR`, `PROCESSED_DATA_DIR`, `EXTERNAL_DATA_DIR`, `ACTIVITY_DIR`, `METADATA_DB`, `COPAIRS_RESULTS_DB`, `ANNDATA_DIR`, `DEFAULT_DPI`, `JUMP_CPUS` | All paths and constants. Every notebook imports from here. |
| `nb02_ss_queries` | `query_activity_results(dataset, preprocessing, filter_name, activity_params)`, `query_cross_source_reproducibility(dataset, preprocessing, filter_name)`, `query_consistency_results(dataset, preprocessing, filter_name, group_type, distance)` | DuckDB queries against copairs_results.duckdb. Always use these instead of raw SQL. |
| `nb03_ss_profiles` | `load_profiles(name, level)`, `join_metadata(adata, modality, level)`, `load_umap(dataset, level, metric, filter_name)`, `join_activity(adata, dataset, preprocessing, filter_name)` | Load h5ad profiles as AnnData, join metadata and activity from DuckDB. |
| `nb04_ss_visualization` | Constants: `SIG_COLOR`, `NONSIG_COLOR`, `BBOX_STYLE`, `MAP_COL`, `SIG_COL`. Dataclass: `TraitConfig`. Stats: `split_by_significance()`, `compute_activity_stats()`, `compute_partial_correlation()`, `format_stat_annotation()`. Plots: `plot_categorical_box()`, `plot_scatter_with_marginals()`, `plot_hexbin_with_marginals()`, `compute_umap_bounds()`, `plot_violin_by_group()`, `plot_survival_curve()`, `plot_activity_rate_by_bin()`, `plot_dotplot()`, `plot_plate_heatmap()` | All visualization constants, statistical helpers, and plot functions. |

### Analysis notebooks (early)

| Module | Reusable functions | What they do |
|---|---|---|
| `nb01_ss_cosine_comparison` | `query_cosine_comparison(dataset)`, `plot_cosine_comparison(df, dataset)` | Query cosine vs abs_cosine consistency nMAP, produce significance-colored scatter. |
| `nb05_ss_threshold_sweep` | `query_threshold_sweep(dataset)`, `plot_threshold_summary(df)`, `plot_threshold_tradeoff(df)` | Threshold sensitivity analysis: how activity p-value threshold affects consistency. |
| `nb06_ss_stratification` | (no reusable functions - demo notebook) | Demonstrates why stratification had minimal effect on ROC-AUC with 112K samples. |
| `nb07_nc_circos` | `zero_pad_source(source)`, `load_circos_data()`, `plot_plate_metadata(plate_df)`, `plot_experiment_properties(cp_scope_df, sources)` | Circos plots of JUMP experiment metadata (plate types, microscope configs). |
| `nb08_ss_umap_seed_scan` | `compute_umap(adata, n_neighbors, n_pcs, min_dist, metric, random_state)`, `save_minimal_h5ad(adata, output_path)`, `compute_layout_similarity(results, metrics, seeds)` | UMAP seed stability: 5 seeds x 2 metrics, GPU/CPU fallback, Procrustes analysis. |
| `nb09_ss_source7_batch` | `compute_knn_graph(adata, n_neighbors)`, `compute_ilisi(adata, batch_key)`, `compute_kbet(adata, batch_key, subsample, alpha, seed)`, `process_dataset(label, dataset, n_neighbors, subsample_kbet)` | Source 7 batch effect: iLISI + kBET on CP vs DL x Source 7. GPU-only (rapids_singlecell). |

### Analysis notebooks (bulk analysis)

| Module | Env | What it does |
|---|---|---|
| `nb10_jfh_profile_comparison` | CPU | Compare activity/consistency between two profile types (CP vs DL). UpSet plots. |
| `nb11_ss_source7_4way_comparison` | CPU | 4-way comparison: CP/DL x with/without source 7. Status change tracking. |
| `nb12_ss_harmony_comparison` | CPU | Compare 6 Harmony batch-correction configurations. |
| `nb13_ss_umap_visualization` | CPU | UMAP gallery: 8 annotation colorings per dataset/metric. |
| `nb14_ss_cross_source_reproducibility` | CPU | Within-source vs cross-source activity scatter/box/diff plots. |
| `nb15_ss_cell_count_quality` | CPU | Cell density, plate edge effects, activity confounding. |
| `nb16_ss_target_consistency` | CPU | Volcano, dot plot, UMAP grid for target consistency annotations. |
| `nb17_ss_chemical_space` | rapids | Chemical property distributions, scaffold diversity, structure/property UMAPs. GPU UMAP with CPU fallback. |
| `nb18_ss_fingerprint_metrics` | cheminformatics | Morgan fingerprint transformations: L2, MinHash, IDF, random features vs Tanimoto. |
| `nb19_ss_compound_traits` | CPU | Compound property correlations with phenotypic activity. |
| `nb20_ss_structure_morphology` | rapids | kNN overlap between structural and morphological neighbors. GPU-accelerated. |
| `nb21_ss_phenotype_prediction` | deepchem | Predict mAP from structure (XGBoost, scaffold splits). Requires rdkit from conda. |
| `nb22_ss_activity_consistency_relationship` | CPU | Activity-consistency relationship: enrichment, survival, dose-response. |
| `nb23_ss_power_analysis` | CPU | Statistical power: how many perturbations needed per target group. |

### Srijit Seal's notebooks

| Module | Env | What it does |
|---|---|---|
| `nb24_seal_pains_prediction` | CPU | Predict PAINS structural alerts from morphological profiles (HistGBT, cross-validation). |
| `nb25_seal_activity_cliffs` | rapids | Activity cliffs: structurally similar but morphologically different compound pairs. GPU-accelerated permutation tests. |
| `nb26_seal_phenoseeker_cliffs` | CPU | PhenoSeeker-style scaffold-based activity cliff detection with null separation models. |
| `nb27_seal_sar_vignette` | cheminformatics | SAR via BRICS fragment enumeration - structural alerts predicting phenotypic activity. |
| `nb28_seal_toxicity_pk` | CPU | Predict DILI, DICT, MitoTox, PK endpoints from morphology, fingerprints, or both. |
| `nb29_seal_commercial_compounds` | cheminformatics | Curate commercially available compounds with distinctive Cell Painting profiles. |
| `nb30_seal_mitotox_morphology` | CPU | Cluster mitotoxicants by morphology, test functional submechanism separation. |
| `nb31_seal_mmp9_inhibitors` | CPU | Identify target gene inhibitors among JUMP compounds with MoA context and UMAP. |

### Processing notebooks (nb32-nb42)

Called by `workflow.py` and rarely used interactively.
Their `@app.function` helpers are importable but typically invoked via redun tasks.

| Module | Env | What it does |
|---|---|---|
| `nb32_ss_processing_utils` | CPU | Metadata processing utilities. Lazy rdkit import used only via workflow.py cheminformatics env. |
| `nb33_ss_annotation_processing` | CPU | Annotation merging and augmentation for metadata DB. |
| `nb34_ss_probe_curation` | cheminformatics | Chemical probe and kinase inhibitor curation (Probes & Drugs, KCGS). |
| `nb35_ss_toxcast` | CPU | ToxCast assay endpoint processing. |
| `nb36_ss_toxicity_annotations` | cheminformatics | Toxicity and PK annotation processing (DILI, MitoTox). |
| `nb37_ss_compound_featurization` | cheminformatics / chemberta | Properties and Morgan fingerprints (cheminformatics), ChemBERTa embeddings (chemberta). |
| `nb38_ss_batch_correction` | rapids | Harmony batch correction on GPU. |
| `nb39_ss_profile_conversion` | rapids | Profile format conversion (h5ad generation). |
| `nb40_ss_umap_pipeline` | rapids | UMAP computation pipeline on GPU. |
| `nb41_ss_copairs_db` | CPU | Copairs results database assembly. |
| `nb42_ss_profile_filtering` | CPU | Profile filtering (source exclusions). |

### Naming convention

**Format:** `nbNN_initials_description.py`

- `nbNN` - sequential number, zero-padded (nb01, nb02, ...)
- `initials` - author's initials in lowercase (ss, seal, jfh, nc)
- `description` - short snake_case description

## Data sources

Two DuckDB databases are the primary query targets:

**`data/interim/jump_metadata_augmented.duckdb`** - augmented metadata.
Contains compound annotations, target mappings (ChEMBL, Repurposing Hub, Chemical Probes, Motive, ToxCast), compound properties, cell counts, image dimensions.
Query `SELECT * FROM duckdb_tables()` and `SELECT * FROM duckdb_columns()` to explore.

**`data/processed/copairs_results.duckdb`** - copairs analysis results.
Four tables: `activity_results`, `activity_scores`, `consistency_results`, `consistency_scores`.
Each has config metadata columns prefixed with `_` (`_dataset`, `_preprocessing`, `_filter`, `_group_type`, `_distance`, `_activity_params`, etc.).
Always filter carefully - the database contains results from many run configurations.
Read `configs/copairs/copairs_runs.yaml` to understand which configurations exist.

Key conventions:
- Use `mean_normalized_average_precision` (not raw mAP) and `corrected_p_value` (not raw p-value)
- Filter `_preprocessing NOT LIKE '%_sweep'` for regular runs vs sweep runs
- AnnData obs column is `JCP2022` (not `Metadata_JCP2022`)

**Profiles** - h5ad files under `data/interim/anndata/`. Load via `nb03_ss_profiles.load_profiles(name, level)`.

## Cross-notebook import recipe

Each catalog file uses an `nbNN_` prefix so it's a valid Python module name.
Import helpers by adding `notebooks/` to `sys.path`:

```python
with app.setup:
    import sys
    from pathlib import Path

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import PROJ_ROOT, COPAIRS_RESULTS_DB, DEFAULT_DPI
    from nb02_ss_queries import query_activity_results
    from nb03_ss_profiles import load_profiles, join_metadata
    from nb04_ss_visualization import (
        SIG_COLOR, NONSIG_COLOR, MAP_COL, SIG_COL,
        plot_scatter_with_marginals, plot_dotplot,
    )
```

## The composition pattern

### 1. Setup cell

Use `with app.setup` for shared imports and constants.
Symbols are available to all `@app.function` bodies and `@app.cell` bodies.
No `return` needed.

```python
with app.setup:
    import sys
    from pathlib import Path

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from nb00_ss_config import COPAIRS_RESULTS_DB, DEFAULT_DPI
    from nb04_ss_visualization import SIG_COLOR, MAP_COL, SIG_COL
```

### 2. Narrative first

Every notebook starts with a `mo.md()` cell that states the question, the verdict, and the relevant context.
A notebook without narrative is a script with extra steps.

```python
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cosine vs Absolute Cosine Distance

    **Verdict:** cosine is the better default - most groups lose nMAP under absolute cosine.

    *Related: [GitHub Issue #22](...)*
    """)
    return
```

### 3. Interactive UI

Use `mo.ui.dropdown` for categorical choices (dataset, preprocessing, group_type).
Use `mo.ui.slider` for numeric ranges.

**Populate dropdowns from the data, not hardcoded lists.**
Hardcoded options silently produce empty results when the database doesn't match.

Guard expensive steps with `mo.ui.run_button()` + `mo.stop(not run_button.value)`.

### 4. Hoistable helpers

Define reusable functions with `@app.function`.
They can only reference symbols from `app.setup` or other `@app.function` definitions.
This is what makes them importable from other notebooks.

### 5. PEP 723 inline dependencies

Every notebook needs a script header listing its dependencies.
When marimo launches with `--sandbox`, these are installed automatically into an isolated env.

The deps must include transitive dependencies pulled in by foundation notebook imports.
The sandbox environment is isolated - it can't see packages from the outer pixi env.

| If you import | You must also list |
|---|---|
| `nb00_ss_config` | `python-dotenv`, `loguru` |
| `nb02_ss_queries` | `duckdb`, `pandas` |
| `nb03_ss_profiles` | `anndata`, `duckdb`, `pandas`, `loguru` |
| `nb04_ss_visualization` | `matplotlib`, `numpy`, `pandas`, `scipy`, `seaborn`, `scanpy` |

Do not use exact version pins (e.g., `pandas==3.0.3`) - they cause resolution conflicts between packages.
Do not list conda-only packages (rdkit, deepchem, cupy, rapids_singlecell) - they come from the pixi env.

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "python-dotenv",
#     "loguru",
#     "scanpy",
# ]
# ///
```

For cheminformatics notebooks, add a NOTE comment above `# ///`:
```python
# NOTE: Run with pixi run -e cheminformatics marimo edit/run
# (rdkit is conda-only and comes from the pixi env)
# ///
```

## Launching notebooks

**`marimo run` vs `marimo edit`:** Use `run` to view notebooks (cells auto-execute, app mode). Use `edit` to modify notebooks (cells start stale). Default to `run`.

There are two launch paths depending on whether the notebook needs conda packages (GPU, rapids, scanpy from conda, etc.) or only PyPI packages.

### Pure-Python notebooks (no GPU, no conda deps)

Use `uvx marimo@latest` and `--sandbox` so PEP 723 inline deps get installed into an isolated env.

```bash
nohup uvx marimo@latest run notebooks/nbNN_topic.py \
    --port <PORT> --host 0.0.0.0 --headless --sandbox --no-token \
    > /tmp/marimo-nbNN.log 2>&1 &
```

### GPU / conda-dependent notebooks

Notebooks that import `rapids_singlecell`, `cupy`, `scanpy` (from conda), or any other pixi-managed package cannot use `uvx marimo@latest` or `--sandbox`. Both create isolated environments that shadow the pixi env's packages, causing silent ImportErrors that make cells appear to complete with no errors while doing no work.

marimo is a declared conda dependency in the rapids feature (`pyproject.toml`), so no imperative installs are needed.

**Launch with `pixi run -e rapids marimo`:**
```bash
nohup pixi run -e rapids marimo run notebooks/nbNN_topic.py \
    --port <PORT> --host 0.0.0.0 --headless --no-token \
    > /tmp/marimo-nbNN.log 2>&1 &
```

Do NOT use `uvx marimo@latest` or `--sandbox` - they create isolated envs that shadow the conda packages the notebook needs.

### Common flags and commands

```bash
# Find free ports (other users share the server)
for p in 2731 2732 2733; do
    ss -tlnp | grep -q ":$p " && echo "$p in use" || echo "$p free"
done

# Verify it's up
ss -tlnp | grep :<PORT>

# Access via Tailscale: http://<server>:<PORT>

# Stop when done
fuser -k <PORT>/tcp
```

Key flags:
- `--sandbox` - (pure-Python only) installs PEP 723 inline deps. Without this, you get ImportError for duckdb, pycirclize, etc.
- `--host 0.0.0.0` - binds to all interfaces. Default is localhost-only, unreachable from browser on another machine.
- `--no-token` - disables auth tokens in the URL.
- `--headless` - no auto-open browser (needed on remote servers).

### Which notebooks need which path

| Notebook | Env | Launch path |
|---|---|---|
| nb00-nb07, nb10-nb16, nb19, nb22-nb24, nb26, nb28, nb30-nb33, nb35, nb41-nb42 | CPU | `uvx marimo@latest run --sandbox` |
| nb08-nb09, nb17, nb20, nb25, nb38-nb40 | rapids | `pixi run -e rapids marimo run` |
| nb18, nb27, nb29, nb34, nb36-nb37 | cheminformatics | `pixi run -e cheminformatics marimo run` |
| nb37 (`compute_chemberta`) | chemberta | `pixi run -e chemberta marimo run` |
| nb21 | deepchem | `pixi run -e deepchem marimo run` |

nb06 `run_stratification_demo()` requires deepchem via workflow.py; interactive use is CPU-only.
nb32 has a lazy rdkit import used only via workflow.py; interactive use is CPU-only.
nb37 spans two envs: properties/Morgan use cheminformatics, ChemBERTa uses chemberta.

## Process for a new composition

1. **Map the question to catalog functions.** Which nbNN helpers give you each step?
   If something isn't in the catalog yet, add it as an `@app.function` in a new or existing catalog notebook.
2. **Check the data.** Before writing the full notebook, validate that the columns/tables you need exist:
   ```python
   con = duckdb.connect("data/processed/copairs_results.duckdb", read_only=True)
   print(con.execute("SELECT column_name FROM duckdb_columns() WHERE table_name = 'consistency_results'").fetchall())
   ```
3. **Draft the notebook** with `@app.function` helpers first, then `@app.cell` UI on top.
4. **Verify syntax:** `pixi run python -c "import ast; ast.parse(open('notebooks/nbNN.py').read())"`
5. **Launch and test** using the launch recipe above. After launch, always verify cells actually run: `curl -sf http://localhost:<PORT>/ > /dev/null` to create a session, then use marimo-pair's `execute-code.sh` to run all cells and check for errors. Don't assume a listening port means the notebook works.

## Known gotchas

- **Wrong env = silent failures.** If a GPU/conda notebook runs under `uvx marimo@latest --sandbox`, cells that import conda packages (rapids_singlecell, scanpy, etc.) will show status "idle" with 0 errors but do no work. marimo catches the ImportError inside the cell and reports it as a non-error output. The symptom is: all cells complete in seconds, no output files are created, no error messages. Always verify output files exist after a run - a listening port and idle cells do not mean the notebook worked.
- **Unique variable names across cells.** marimo's `MultipleDefinitionError` fires when two cells define the same name. Prefix cell-local variables with `_` to make them private to that cell. This applies to loop variables, temp DataFrames, and format strings that appear in multiple cells.
  ```python
  # Cell 1: _lines, _pcts_str are cell-local
  _lines = ["### Results\n"]
  _pcts_str = ", ".join(...)
  mo.md("\n".join(_lines))

  # Cell 2: different _ names, no conflict
  _summary_lines = ["### Summary\n"]
  ```
- **Only the last expression renders.** If a cell has `mo.md(intro)` then `mo.vstack([table])`, the intro is silently discarded. Wrap in a single `mo.vstack([mo.md(intro), table])`.
- **Underscore names are ALWAYS cell-local.** `_helper()` defined in one cell or `@app.function` is invisible to all other cells AND all other `@app.function` definitions. This includes `@app.function def _my_helper()` - the underscore prefix makes it private even though it has the decorator. If a function needs to be called from anywhere outside its own definition, its name must NOT start with `_`. Only use underscore prefix for variables that are truly local to one cell.
- **`mo.stop` shows as "exception" in code_mode.** This is expected - downstream cells show "cancelled". Both are normal.
- **DuckDB and underscore variables.** DuckDB's Python-scope lookup can't resolve underscore-prefixed variables in marimo cells due to name mangling. Use non-underscore names for variables referenced in SQL.
- **copairs_results.duckdb has many run configurations.** Always filter by `_dataset`, `_preprocessing`, `_filter`, etc.

## When NOT to use this skill

- Editing Justfile commands or pipeline orchestration (`workflow.py`)
- Pure infrastructure tasks (S3 sync, CI, packaging)
