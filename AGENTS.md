# AGENTS.md - jpx

Project-specific guidance for agents working in this repository.
This is the production pipeline and analysis catalog for JUMP Cell Painting data.
It follows the same catalog pattern as [jx](https://github.com/broadinstitute/jx), [fgx](https://github.com/broadinstitute/fgx), [prx](https://github.com/broadinstitute/prx), and [dmx](https://github.com/broadinstitute/dmx).

`README.md` is the human entry point.
The skills under `.claude/skills/` are the operational entry points: `getting-started` for first-run setup, `compose-notebook` for marimo composition, `copairs` for mAP analysis, and `add-annotation` for integrating new data sources.

## Architecture

Catalog over library.
All analysis logic lives in 43 marimo notebooks (`notebooks/nb00_*.py` through `nb42_*.py`).
Helpers are `@app.function` cells, importable by other notebooks via `sys.path` insertion.
No `src/` package, no CLI wrappers.

Pipeline orchestration is in `workflow.py` (65 redun `@task()` functions).
Cross-environment execution uses `run_task.py` to call notebook helpers from different pixi environments (rapids, cheminformatics, chemberta, deepchem).

Two layers:
1. **Pipeline** - `workflow.py` orchestrates data processing, copairs analysis, and figure generation via redun.
2. **Catalog** - the same notebooks are interactive marimo vignettes for exploration and agent-driven composition.

## Launching notebooks

There are two launch paths depending on whether the notebook needs conda packages.

**Pure-Python notebooks** (no GPU, no conda deps):

```bash
uvx marimo@latest edit --sandbox notebooks/nbNN_*.py
```

`--sandbox` provisions the PEP 723 inline dependencies into an isolated env.
Without it, notebooks fail with `ModuleNotFoundError`.

**GPU / conda-dependent notebooks** (rapids, cheminformatics, chemberta, deepchem):

```bash
pixi run -e rapids marimo edit notebooks/nbNN_*.py
```

Do NOT use `--sandbox` with pixi-managed notebooks - it shadows conda packages and causes silent failures.

| Notebook | Env | Launch |
|---|---|---|
| nb00-nb07, nb10-nb16, nb19, nb22-nb24, nb26, nb28, nb30-nb33, nb35, nb41-nb42 | CPU | `uvx marimo@latest edit --sandbox` |
| nb08-nb09, nb17, nb20, nb25, nb38-nb40 | rapids | `pixi run -e rapids marimo edit` |
| nb18, nb27, nb29, nb34, nb36-nb37 | cheminformatics | `pixi run -e cheminformatics marimo edit` |
| nb37 (`compute_chemberta`) | chemberta | `pixi run -e chemberta marimo edit` |
| nb21 | deepchem | `pixi run -e deepchem marimo edit` |

nb06 `run_stratification_demo()` requires the deepchem env when called from workflow.py; interactive use is CPU-only.
nb32 has a lazy rdkit import used only when called from workflow.py in the cheminformatics env; interactive use is CPU-only.
nb37 spans two envs: properties/Morgan functions use cheminformatics, ChemBERTa embedding uses chemberta.

## Validation rule

After composing or editing any notebook in `notebooks/`, launch it and run all cells before reporting the task complete.
Static checks do not catch wrong outputs, empty tables, or broken plots.

```bash
pixi run ruff check notebooks/
pixi run ruff format notebooks/
```

## Running the pipeline

```bash
just get-inputs             # Download input data from public S3
just redun main             # Full pipeline (copairs + all analysis)
just redun batch_source7    # Batch correction sidebar
just redun explore          # Exploration notebooks
just redun srijit_all       # Srijit's analysis notebooks
just redun main --dry-run   # Preview what will run
```

To skip the pipeline and use pre-computed results: `just get-results`.

Full rebuild from scratch takes ~13 minutes on a 4x H100 server.
Order: `main`, then `batch_source7`, `explore`, `srijit_all`.

## When the question fits the catalog

| Need | Start here |
|---|---|
| Path constants, DPI, CPU count | `nb00_ss_config` |
| Cosine vs abs-cosine comparison | `nb01_ss_cosine_comparison` |
| DuckDB queries (activity, consistency, reproducibility) | `nb02_ss_queries` |
| Load profiles, join metadata/activity | `nb03_ss_profiles` |
| Plot functions (scatter, box, violin, dotplot, heatmap) | `nb04_ss_visualization` |
| Threshold sensitivity | `nb05_ss_threshold_sweep` |
| Stratification effect | `nb06_ss_stratification` |
| Circos experiment metadata | `nb07_nc_circos` |
| UMAP seed stability | `nb08_ss_umap_seed_scan` |
| Source 7 batch effect (iLISI, kBET) | `nb09_ss_source7_batch` |
| Profile comparison (CP vs DL) | `nb10_jfh_profile_comparison` |
| Source 7 4-way comparison | `nb11_ss_source7_4way_comparison` |
| Harmony batch correction comparison | `nb12_ss_harmony_comparison` |
| UMAP gallery | `nb13_ss_umap_visualization` |
| Cross-source reproducibility | `nb14_ss_cross_source_reproducibility` |
| Cell count and data quality | `nb15_ss_cell_count_quality` |
| Target consistency (volcano, dotplot) | `nb16_ss_target_consistency` |
| Chemical space (properties, scaffolds) | `nb17_ss_chemical_space` |
| Fingerprint metrics (Morgan, MinHash) | `nb18_ss_fingerprint_metrics` |
| Compound traits vs activity | `nb19_ss_compound_traits` |
| Structure-morphology kNN overlap | `nb20_ss_structure_morphology` |
| Phenotype prediction (XGBoost) | `nb21_ss_phenotype_prediction` |
| Activity-consistency relationship | `nb22_ss_activity_consistency_relationship` |
| Power analysis | `nb23_ss_power_analysis` |
| PAINS prediction | `nb24_seal_pains_prediction` |
| Activity cliffs (GPU permutation) | `nb25_seal_activity_cliffs` |
| PhenoSeeker scaffold cliffs | `nb26_seal_phenoseeker_cliffs` |
| SAR via BRICS fragments | `nb27_seal_sar_vignette` |
| Toxicity/PK prediction | `nb28_seal_toxicity_pk` |
| Commercial compound curation | `nb29_seal_commercial_compounds` |
| MitoTox morphology clustering | `nb30_seal_mitotox_morphology` |
| MMP9 inhibitor identification | `nb31_seal_mmp9_inhibitors` |

Processing notebooks (nb32-nb42) are called by `workflow.py` and rarely used interactively.

Read `.claude/skills/compose-notebook/SKILL.md` before writing new notebook code.

## Data sources

Two DuckDB databases are the primary query targets:

**`data/interim/jump_metadata_augmented.duckdb`** - augmented metadata.
Contains compound annotations, target mappings (ChEMBL, Repurposing Hub, Chemical Probes, MOTIVE, ToxCast), compound properties, cell counts, image dimensions.
Query `SELECT * FROM duckdb_tables()` and `SELECT * FROM duckdb_columns()` to explore.

**`data/processed/copairs_results.duckdb`** - copairs analysis results.
Four tables: `activity_results`, `activity_scores`, `consistency_results`, `consistency_scores`.
Each has config metadata columns prefixed with `_` (`_dataset`, `_preprocessing`, `_filter`, `_group_type`, `_distance`, `_activity_params`).
Always filter carefully - the database contains results from many run configurations.
Read `configs/copairs/copairs_runs.yaml` to understand which configurations exist.

Key conventions:
- Use `mean_normalized_average_precision` (not raw mAP) and `corrected_p_value` (not raw p-value)
- Filter `_preprocessing NOT LIKE '%_sweep'` for regular runs vs sweep runs
- AnnData obs column is `JCP2022` (not `Metadata_JCP2022`)
- Load profiles via `nb03_ss_profiles.load_profiles(name, level)`

## Conventions

- **pixi** for package management (multi-environment: default, rapids, cheminformatics, chemberta, deepchem)
- **marimo** for notebooks (reactive, git-friendly, pure Python)
- **DuckDB** for all database queries
- **ruff** for linting/formatting
- Notebook naming: `nbNN_initials_description.py`
- `@app.function` for hoistable helpers, `app.setup` for shared imports
- nb04 plot functions return figures, never close them
