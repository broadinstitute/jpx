# jpx - JUMP Production eXplore

A reproducible analysis pipeline and agent-composable notebook catalog for [JUMP Cell Painting](https://jump-cellpainting.broadinstitute.org/) - the largest public morphological profiling dataset (~116K compounds, ~8K CRISPR knockouts, ~15K gene overexpressions, 1.6 billion cells).

jpx is a catalog of 43 [marimo](https://marimo.io) notebooks that process, analyze, and visualize JUMP data.
Each notebook is both a runnable analysis and a source of `@app.function` helpers that other notebooks can [import and reuse](https://docs.marimo.io/guides/reusing_functions/).
Given a new biological question, an agent picks relevant notebooks, composes their functions into a new analysis, executes it in a live kernel, and hands back a self-contained, re-runnable result.

Pipeline orchestration uses [redun](https://github.com/insitro/redun) (65 tasks, ~13 min full rebuild on 4x H100).
All data is publicly available on S3 (anonymous access, no credentials needed).

Related catalogs of the same pattern: [jx](https://github.com/broadinstitute/jx) for JUMP exploration, [fgx](https://github.com/broadinstitute/fgx) for FinnGen human genetics, [prx](https://github.com/broadinstitute/prx) for PROSPECT chemical genetics, and [dmx](https://github.com/broadinstitute/dmx) for DepMap Breadbox.

## The catalog

### Foundation notebooks

| Notebook | Role |
|---|---|
| [`nb00_ss_config.py`](notebooks/nb00_ss_config.py) | Path constants and project configuration |
| [`nb02_ss_queries.py`](notebooks/nb02_ss_queries.py) | DuckDB query helpers for copairs results |
| [`nb03_ss_profiles.py`](notebooks/nb03_ss_profiles.py) | Load AnnData profiles, join metadata and activity |
| [`nb04_ss_visualization.py`](notebooks/nb04_ss_visualization.py) | Plot functions (scatter, box, violin, dotplot, heatmap) |

### Analysis notebooks

| Notebook | What it does |
|---|---|
| [`nb01_ss_cosine_comparison.py`](notebooks/nb01_ss_cosine_comparison.py) | Cosine vs absolute cosine distance comparison |
| [`nb05_ss_threshold_sweep.py`](notebooks/nb05_ss_threshold_sweep.py) | Threshold sensitivity analysis |
| [`nb06_ss_stratification.py`](notebooks/nb06_ss_stratification.py) | Stratification effect on ROC-AUC |
| [`nb07_nc_circos.py`](notebooks/nb07_nc_circos.py) | Circos plots of experiment metadata |
| [`nb08_ss_umap_seed_scan.py`](notebooks/nb08_ss_umap_seed_scan.py) | UMAP seed stability (GPU) |
| [`nb09_ss_source7_batch.py`](notebooks/nb09_ss_source7_batch.py) | Source 7 batch effect quantification (GPU) |
| [`nb10_jfh_profile_comparison.py`](notebooks/nb10_jfh_profile_comparison.py) | CP vs DL profile comparison |
| [`nb11_ss_source7_4way_comparison.py`](notebooks/nb11_ss_source7_4way_comparison.py) | 4-way source 7 comparison |
| [`nb12_ss_harmony_comparison.py`](notebooks/nb12_ss_harmony_comparison.py) | Harmony batch correction comparison |
| [`nb13_ss_umap_visualization.py`](notebooks/nb13_ss_umap_visualization.py) | UMAP gallery with 8 annotation colorings |
| [`nb14_ss_cross_source_reproducibility.py`](notebooks/nb14_ss_cross_source_reproducibility.py) | Cross-source activity reproducibility |
| [`nb15_ss_cell_count_quality.py`](notebooks/nb15_ss_cell_count_quality.py) | Cell density and plate edge effects |
| [`nb16_ss_target_consistency.py`](notebooks/nb16_ss_target_consistency.py) | Target consistency (volcano, dotplot, UMAP grid) |
| [`nb17_ss_chemical_space.py`](notebooks/nb17_ss_chemical_space.py) | Chemical property distributions and scaffold diversity (GPU) |
| [`nb18_ss_fingerprint_metrics.py`](notebooks/nb18_ss_fingerprint_metrics.py) | Fingerprint transformation comparison |
| [`nb19_ss_compound_traits.py`](notebooks/nb19_ss_compound_traits.py) | Compound property-activity correlations |
| [`nb20_ss_structure_morphology.py`](notebooks/nb20_ss_structure_morphology.py) | Structure-morphology kNN overlap (GPU) |
| [`nb21_ss_phenotype_prediction.py`](notebooks/nb21_ss_phenotype_prediction.py) | Predict mAP from structure (XGBoost) |
| [`nb22_ss_activity_consistency_relationship.py`](notebooks/nb22_ss_activity_consistency_relationship.py) | Activity-consistency enrichment and survival |
| [`nb23_ss_power_analysis.py`](notebooks/nb23_ss_power_analysis.py) | Statistical power per target group |
| [`nb24_seal_pains_prediction.py`](notebooks/nb24_seal_pains_prediction.py) | PAINS structural alert prediction |
| [`nb25_seal_activity_cliffs.py`](notebooks/nb25_seal_activity_cliffs.py) | Activity cliff detection (GPU) |
| [`nb26_seal_phenoseeker_cliffs.py`](notebooks/nb26_seal_phenoseeker_cliffs.py) | Scaffold-based activity cliffs |
| [`nb27_seal_sar_vignette.py`](notebooks/nb27_seal_sar_vignette.py) | SAR via BRICS fragment enumeration |
| [`nb28_seal_toxicity_pk.py`](notebooks/nb28_seal_toxicity_pk.py) | Toxicity and PK endpoint prediction |
| [`nb29_seal_commercial_compounds.py`](notebooks/nb29_seal_commercial_compounds.py) | Commercial compound curation |
| [`nb30_seal_mitotox_morphology.py`](notebooks/nb30_seal_mitotox_morphology.py) | MitoTox morphology clustering |
| [`nb31_seal_mmp9_inhibitors.py`](notebooks/nb31_seal_mmp9_inhibitors.py) | MMP9 inhibitor identification |

Processing notebooks (nb32-nb42) handle data ingestion and are called by the pipeline, not used interactively.

## Getting started

### Prerequisites

- [pixi](https://pixi.sh) - install with `curl -fsSL https://pixi.sh/install.sh | bash`
- [uv](https://docs.astral.sh/uv/) - install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [rclone](https://rclone.org) - for downloading data from public S3 (no credentials needed)

`just` and `marimo` are installed automatically by `pixi install`.
Run `pixi shell` after install to activate the environment (puts `just`, `marimo`, etc. on PATH).
GPU notebooks (nb08, nb09, nb17, nb20, nb25, nb38-nb40) require NVIDIA GPUs with CUDA; everything else runs on CPU.

### Option 1: Use pre-computed results (fastest)

Downloads ~12 GB of pre-computed results.

```bash
git clone https://github.com/broadinstitute/jpx.git && cd jpx
pixi install
pixi shell
just get-results
uvx marimo@latest edit --sandbox notebooks/nb16_ss_target_consistency.py
```

`uvx marimo@latest --sandbox` runs notebooks with their PEP 723 inline dependencies in an isolated env, independent of the pixi env.

### Option 2: Reproduce from scratch

Downloads ~32 GB of input data, then runs the full pipeline.

```bash
git clone https://github.com/broadinstitute/jpx.git && cd jpx
pixi install
pixi shell
just get-inputs
just redun main
just redun batch_source7
just redun explore
just redun srijit_all
```

Full rebuild takes ~13 minutes on a server with 4x NVIDIA H100 GPUs.

### Option 3: Agent-driven exploration

Clone, open [Claude Code](https://claude.ai/code) inside the repo, and ask a question.
The `compose-notebook` skill (in `.claude/skills/compose-notebook/`) picks relevant catalog notebooks and composes a new analysis.

```bash
git clone https://github.com/broadinstitute/jpx.git && cd jpx
pixi install
pixi shell
just get-results
# Open Claude Code and ask: "Which targets have the most consistent morphological profiles?"
```

## Data

All data is on a public S3 bucket (anonymous access, no AWS credentials needed).
`just get-inputs` and `just get-results` handle the download.

- `data/external/` - annotation sources (ChEMBL, Drug Repurposing Hub, Chemical Probes, ToxCast, MOTIVE)
- `data/raw/profiles/` - JUMP compound morphological profiles (parquet)
- `data/interim/` - augmented metadata database (DuckDB)
- `data/processed/` - copairs results database (DuckDB), analysis outputs

## Citation

See [CITATION.cff](CITATION.cff) for citation metadata (author list to be finalized before publication).

## License

BSD 3-Clause - see [LICENSE](LICENSE).
