# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "python-dotenv==1.2.2",
#     "loguru==0.7.3",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from loguru import logger

    load_dotenv()

    if os.environ.get("JUMP_PRODUCTION_ROOT"):
        PROJ_ROOT = Path(os.environ["JUMP_PRODUCTION_ROOT"]).resolve()
    else:
        PROJ_ROOT = Path(__file__).resolve().parents[1]
    logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

    DATA_DIR = PROJ_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    INTERIM_DATA_DIR = DATA_DIR / "interim"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"
    ACTIVITY_DIR = PROCESSED_DATA_DIR / "copairs/runs/activity"

    METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
    COPAIRS_RESULTS_DB = PROCESSED_DATA_DIR / "copairs_results.duckdb"
    ANNDATA_DIR = INTERIM_DATA_DIR / "anndata"

    DEFAULT_DPI = 150

    try:
        JUMP_CPUS: int = len(os.sched_getaffinity(0))
    except AttributeError:
        JUMP_CPUS: int = os.cpu_count() or 1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Configuration

    Foundation notebook defining all paths and constants for the jump_production catalog.
    Other notebooks import from here via `from nb00_ss_config import PROJ_ROOT, DATA_DIR, ...`

    **Paths:**
    - `PROJ_ROOT` - repository root
    - `DATA_DIR`, `RAW_DATA_DIR`, `INTERIM_DATA_DIR`, `PROCESSED_DATA_DIR`, `EXTERNAL_DATA_DIR`
    - `ACTIVITY_DIR` - copairs activity run outputs
    - `METADATA_DB` - augmented metadata DuckDB
    - `COPAIRS_RESULTS_DB` - copairs results DuckDB
    - `ANNDATA_DIR` - h5ad profile files

    **Constants:**
    - `DEFAULT_DPI` (150) - figure resolution
    - `JUMP_CPUS` - CPU count respecting cgroup/Slurm limits
    """)
    return


@app.cell
def _(mo):
    mo.md(f"""
    ## Current paths

    | Variable | Path |
    |----------|------|
    | PROJ_ROOT | `{PROJ_ROOT}` |
    | DATA_DIR | `{DATA_DIR}` |
    | METADATA_DB | `{METADATA_DB}` |
    | COPAIRS_RESULTS_DB | `{COPAIRS_RESULTS_DB}` |
    | ANNDATA_DIR | `{ANNDATA_DIR}` |
    | JUMP_CPUS | `{JUMP_CPUS}` |
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
