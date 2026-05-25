# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anndata==0.12.16",
#     "duckdb==1.5.3",
#     "pandas==2.3.3",
#     "python-dotenv",
#     "loguru==0.7.3",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    import anndata as ad
    import duckdb
    from loguru import logger

    from nb00_ss_config import ANNDATA_DIR, COPAIRS_RESULTS_DB, METADATA_DB


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Profiles

    Foundation notebook for loading JUMP profiles as AnnData and joining metadata/activity.
    Other notebooks import via `from nb03_ss_profiles import load_profiles, join_metadata, ...`

    **Exported functions:**
    - `load_profiles(name, level)` - load h5ad profile file
    - `join_metadata(adata, modality, level)` - join compound/gene annotations from DuckDB
    - `load_umap(dataset, level, metric, filter_name)` - load UMAP h5ad + metadata
    - `join_activity(adata, dataset, preprocessing, filter_name)` - join copairs activity results

    **Design:** `load_profiles()` only loads h5ad. Metadata and activity are joined separately.
    This decouples UMAP computation (no metadata needed) from visualization (metadata needed).
    """)
    return


@app.function
def load_profiles(name: str, level: str = "well") -> ad.AnnData:
    """Load a profile dataset as AnnData (without metadata)."""
    suffix = "_perturbation" if level == "perturbation" else ""
    h5ad_path = ANNDATA_DIR / f"{name}{suffix}.h5ad"

    if not h5ad_path.exists():
        raise FileNotFoundError(f"Profile not found: {h5ad_path}. Run: just redun main")

    logger.info(f"Loading {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    logger.info(f"Loaded {adata.n_obs:,} obs x {adata.n_vars} vars")
    return adata


@app.function
def join_metadata(adata: ad.AnnData, modality: str = "compound", level: str = "well") -> ad.AnnData:
    """Join metadata from DuckDB to AnnData obs.

    Queries compound_metadata or gene_metadata view from jump_metadata_augmented.duckdb.
    Strips Metadata_ prefix from column names.
    """
    if not METADATA_DB.exists():
        logger.warning(f"Metadata DB not found: {METADATA_DB}, skipping")
        return adata

    con = duckdb.connect(str(METADATA_DB), read_only=True)

    jcp_ids = adata.obs["JCP2022"].unique().tolist()
    logger.info(f"Joining {modality} metadata for {len(jcp_ids):,} unique perturbations")

    view_name = "compound_metadata" if modality == "compound" else "gene_metadata"
    metadata_df = con.execute(
        f"SELECT * FROM {view_name} WHERE Metadata_JCP2022 = ANY(?)",
        [jcp_ids],
    ).df()

    metadata_df.columns = [c.replace("Metadata_", "") if c.startswith("Metadata_") else c for c in metadata_df.columns]

    obs = adata.obs.merge(metadata_df, on="JCP2022", how="left")
    obs.index = adata.obs.index
    adata.obs = obs

    n_joined = metadata_df["JCP2022"].nunique()
    logger.info(f"Joined metadata for {n_joined:,} of {len(jcp_ids):,} perturbations")

    if level == "well" and all(c in adata.obs.columns for c in ["Source", "Plate", "Well"]):
        cell_counts_df = con.execute(
            "SELECT Metadata_Source, Metadata_Plate, Metadata_Well, Metadata_Count_Cells FROM cell_counts"
        ).df()
        cell_counts_df.columns = ["Source", "Plate", "Well", "Count_Cells"]

        obs = adata.obs.merge(cell_counts_df, on=["Source", "Plate", "Well"], how="left")
        obs.index = adata.obs.index
        adata.obs = obs

        n_with_counts = obs["Count_Cells"].notna().sum()
        logger.info(f"Cell counts joined: {n_with_counts:,} wells")

    con.close()
    return adata


@app.function
def load_umap(
    dataset: str,
    level: str = "perturbation",
    metric: str = "cosine",
    filter_name: str = "all",
) -> ad.AnnData:
    """Load UMAP h5ad with metadata joined."""
    prefix = f"{dataset}_perturbation" if level == "perturbation" else dataset
    h5ad_path = ANNDATA_DIR / f"{prefix}_{metric}_{filter_name}_umap.h5ad"

    if not h5ad_path.exists():
        raise FileNotFoundError(f"UMAP h5ad not found: {h5ad_path}")

    logger.info(f"Loading UMAP from {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    logger.info(f"Loaded {adata.n_obs:,} obs")

    adata = join_metadata(adata, modality="compound", level=level)
    return adata


@app.function
def join_activity(
    adata: ad.AnnData,
    dataset: str,
    preprocessing: str = "activity_no_target2",
    filter_name: str = "all_sources",
) -> ad.AnnData:
    """Join phenotypic activity results to AnnData obs."""
    if not COPAIRS_RESULTS_DB.exists():
        logger.warning(f"Copairs results DB not found: {COPAIRS_RESULTS_DB}, skipping")
        return adata

    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)

    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if "activity_results" not in tables:
        logger.warning("activity_results table not found, skipping")
        con.close()
        return adata

    jcp_ids = adata.obs["JCP2022"].unique().tolist()
    logger.info(
        f"Querying activity for {len(jcp_ids):,} perturbations (preprocessing={preprocessing}, filter={filter_name})"
    )

    activity_df = con.execute(
        """
        SELECT Metadata_JCP2022, mean_average_precision, mean_normalized_average_precision,
               p_value, corrected_p_value, below_corrected_p
        FROM activity_results
        WHERE _dataset = ? AND _preprocessing = ? AND _filter = ?
          AND Metadata_JCP2022 = ANY(?)
        """,
        [dataset, preprocessing, filter_name, jcp_ids],
    ).df()
    con.close()

    logger.info(f"Retrieved activity for {len(activity_df):,} perturbations")

    if activity_df.empty:
        logger.warning("No matching activity results found")
        return adata

    obs = adata.obs.merge(
        activity_df,
        left_on="JCP2022",
        right_on="Metadata_JCP2022",
        how="left",
    ).drop(columns=["Metadata_JCP2022"], errors="ignore")

    if "below_corrected_p" in obs.columns:
        obs["below_corrected_p"] = obs["below_corrected_p"].map({True: "True", False: "False"}).fillna("NA")

    obs.index = adata.obs.index
    adata.obs = obs

    n_significant = (obs["below_corrected_p"] == "True").sum()
    n_with_activity = obs["p_value"].notna().sum()
    logger.info(f"Activity joined: {n_with_activity:,} with p-values, {n_significant:,} significant")

    return adata


@app.cell
def _(mo):
    dataset_dropdown = mo.ui.dropdown(
        options=[
            "compound_no_source7",
            "compound_DL_CPCNN_no_source7",
        ],
        value="compound_no_source7",
        label="Dataset",
    )
    level_dropdown = mo.ui.dropdown(
        options=["well", "perturbation"],
        value="perturbation",
        label="Level",
    )
    mo.hstack([dataset_dropdown, level_dropdown])
    return (dataset_dropdown, level_dropdown)


@app.cell
def _(dataset_dropdown, level_dropdown, mo):
    adata = load_profiles(dataset_dropdown.value, level=level_dropdown.value)
    mo.md(f"""
    ## Loaded profiles

    **Dataset:** {dataset_dropdown.value} ({level_dropdown.value})
    **Shape:** {adata.n_obs:,} obs x {adata.n_vars} vars
    **obs columns:** {", ".join(adata.obs.columns[:10])}...
    """)
    return (adata,)


@app.cell
def _(adata, mo):
    mo.ui.dataframe(adata.obs.head(10))
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
