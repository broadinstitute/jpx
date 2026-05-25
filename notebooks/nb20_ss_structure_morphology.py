# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb",
#     "loguru",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "python-dotenv",
#     "scanpy",
#     "scipy",
# ]
# NOTE: Run with pixi run -e rapids marimo edit/run
# rdkit and rapids_singlecell come from the pixi rapids env (conda-only).
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from loguru import logger

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import (
        COPAIRS_RESULTS_DB,
        DATA_DIR,
        DEFAULT_DPI,
        INTERIM_DATA_DIR,
        METADATA_DB,
        PROCESSED_DATA_DIR,
    )
    from nb02_ss_queries import query_activity_results
    from nb04_ss_visualization import SIG_COLOR, plot_dotplot

    # GPU detection (cupy + rapids_singlecell)
    HAS_GPU = False
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        HAS_GPU = True
    except Exception:
        pass

    if HAS_GPU:
        try:
            import rapids_singlecell as rsc  # noqa: F401
        except ImportError:
            HAS_GPU = False

    logger.info(f"GPU acceleration: {'enabled' if HAS_GPU else 'disabled (cupy not available)'}")

    K_NEIGHBORS = 50


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Structure-Morphology Relationship

    Analyzes the relationship between chemical structure and morphological profiles
    for phenotypically active compounds:

    - Are structurally similar compounds also morphologically similar?
    - Which mechanisms of action show highest structure-morphology coherence?

    Supports two structural representations:

    - **Morgan fingerprints** (2048-bit): L2 norm + cosine approximates Tanimoto (rho=0.998)
    - **ChemBERTa embeddings** (384-dim): Learned chemical semantics from SMILES

    **Requires rapids pixi environment:** `pixi run -e rapids marimo edit notebooks/nb20_ss_structure_morphology.py`

    *Outputs:* `data/processed/structure-morphology/{dataset}/{preprocessing}/{structure_rep}/`
    """)
    return


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
    preprocessing_dropdown = mo.ui.dropdown(
        options=[
            "activity_no_target2",
        ],
        value="activity_no_target2",
        label="Preprocessing",
    )
    filter_dropdown = mo.ui.dropdown(
        options=[
            "all_sources",
        ],
        value="all_sources",
        label="Filter",
    )
    activity_params_dropdown = mo.ui.dropdown(
        options=[
            "default",
            "withinsource",
            "crosssource",
        ],
        value="default",
        label="Activity params",
    )
    structure_dropdown = mo.ui.dropdown(
        options=[
            "morgan",
            "chemberta",
        ],
        value="morgan",
        label="Structure representation",
    )
    mo.hstack(
        [dataset_dropdown, preprocessing_dropdown, filter_dropdown, activity_params_dropdown, structure_dropdown],
        justify="start",
    )
    return (
        activity_params_dropdown,
        dataset_dropdown,
        filter_dropdown,
        preprocessing_dropdown,
        structure_dropdown,
    )


@app.function
def load_fingerprints() -> tuple[np.ndarray, np.ndarray]:
    """Load compound fingerprints (Morgan FP, 2048-bit)."""
    _fp_data = np.load(DATA_DIR / "interim/compound_featurization/morgan_fp.npz", allow_pickle=True)
    _fp_array = _fp_data["fingerprints"]  # Shape: (N, 2048) uint8
    _fp_jcp = _fp_data["jcp2022"]
    logger.info(f"Morgan fingerprints: {_fp_array.shape[0]:,} compounds x {_fp_array.shape[1]} bits")
    return _fp_array, _fp_jcp


@app.function
def load_chemberta_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Load ChemBERTa-77M-MLM embeddings (384-dim)."""
    _chemberta_file = DATA_DIR / "interim/compound_featurization/chemberta_77m_mlm.npz"
    _emb_data = np.load(_chemberta_file, allow_pickle=True)
    _embeddings = _emb_data["embeddings"]  # Shape: (N, 384) float32
    _jcp = _emb_data["jcp2022"]
    logger.info(f"ChemBERTa embeddings: {_embeddings.shape[0]:,} compounds x {_embeddings.shape[1]} dims")
    return _embeddings, _jcp


@app.function
def get_valid_jcp_intersection() -> set[str]:
    """Get JCP2022 IDs valid in both Morgan FP and ChemBERTa embeddings.

    Ensures apples-to-apples comparison between representations by
    restricting to compounds where both have valid (non-zero) features.
    """
    # Load Morgan FP
    _morgan_data = np.load(DATA_DIR / "interim/compound_featurization/morgan_fp.npz", allow_pickle=True)
    _morgan_fp = _morgan_data["fingerprints"].astype(np.float32)
    _morgan_jcp = _morgan_data["jcp2022"]
    _morgan_norms = np.linalg.norm(_morgan_fp, axis=1)
    _morgan_valid = set(_morgan_jcp[_morgan_norms > 0])

    # Load ChemBERTa
    _chemberta_data = np.load(DATA_DIR / "interim/compound_featurization/chemberta_77m_mlm.npz", allow_pickle=True)
    _chemberta_emb = _chemberta_data["embeddings"]
    _chemberta_jcp = _chemberta_data["jcp2022"]
    _chemberta_norms = np.linalg.norm(_chemberta_emb, axis=1)
    _chemberta_valid = set(_chemberta_jcp[_chemberta_norms > 0])

    _intersection = _morgan_valid & _chemberta_valid
    logger.info(
        f"Valid in Morgan: {len(_morgan_valid):,}, ChemBERTa: {len(_chemberta_valid):,}, both: {len(_intersection):,}"
    )
    return _intersection


@app.function
def load_moa_annotations() -> pd.DataFrame:
    """Load MoA annotations from Drug Repurposing Hub."""
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _df = _con.execute("""
        SELECT
            Metadata_JCP2022 as JCP2022,
            Metadata_repurposing_moa as moa
        FROM compound_metadata
        WHERE Metadata_repurposing_moa IS NOT NULL
    """).df()
    _con.close()
    logger.info(f"MoA annotations: {len(_df):,} compounds")
    return _df


@app.function
def load_activity_data(
    dataset: str,
    preprocessing: str,
    filter_name: str,
    activity_params: str,
) -> pd.DataFrame:
    """Load activity status from copairs results, renaming columns for this analysis."""
    _df = query_activity_results(dataset, preprocessing, filter_name, activity_params)
    _df = _df.rename(
        columns={
            "Metadata_JCP2022": "JCP2022",
            "mean_average_precision": "mAP",
            "mean_normalized_average_precision": "nmAP",
            "below_corrected_p": "active",
        }
    )
    logger.info(f"Activity: {len(_df):,} compounds, {_df.active.sum():,} active ({_df.active.mean() * 100:.1f}%)")
    return _df[["JCP2022", "mAP", "nmAP", "active"]]


@app.function
def align_data(
    adata_profiles: sc.AnnData,
    fp_array: np.ndarray,
    fp_jcp: np.ndarray,
    activity_df: pd.DataFrame,
    valid_jcp: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align fingerprints and profiles by JCP2022 ID, filtering to active compounds.

    Args:
        adata_profiles: AnnData with profiles (obs must have JCP2022 column).
        fp_array: Structural feature array (Morgan FP or ChemBERTa embeddings).
        fp_jcp: JCP2022 IDs corresponding to fp_array rows.
        activity_df: Activity results with JCP2022 and active columns.
        valid_jcp: Optional set of JCP2022 IDs to restrict to (for fair
            cross-representation comparison).

    Returns:
        Tuple of (aligned_fp, aligned_morph, aligned_jcp_ids) for valid active compounds.
    """
    # Filter to active compounds first
    _active_jcp = set(activity_df[activity_df["active"]]["JCP2022"])
    logger.info(f"Active compounds: {len(_active_jcp):,}")

    _morph_jcp = adata_profiles.obs["JCP2022"].values
    _common_jcp = set(fp_jcp) & set(_morph_jcp) & _active_jcp

    # Apply additional valid_jcp filter if provided
    if valid_jcp is not None:
        _common_jcp = _common_jcp & valid_jcp
        logger.info(f"After valid_jcp filter: {len(_common_jcp):,}")

    _common_jcp = sorted(_common_jcp)
    logger.info(f"Common active compounds: {len(_common_jcp):,}")

    _fp_idx = {k: i for i, k in enumerate(fp_jcp)}
    _morph_idx = {k: i for i, k in enumerate(_morph_jcp)}

    _fp_order = [_fp_idx[k] for k in _common_jcp]
    _morph_order = [_morph_idx[k] for k in _common_jcp]

    _X_fp = fp_array[_fp_order].astype(np.float32)
    _X_morph = adata_profiles.X[_morph_order].astype(np.float32)
    _jcp_ids = np.array(_common_jcp)

    # Filter valid (no NaN, non-zero norm)
    _fp_norms = np.linalg.norm(_X_fp, axis=1)
    _morph_norms = np.linalg.norm(_X_morph, axis=1)
    _valid_mask = (_fp_norms > 0) & (_morph_norms > 0) & ~np.isnan(_morph_norms)
    logger.info(f"Valid active compounds: {_valid_mask.sum():,}")

    return _X_fp[_valid_mask], _X_morph[_valid_mask], _jcp_ids[_valid_mask]


@app.function
def compute_knn(
    X_fp: np.ndarray, X_morph: np.ndarray, jcp_ids: np.ndarray, k: int = K_NEIGHBORS
) -> tuple[sc.AnnData, sc.AnnData]:
    """Compute k-NN in structure and morphology spaces (GPU if available)."""
    # L2 normalize fingerprints (cosine on L2-normed approximates Tanimoto)
    _X_fp_norm = X_fp / np.linalg.norm(X_fp, axis=1, keepdims=True)
    _X_morph_norm = X_morph / np.linalg.norm(X_morph, axis=1, keepdims=True)

    # Select neighbors function based on GPU availability
    if HAS_GPU:
        import rapids_singlecell as rsc

        _neighbors_fn = rsc.pp.neighbors
        logger.info("Using GPU-accelerated k-NN (rapids_singlecell)")
    else:
        _neighbors_fn = sc.pp.neighbors
        logger.info("Using CPU k-NN (scanpy)")

    # Structure k-NN
    logger.info(f"Computing structure k-NN (k={k})...")
    _adata_struct = sc.AnnData(X=_X_fp_norm)
    _adata_struct.obs["JCP2022"] = jcp_ids
    _neighbors_fn(_adata_struct, n_neighbors=k, use_rep="X", metric="cosine")
    logger.info(f"  Edges: {_adata_struct.obsp['connectivities'].nnz:,}")

    # Morphology k-NN
    logger.info(f"Computing morphology k-NN (k={k})...")
    _adata_morph = sc.AnnData(X=_X_morph_norm)
    _adata_morph.obs["JCP2022"] = jcp_ids
    _neighbors_fn(_adata_morph, n_neighbors=k, use_rep="X", metric="cosine")
    logger.info(f"  Edges: {_adata_morph.obsp['connectivities'].nnz:,}")

    return _adata_struct, _adata_morph


@app.function
def compute_neighbor_overlap(adata_struct: sc.AnnData, adata_morph: sc.AnnData) -> np.ndarray:
    """Compute Jaccard overlap of k-NN between structure and morphology."""
    logger.info("Computing neighbor overlap...")
    _dist1 = adata_struct.obsp["distances"]
    _dist2 = adata_morph.obsp["distances"]

    _overlaps = []
    for _i in range(adata_struct.n_obs):
        _n1 = set(_dist1[_i].nonzero()[1])
        _n2 = set(_dist2[_i].nonzero()[1])
        if _n1 | _n2:
            _overlaps.append(len(_n1 & _n2) / len(_n1 | _n2))
        else:
            _overlaps.append(0.0)

    return np.array(_overlaps)


@app.function
def compute_moa_stats(
    df: pd.DataFrame, moa_df: pd.DataFrame, k: int, min_count: int = 10, top_n: int = 20
) -> pd.DataFrame:
    """Compute MoA statistics for overlap analysis.

    Returns DataFrame with columns: moa, mean_overlap, count, enrichment
    """
    _n_compounds = len(df)
    _expected_random = k / (2 * _n_compounds)

    _merged = df.merge(moa_df, on="JCP2022", how="inner")
    _moa_stats = _merged.groupby("moa").agg({"neighbor_overlap": ["mean", "count"]}).reset_index()
    _moa_stats.columns = ["moa", "mean_overlap", "count"]
    _moa_stats["enrichment"] = _moa_stats["mean_overlap"] / _expected_random
    _moa_stats = _moa_stats[_moa_stats["count"] >= min_count].sort_values("enrichment", ascending=True)
    return _moa_stats.tail(top_n)


@app.function
def plot_combined_figure(
    df: pd.DataFrame,
    moa_df: pd.DataFrame,
    k: int = K_NEIGHBORS,
    structure_rep: str = "morgan",
) -> plt.Figure:
    """Create combined publication figure for active compounds.

    Layout: A (overlap histogram) | B (MoA dotplot) side by side.
    Returns the figure object (caller saves if needed).
    """
    _rep_label = "Morgan FP" if structure_rep == "morgan" else "ChemBERTa"
    logger.info(f"Creating combined publication figure ({_rep_label})...")

    fig, _axes = plt.subplots(1, 2, figsize=(10, 4))

    # Compute summary stats for annotations
    _overlap = df["neighbor_overlap"].values
    _n_compounds = len(df)
    _mean_overlap = _overlap.mean()
    _pct_nonzero = (_overlap > 0).mean() * 100
    _expected_random = k / (2 * _n_compounds)
    _ratio_vs_random = _mean_overlap / _expected_random

    # Panel A: Overlap distribution histogram
    _ax_a = _axes[0]
    _ax_a.hist(_overlap, bins=50, color=SIG_COLOR, edgecolor="black", linewidth=0.5, alpha=0.8)
    _ax_a.set_xlabel(f"Jaccard Overlap (k={k})", fontsize=10)
    _ax_a.set_ylabel("Count", fontsize=10)
    _ax_a.set_title(f"A. k-NN Overlap Distribution ({_rep_label})", fontsize=11)
    _ax_a.axvline(_mean_overlap, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {_mean_overlap:.3f}")
    _ax_a.axvline(
        _expected_random, color="gray", linestyle=":", linewidth=1.5, label=f"Random = {_expected_random:.4f}"
    )
    # Stats text box
    _stats_text = f"n = {_n_compounds:,}\nNon-zero: {_pct_nonzero:.1f}%\n{_ratio_vs_random:.1f}x random"
    _ax_a.text(
        0.95,
        0.95,
        _stats_text,
        transform=_ax_a.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    _ax_a.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 0.78))

    # Panel B: MoA dotplot (enrichment vs random)
    _ax_b = _axes[1]
    _moa_stats = compute_moa_stats(df, moa_df, k=k)
    plot_dotplot(
        _moa_stats,
        value_col="enrichment",
        label_col="moa",
        count_col="count",
        top_n=20,
        title=f"B. Structure-Morphology Coherence ({_rep_label})",
        xlabel="Enrichment (x random)",
        size_scale=8.0,
        size_range=(40, 200),
        ax=_ax_b,
    )

    plt.tight_layout()
    return fig


@app.function
def generate_summary(
    df: pd.DataFrame,
    n_compounds: int,
    k: int,
    structure_rep: str = "morgan",
) -> dict:
    """Generate summary statistics for structure-morphology analysis (active compounds only)."""
    _overlap = df["neighbor_overlap"].values

    _mean_overlap = _overlap.mean()
    _median_overlap = float(np.median(_overlap))
    _expected_random = k / (2 * n_compounds)
    _ratio_vs_random = _mean_overlap / _expected_random
    _pct_zero = (_overlap == 0).mean() * 100
    _pct_nonzero = 100 - _pct_zero

    _summary = {
        "structure_rep": structure_rep,
        "n_compounds": n_compounds,
        "k_neighbors": k,
        "mean_jaccard": float(_mean_overlap),
        "median_jaccard": _median_overlap,
        "expected_random": float(_expected_random),
        "ratio_vs_random": float(_ratio_vs_random),
        "pct_zero_overlap": float(_pct_zero),
        "pct_nonzero_overlap": float(_pct_nonzero),
        "use_gpu": HAS_GPU,
    }

    logger.info(f"=== RESULTS (n={n_compounds:,} active compounds, k={k}) ===")
    logger.info(f"Mean Jaccard:     {_mean_overlap:.4f}")
    logger.info(f"Median Jaccard:   {_median_overlap:.4f}")
    logger.info(f"Expected random:  {_expected_random:.6f}")
    logger.info(f"Ratio vs random:  {_ratio_vs_random:.1f}x")
    logger.info(f"Non-zero overlap: {_pct_nonzero:.1f}%")

    return _summary


@app.cell
def _(
    activity_params_dropdown,
    dataset_dropdown,
    filter_dropdown,
    mo,
    preprocessing_dropdown,
    structure_dropdown,
):
    mo.stop(
        not COPAIRS_RESULTS_DB.exists(),
        mo.md(f"**Copairs results database not found:** `{COPAIRS_RESULTS_DB}`"),
    )
    mo.stop(
        not METADATA_DB.exists(),
        mo.md(f"**Metadata database not found:** `{METADATA_DB}`"),
    )

    _dataset = dataset_dropdown.value
    _preprocessing = preprocessing_dropdown.value
    _filter_name = filter_dropdown.value
    _activity_params = activity_params_dropdown.value
    _structure_rep = structure_dropdown.value

    # Load structural features
    if _structure_rep == "morgan":
        fp_array, fp_jcp = load_fingerprints()
    else:
        fp_array, fp_jcp = load_chemberta_embeddings()

    # Load profiles
    _anndata_path = INTERIM_DATA_DIR / "anndata" / f"{_dataset}_perturbation.h5ad"
    mo.stop(
        not _anndata_path.exists(),
        mo.md(f"**Profile not found:** `{_anndata_path}`"),
    )
    adata_profiles = sc.read_h5ad(_anndata_path)
    logger.info(f"Profiles: {adata_profiles.shape[0]:,} compounds x {adata_profiles.shape[1]} features")

    # Load activity data
    activity_df = load_activity_data(_dataset, _preprocessing, _filter_name, _activity_params)

    # Load MoA annotations
    moa_df = load_moa_annotations()

    # Get valid intersection for fair cross-representation comparison
    valid_jcp = get_valid_jcp_intersection()

    mo.md(f"""
    ## Data loaded

    **Dataset:** {_dataset} | **Preprocessing:** {_preprocessing}
    | **Filter:** {_filter_name} | **Activity params:** {_activity_params}
    | **Structure:** {_structure_rep}

    - Structural features: {fp_array.shape[0]:,} compounds x {fp_array.shape[1]} dims
    - Profiles: {adata_profiles.shape[0]:,} compounds x {adata_profiles.shape[1]} features
    - Activity: {len(activity_df):,} compounds ({activity_df.active.sum():,} active)
    - MoA annotations: {len(moa_df):,} compounds
    - Valid in both representations: {len(valid_jcp):,}
    """)
    return activity_df, adata_profiles, fp_array, fp_jcp, moa_df, valid_jcp


@app.cell
def _(activity_df, adata_profiles, fp_array, fp_jcp, mo, valid_jcp):
    # Align data (filters to active compounds with valid representations)
    X_fp, X_morph, jcp_ids = align_data(adata_profiles, fp_array, fp_jcp, activity_df, valid_jcp=valid_jcp)
    _n_compounds = len(jcp_ids)

    mo.md(f"""
    ## Alignment

    **Valid active compounds for analysis:** {_n_compounds:,}
    """)
    return X_fp, X_morph, jcp_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## k-NN Computation

    Computing k-nearest neighbors in both structure space and morphology space,
    then measuring the Jaccard overlap of neighbor sets.
    """)
    return


@app.cell
def _(X_fp, X_morph, activity_df, jcp_ids, mo, structure_dropdown):
    # Compute k-NN in both spaces
    _adata_struct, _adata_morph = compute_knn(X_fp, X_morph, jcp_ids, k=K_NEIGHBORS)

    # Compute neighbor overlap
    _overlap = compute_neighbor_overlap(_adata_struct, _adata_morph)

    # Build results DataFrame with activity scores
    results_df = pd.DataFrame({"JCP2022": jcp_ids, "neighbor_overlap": _overlap})
    results_df = results_df.merge(activity_df[["JCP2022", "mAP", "nmAP"]], on="JCP2022", how="left")

    _mean_ov = _overlap.mean()
    _pct_nonzero = (_overlap > 0).mean() * 100
    _expected = K_NEIGHBORS / (2 * len(jcp_ids))

    mo.md(f"""
    ## Results

    **Structure representation:** {structure_dropdown.value}
    | **k:** {K_NEIGHBORS} | **n compounds:** {len(jcp_ids):,}

    - Mean Jaccard overlap: {_mean_ov:.4f}
    - Non-zero overlap: {_pct_nonzero:.1f}%
    - Expected random: {_expected:.6f}
    - Ratio vs random: {_mean_ov / _expected:.1f}x
    """)
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Publication Figure

    **Panel A:** Distribution of Jaccard overlap between structure and morphology k-NN.

    **Panel B:** MoA-level enrichment (mean overlap / expected random) for mechanisms
    with at least 10 compounds.
    """)
    return


@app.cell
def _(moa_df, results_df, structure_dropdown):
    fig_combined = plot_combined_figure(
        results_df,
        moa_df,
        k=K_NEIGHBORS,
        structure_rep=structure_dropdown.value,
    )
    fig_combined
    return (fig_combined,)


@app.cell
def _(
    dataset_dropdown,
    fig_combined,
    mo,
    preprocessing_dropdown,
    results_df,
    structure_dropdown,
):
    _output_dir = (
        PROCESSED_DATA_DIR
        / "structure-morphology"
        / dataset_dropdown.value
        / preprocessing_dropdown.value
        / structure_dropdown.value
    )
    _output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined figure
    fig_combined.savefig(_output_dir / "structure_morphology_combined.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_combined.savefig(_output_dir / "structure_morphology_combined.pdf", bbox_inches="tight")

    # Save results parquet
    results_df.to_parquet(_output_dir / "neighbor_overlap.parquet", index=False)

    # Generate and save summary JSON
    _summary = generate_summary(
        results_df,
        n_compounds=len(results_df),
        k=K_NEIGHBORS,
        structure_rep=structure_dropdown.value,
    )
    _summary_path = _output_dir / "structure_morphology_summary.json"
    with open(_summary_path, "w") as _f:
        json.dump(_summary, _f, indent=2)
    logger.info(f"Saved summary to {_summary_path}")

    mo.md(f"""
    **Saved outputs to** `{_output_dir}`

    - `structure_morphology_combined.png` / `.pdf` - combined publication figure
    - `neighbor_overlap.parquet` - per-compound overlap results
    - `structure_morphology_summary.json` - summary statistics
    """)
    return


@app.function
def run_structure_morphology(
    dataset: str = "compound_no_source7",
    preprocessing: str = "activity_no_target2",
    structure_rep: str = "morgan",
    filter_name: str = "all_sources",
    activity_params: str = "default",
    output_dir=None,
) -> str:
    """Run the full structure-morphology k-NN overlap analysis.

    Composes load_fingerprints/load_chemberta_embeddings, load_activity_data,
    align_data, compute_knn, compute_neighbor_overlap, plot_combined_figure,
    and generate_summary. Saves combined plot, neighbor_overlap.parquet,
    summary.json, and .complete marker.

    Called from workflow.py via run_task.py.

    Args:
        dataset: Dataset name (e.g., "compound_no_source7")
        preprocessing: Preprocessing name (e.g., "activity_no_target2")
        structure_rep: "morgan" or "chemberta"
        filter_name: Filter name (e.g., "all_sources")
        activity_params: Activity params (e.g., "default")
        output_dir: Output directory (default: data/processed/structure-morphology/...)

    Returns:
        Path to .complete marker file.
    """
    from nb00_ss_config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "structure-morphology" / dataset / preprocessing / structure_rep
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load structural features
    if structure_rep == "morgan":
        _fp_array, _fp_jcp = load_fingerprints()
    else:
        _fp_array, _fp_jcp = load_chemberta_embeddings()

    # Load profiles
    _anndata_path = INTERIM_DATA_DIR / "anndata" / f"{dataset}_perturbation.h5ad"
    if not _anndata_path.exists():
        msg = f"Profile not found: {_anndata_path}"
        raise FileNotFoundError(msg)
    _adata_profiles = sc.read_h5ad(_anndata_path)
    logger.info(f"Profiles: {_adata_profiles.shape[0]:,} compounds x {_adata_profiles.shape[1]} features")

    # Load activity data and MoA annotations
    _activity_df = load_activity_data(dataset, preprocessing, filter_name, activity_params)
    _moa_df = load_moa_annotations()

    # Get valid intersection for fair cross-representation comparison
    _valid_jcp = get_valid_jcp_intersection()

    # Align data (filters to active compounds with valid representations)
    _X_fp, _X_morph, _jcp_ids = align_data(_adata_profiles, _fp_array, _fp_jcp, _activity_df, valid_jcp=_valid_jcp)

    # Compute k-NN in both spaces
    _adata_struct, _adata_morph = compute_knn(_X_fp, _X_morph, _jcp_ids, k=K_NEIGHBORS)

    # Compute neighbor overlap
    _overlap = compute_neighbor_overlap(_adata_struct, _adata_morph)

    # Build results DataFrame
    _results_df = pd.DataFrame({"JCP2022": _jcp_ids, "neighbor_overlap": _overlap})
    _results_df = _results_df.merge(_activity_df[["JCP2022", "mAP", "nmAP"]], on="JCP2022", how="left")

    # Save combined figure
    _fig = plot_combined_figure(_results_df, _moa_df, k=K_NEIGHBORS, structure_rep=structure_rep)
    _fig.savefig(output_dir / "structure_morphology_combined.png", dpi=150, bbox_inches="tight", facecolor="white")
    _fig.savefig(output_dir / "structure_morphology_combined.pdf", bbox_inches="tight")
    plt.close(_fig)

    # Save results parquet
    _results_df.to_parquet(output_dir / "neighbor_overlap.parquet", index=False)

    # Generate and save summary
    _summary = generate_summary(_results_df, n_compounds=len(_results_df), k=K_NEIGHBORS, structure_rep=structure_rep)
    Path(output_dir, "summary.json").write_text(json.dumps(_summary, indent=2))

    # Write .complete marker
    _marker = output_dir / ".complete"
    _marker.touch()
    logger.success(f"Saved structure-morphology outputs to {output_dir}")
    return str(_marker)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
