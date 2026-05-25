# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anndata==0.12.16",
#     "duckdb==1.5.3",
#     "loguru==0.7.3",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==2.3.3",
#     "python-dotenv",
#     "scanpy==1.12.1",
#     "scipy==1.17.1",
#     "scikit-learn==1.8.0",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.23.6"
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
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import silhouette_score

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import (
        ANNDATA_DIR,
        DEFAULT_DPI,
        METADATA_DB,
        PROCESSED_DATA_DIR,
    )
    from nb03_ss_profiles import load_profiles

    # Paths for precomputed UMAPs
    MORPH_UMAP_PATH = ANNDATA_DIR / "compound_no_source7_perturbation_cosine_all_umap.h5ad"
    CHEM_UMAP_PATH = PROCESSED_DATA_DIR / "chemical-space" / "structure_umap.h5ad"

    # 8 top-level MitoTox functional mechanisms
    MECHANISMS = [
        "Transmembrane potential",
        "Function of mitochondria",
        "Organization of mitochondria",
        "Movement of mitochondria",
        "Oxidative stress",
        "Mitochondrial DNA",
        "Cell death",
        "Signaling",
    ]

    # Short names for plotting
    MECHANISM_SHORT = {
        "Transmembrane potential": "F01: Membrane potential",
        "Function of mitochondria": "F02: Mito function",
        "Organization of mitochondria": "F03: Mito organization",
        "Movement of mitochondria": "F04: Mito movement",
        "Oxidative stress": "F05: Oxidative stress",
        "Mitochondrial DNA": "F06: mtDNA",
        "Cell death": "F07: Cell death",
        "Signaling": "F08: Signaling",
    }


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MitoTox Morphological Mechanism Analysis

    Clusters mitotoxicants by morphology and tests whether functional
    submechanisms (F01-F08) separate in phenotypic space.

    **Research question:** Can morphology distinguish mitotoxicity
    submechanisms, not just toxic vs non-toxic?

    **Success criteria:**

    - Positive separation scores (between-mechanism > within-mechanism distance)
    - Significant p-values from permutation tests
    - Visual clustering in UMAP by mechanism

    **Workflow:**

    1. Load MitoTox annotations matched to JUMP compounds
    2. Overlay on morphology and chemistry UMAPs (full background + MitoTox highlights)
    3. Hierarchical clustering and mechanism separation statistics
    4. Silhouette score with permutation test for significance
    """)
    return


# ============================================================================
# Data Loading
# ============================================================================


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
    n_permutations_slider = mo.ui.slider(
        start=100,
        stop=2000,
        step=100,
        value=1000,
        label="Permutations",
    )
    mo.hstack([dataset_dropdown, n_permutations_slider])
    return (dataset_dropdown, n_permutations_slider)


@app.cell
def _(mo):
    _db_exists = METADATA_DB.exists()
    mo.stop(
        not _db_exists,
        mo.md(f"**Error:** Metadata database not found at `{METADATA_DB}`. Run `just run` first."),
    )

    con = duckdb.connect(str(METADATA_DB), read_only=True)
    matched_df = con.execute("""
        SELECT
            Metadata_JCP2022,
            Metadata_mitotox_label AS mitotox_label,
            Metadata_mitotox_toxic,
            Metadata_mitotox_mechanisms AS functional_mechanism
        FROM mitotox_annotations
    """).df()
    con.close()
    logger.info(f"Loaded {len(matched_df)} MitoTox compounds from database")

    # Parse mechanisms into binary columns
    for _mech in MECHANISMS:
        matched_df[f"has_{_mech}"] = matched_df["functional_mechanism"].fillna("").str.contains(_mech, regex=False)

    # Add toxic flag
    matched_df["is_toxic"] = matched_df["Metadata_mitotox_toxic"] == 1

    _toxic_count = matched_df["is_toxic"].sum()
    _lines = [
        "## MitoTox Data Loaded",
        "",
        f"**Total compounds:** {len(matched_df)}",
        f"**Toxic:** {_toxic_count} | **Non-toxic:** {len(matched_df) - _toxic_count}",
        "",
        "| Mechanism | Count |",
        "|-----------|-------|",
    ]
    for _mech in MECHANISMS:
        _count = matched_df[f"has_{_mech}"].sum()
        _lines.append(f"| {MECHANISM_SHORT[_mech]} | {_count} |")

    mo.md("\n".join(_lines))
    return (matched_df,)


# ============================================================================
# Load UMAPs
# ============================================================================


@app.cell
def _(matched_df, mo):
    mo.stop(
        not MORPH_UMAP_PATH.exists(),
        mo.md(f"**Error:** Morphology UMAP not found at `{MORPH_UMAP_PATH}`."),
    )
    mo.stop(
        not CHEM_UMAP_PATH.exists(),
        mo.md(f"**Error:** Chemistry UMAP not found at `{CHEM_UMAP_PATH}`."),
    )

    matched_jcp = set(matched_df["Metadata_JCP2022"])

    logger.info("Loading precomputed morphology UMAP...")
    adata_morph_full = sc.read_h5ad(MORPH_UMAP_PATH)
    logger.info(f"Full morphology UMAP: {adata_morph_full.n_obs} compounds")

    logger.info("Loading precomputed chemistry UMAP...")
    adata_chem_full = sc.read_h5ad(CHEM_UMAP_PATH)
    if "Metadata_JCP2022" in adata_chem_full.obs.columns:
        adata_chem_full.obs["JCP2022"] = adata_chem_full.obs["Metadata_JCP2022"]
    logger.info(f"Full chemistry UMAP: {adata_chem_full.n_obs} compounds")

    # Find intersection of compounds in both UMAPs
    _morph_jcp = set(adata_morph_full.obs["JCP2022"])
    _chem_jcp = set(adata_chem_full.obs["JCP2022"])
    common_jcp = _morph_jcp & _chem_jcp
    logger.info(f"Compounds in both UMAPs: {len(common_jcp)}")

    # Filter both full UMAPs to common compounds
    adata_morph_full = adata_morph_full[adata_morph_full.obs["JCP2022"].isin(common_jcp)].copy()
    adata_chem_full = adata_chem_full[adata_chem_full.obs["JCP2022"].isin(common_jcp)].copy()

    # Filter MitoTox to compounds in both UMAPs
    matched_jcp_common = matched_jcp & common_jcp

    # MitoTox subsets
    _morph_mask = adata_morph_full.obs["JCP2022"].isin(matched_jcp_common)
    adata_morph = adata_morph_full[_morph_mask].copy()

    _chem_mask = adata_chem_full.obs["JCP2022"].isin(matched_jcp_common)
    adata_chem = adata_chem_full[_chem_mask].copy()

    # Join mechanism annotations
    _mech_cols = [f"has_{m}" for m in MECHANISMS] + ["is_toxic", "mitotox_label", "functional_mechanism"]
    _matched_dedup = matched_df.drop_duplicates(subset=["Metadata_JCP2022"])[["Metadata_JCP2022"] + _mech_cols]
    _matched_dedup = _matched_dedup.rename(columns={"Metadata_JCP2022": "JCP2022"})

    # Morphology obs
    _morph_obs = adata_morph.obs.copy().reset_index()
    _morph_obs = _morph_obs.merge(_matched_dedup, on="JCP2022", how="left")
    _morph_obs = _morph_obs.set_index(adata_morph.obs.index.name or "index")
    adata_morph.obs = _morph_obs

    # Chemistry obs
    _chem_obs = adata_chem.obs.copy().reset_index()
    _chem_obs = _chem_obs.merge(_matched_dedup, on="JCP2022", how="left")
    _chem_obs = _chem_obs.set_index(adata_chem.obs.index.name or "index")
    adata_chem.obs = _chem_obs

    mo.md(f"""
    ## UMAPs Loaded

    - Morphology: **{adata_morph_full.n_obs:,}** total, **{adata_morph.n_obs}** MitoTox
    - Chemistry: **{adata_chem_full.n_obs:,}** total, **{adata_chem.n_obs}** MitoTox
    - Common compounds across both UMAPs: **{len(common_jcp):,}**
    """)
    return (adata_chem, adata_chem_full, adata_morph, adata_morph_full, matched_jcp_common)


# ============================================================================
# UMAP Visualization Helpers
# ============================================================================


@app.function
def compute_umap_limits(coords: np.ndarray, iqr_multiplier: float = 1.5) -> tuple[float, float, float, float]:
    """Compute axis limits using IQR-based outlier clipping."""
    x, y = coords[:, 0], coords[:, 1]

    q1_x, q3_x = np.percentile(x, [25, 75])
    iqr_x = q3_x - q1_x
    xlim = (q1_x - iqr_multiplier * iqr_x, q3_x + iqr_multiplier * iqr_x)

    q1_y, q3_y = np.percentile(y, [25, 75])
    iqr_y = q3_y - q1_y
    ylim = (q1_y - iqr_multiplier * iqr_y, q3_y + iqr_multiplier * iqr_y)

    return xlim[0], xlim[1], ylim[0], ylim[1]


@app.function
def plot_umap_full_with_mitotox(
    full_coords: np.ndarray,
    mitotox_coords: np.ndarray,
    mitotox_obs: pd.DataFrame,
    title_prefix: str,
) -> plt.Figure:
    """Plot full UMAP with all compounds as background, MitoTox colored by mechanism.

    Creates a 3x3 grid: first panel is Non-toxic, panels 2-9 are the 8 mechanisms.
    """
    xlim_min, xlim_max, ylim_min, ylim_max = compute_umap_limits(full_coords)

    categories = ["Non-toxic"] + MECHANISMS
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"{title_prefix} - Full UMAP with MitoTox Compounds", fontsize=14, y=1.02)

    for _ax, _cat, _color in zip(axes.flat, categories, colors):
        _ax.scatter(
            full_coords[:, 0],
            full_coords[:, 1],
            c="lightgray",
            s=1,
            alpha=0.3,
            rasterized=True,
        )

        if _cat == "Non-toxic":
            _mask = ~mitotox_obs["is_toxic"].values
            _title = f"Non-toxic (n={_mask.sum()})"
        else:
            _mask = mitotox_obs[f"has_{_cat}"].values
            _title = f"{MECHANISM_SHORT[_cat]} (n={_mask.sum()})"

        if _mask.sum() > 0:
            _ax.scatter(
                mitotox_coords[_mask, 0],
                mitotox_coords[_mask, 1],
                c=[_color],
                s=20,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.3,
                rasterized=True,
            )

        _ax.set_title(_title, fontsize=10)
        _ax.set_xlim(xlim_min, xlim_max)
        _ax.set_ylim(ylim_min, ylim_max)
        _ax.set_xticks([])
        _ax.set_yticks([])

    plt.tight_layout()
    return fig


@app.function
def plot_umap_by_mechanisms(
    coords: np.ndarray,
    obs_df: pd.DataFrame,
    title_prefix: str,
) -> plt.Figure:
    """Plot UMAP colored by each mechanism in a 2x4 grid."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{title_prefix} - MitoTox Compounds by Mechanism", fontsize=14, y=1.02)

    for _ax, _mech in zip(axes.flat, MECHANISMS):
        _has_mech = obs_df[f"has_{_mech}"].values

        _ax.scatter(
            coords[~_has_mech, 0],
            coords[~_has_mech, 1],
            c="lightgray",
            s=10,
            alpha=0.5,
            label="Other",
            rasterized=True,
        )

        _ax.scatter(
            coords[_has_mech, 0],
            coords[_has_mech, 1],
            c="tab:red",
            s=15,
            alpha=0.8,
            label=f"n={_has_mech.sum()}",
            rasterized=True,
        )

        _ax.set_title(MECHANISM_SHORT[_mech], fontsize=10)
        _ax.legend(loc="upper right", fontsize=8)
        _ax.set_xticks([])
        _ax.set_yticks([])

    plt.tight_layout()
    return fig


@app.function
def plot_umap_toxic_vs_nontoxic(
    coords: np.ndarray,
    obs_df: pd.DataFrame,
    title_prefix: str,
) -> plt.Figure:
    """Plot UMAP colored by toxic vs non-toxic."""
    fig, ax = plt.subplots(figsize=(8, 8))

    _is_toxic = obs_df["is_toxic"].values

    ax.scatter(
        coords[~_is_toxic, 0],
        coords[~_is_toxic, 1],
        c="tab:blue",
        s=15,
        alpha=0.7,
        label=f"Non-toxic (n={(~_is_toxic).sum()})",
        rasterized=True,
    )

    ax.scatter(
        coords[_is_toxic, 0],
        coords[_is_toxic, 1],
        c="tab:red",
        s=15,
        alpha=0.7,
        label=f"Toxic (n={_is_toxic.sum()})",
        rasterized=True,
    )

    ax.set_title(f"{title_prefix} - Toxic vs Non-toxic")
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    return fig


# ============================================================================
# UMAP Plots: Morphology
# ============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Morphology UMAPs""")
    return


@app.cell
def _(adata_morph, adata_morph_full):
    fig_morph_full = plot_umap_full_with_mitotox(
        adata_morph_full.obsm["X_umap"],
        adata_morph.obsm["X_umap"],
        adata_morph.obs,
        "Morphology",
    )
    fig_morph_full
    return (fig_morph_full,)


@app.cell
def _(adata_morph):
    fig_morph_mechanisms = plot_umap_by_mechanisms(
        adata_morph.obsm["X_umap"],
        adata_morph.obs,
        "Morphology UMAP",
    )
    fig_morph_mechanisms
    return (fig_morph_mechanisms,)


@app.cell
def _(adata_morph):
    fig_morph_toxic = plot_umap_toxic_vs_nontoxic(
        adata_morph.obsm["X_umap"],
        adata_morph.obs,
        "Morphology UMAP",
    )
    fig_morph_toxic
    return (fig_morph_toxic,)


# ============================================================================
# UMAP Plots: Chemistry
# ============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Chemistry UMAPs""")
    return


@app.cell
def _(adata_chem, adata_chem_full):
    fig_chem_full = plot_umap_full_with_mitotox(
        adata_chem_full.obsm["X_umap"],
        adata_chem.obsm["X_umap"],
        adata_chem.obs,
        "Chemistry",
    )
    fig_chem_full
    return (fig_chem_full,)


@app.cell
def _(adata_chem):
    fig_chem_mechanisms = plot_umap_by_mechanisms(
        adata_chem.obsm["X_umap"],
        adata_chem.obs,
        "Chemistry UMAP",
    )
    fig_chem_mechanisms
    return (fig_chem_mechanisms,)


@app.cell
def _(adata_chem):
    fig_chem_toxic = plot_umap_toxic_vs_nontoxic(
        adata_chem.obsm["X_umap"],
        adata_chem.obs,
        "Chemistry UMAP",
    )
    fig_chem_toxic
    return (fig_chem_toxic,)


# ============================================================================
# Hierarchical Clustering & Statistics
# ============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Hierarchical Clustering & Mechanism Separation""")
    return


@app.cell
def _(dataset_dropdown, matched_df, mo):
    _dataset = dataset_dropdown.value
    matched_jcp_all = set(matched_df["Metadata_JCP2022"])

    logger.info("Loading full morphology profiles for clustering...")
    adata_profiles = load_profiles(_dataset, level="perturbation")
    _profile_mask = adata_profiles.obs["JCP2022"].isin(matched_jcp_all)
    adata_profiles = adata_profiles[_profile_mask].copy()
    logger.info(f"Full profiles: {adata_profiles.n_obs} compounds, {adata_profiles.n_vars} features")

    # Join mechanism annotations
    _mech_cols = [f"has_{m}" for m in MECHANISMS] + [
        "is_toxic",
        "mitotox_label",
        "functional_mechanism",
    ]
    _matched_dedup = matched_df.drop_duplicates(subset=["Metadata_JCP2022"])[["Metadata_JCP2022"] + _mech_cols]
    _matched_dedup = _matched_dedup.rename(columns={"Metadata_JCP2022": "JCP2022"})

    _prof_obs = adata_profiles.obs.copy().reset_index()
    _prof_obs = _prof_obs.merge(_matched_dedup, on="JCP2022", how="left")
    _prof_obs = _prof_obs.set_index(adata_profiles.obs.index.name or "index")
    adata_profiles.obs = _prof_obs

    # L2 normalize for cosine distance
    _X = adata_profiles.X
    _norms = np.linalg.norm(_X, axis=1, keepdims=True)
    X_norm = _X / np.maximum(_norms, 1e-10)

    # Compute pairwise cosine distances
    logger.info("Computing pairwise distances...")
    _distances = pdist(X_norm, metric="cosine")
    dist_matrix = squareform(_distances)

    mo.md(f"""
    Loaded **{adata_profiles.n_obs}** MitoTox compounds with **{adata_profiles.n_vars}** features
    from `{_dataset}`.
    """)
    return (adata_profiles, dist_matrix)


@app.cell
def _(adata_profiles, dist_matrix):
    # Create labels for dendrogram
    def _get_primary_mechanism(row):
        _mechs = [m for m in MECHANISMS if row.get(f"has_{m}", False)]
        if len(_mechs) == 0:
            return "None"
        elif len(_mechs) == 1:
            return MECHANISM_SHORT[_mechs[0]][:10]
        else:
            return "Multiple"

    _dendro_labels = [_get_primary_mechanism(row) for _, row in adata_profiles.obs.iterrows()]

    # Convert distance matrix to condensed form for linkage
    _condensed = squareform(dist_matrix, checks=False)
    _Z = linkage(_condensed, method="average")

    fig_dendrogram = plt.figure(figsize=(15, 8))
    _ax = fig_dendrogram.add_subplot(111)
    dendrogram(
        _Z,
        labels=_dendro_labels,
        leaf_rotation=90,
        leaf_font_size=6,
        ax=_ax,
    )
    _ax.set_title("Hierarchical Clustering of MitoTox Compounds (Morphology)")
    _ax.set_ylabel("Cosine Distance")
    plt.tight_layout()
    fig_dendrogram
    return (fig_dendrogram,)


# ============================================================================
# Mechanism Separation Statistics
# ============================================================================


@app.cell
def _(adata_profiles, dist_matrix, mo):
    _results = []

    for _mech in MECHANISMS:
        _mech_mask = adata_profiles.obs[f"has_{_mech}"].values
        _n_mech = _mech_mask.sum()

        if _n_mech < 5:
            logger.warning(f"Skipping {_mech}: only {_n_mech} compounds")
            continue

        # Within-mechanism distances (upper triangle only)
        _within_idx = np.where(_mech_mask)[0]
        _within_dists = []
        for _i, _idx_i in enumerate(_within_idx):
            for _idx_j in _within_idx[_i + 1 :]:
                _within_dists.append(dist_matrix[_idx_i, _idx_j])
        _within_dists = np.array(_within_dists)

        # Between-mechanism distances
        _between_idx = np.where(~_mech_mask)[0]
        _between_dists = dist_matrix[np.ix_(_within_idx, _between_idx)].flatten()

        # Mann-Whitney U test
        if len(_within_dists) > 0 and len(_between_dists) > 0:
            _stat, _pval = mannwhitneyu(_within_dists, _between_dists, alternative="less")
        else:
            _stat, _pval = np.nan, np.nan

        _results.append(
            {
                "mechanism": _mech,
                "n_compounds": _n_mech,
                "mean_within_dist": float(_within_dists.mean()) if len(_within_dists) > 0 else np.nan,
                "std_within_dist": float(_within_dists.std()) if len(_within_dists) > 0 else np.nan,
                "mean_between_dist": float(_between_dists.mean()),
                "std_between_dist": float(_between_dists.std()),
                "separation": (
                    float(_between_dists.mean() - _within_dists.mean()) if len(_within_dists) > 0 else np.nan
                ),
                "mann_whitney_stat": float(_stat),
                "mann_whitney_p": float(_pval),
            }
        )

    separation_df = pd.DataFrame(_results)

    _lines = [
        "## Mechanism Separation Results",
        "",
        "| Mechanism | n | Separation | p-value | Sig? |",
        "|-----------|---|------------|---------|------|",
    ]
    for _, _row in separation_df.iterrows():
        _sig = "Yes" if _row["mann_whitney_p"] < 0.05 else "No"
        _lines.append(
            f"| {_row['mechanism']} | {_row['n_compounds']} | "
            f"{_row['separation']:.4f} | {_row['mann_whitney_p']:.2e} | {_sig} |"
        )

    _n_sig = (separation_df["mann_whitney_p"] < 0.05).sum()
    _lines.append(f"\n**{_n_sig} / {len(separation_df)}** mechanisms show significant separation (p < 0.05).")

    mo.md("\n".join(_lines))
    return (separation_df,)


# ============================================================================
# Silhouette Score with Permutation Test
# ============================================================================


@app.cell
def _(adata_profiles, dist_matrix, mo, n_permutations_slider):
    _n_permutations = n_permutations_slider.value

    # Assign numeric labels for primary mechanism
    def _get_mech_label(row):
        for _i, _m in enumerate(MECHANISMS):
            if row.get(f"has_{_m}", False):
                return _i
        return -1

    _mech_labels = np.array([_get_mech_label(row) for _, row in adata_profiles.obs.iterrows()])

    # Filter to compounds with valid labels
    _valid_mask = _mech_labels >= 0
    if _valid_mask.sum() < 10:
        silhouette_obs = np.nan
        silhouette_pval = np.nan
    else:
        _dist_sub = dist_matrix[np.ix_(_valid_mask, _valid_mask)]
        _labels_sub = _mech_labels[_valid_mask]

        _unique_labels = np.unique(_labels_sub)
        if len(_unique_labels) < 2:
            silhouette_obs = np.nan
            silhouette_pval = np.nan
        else:
            silhouette_obs = silhouette_score(_dist_sub, _labels_sub, metric="precomputed")

            _null_scores = []
            for _ in range(_n_permutations):
                _shuffled = np.random.permutation(_labels_sub)
                _null_scores.append(silhouette_score(_dist_sub, _shuffled, metric="precomputed"))

            _null_scores = np.array(_null_scores)
            silhouette_pval = (np.sum(_null_scores >= silhouette_obs) + 1) / (_n_permutations + 1)

    logger.info(f"Silhouette score: {silhouette_obs:.4f} (p={silhouette_pval:.4f})")

    mo.md(f"""
    ## Silhouette Analysis

    **Silhouette score:** {silhouette_obs:.4f} (p = {silhouette_pval:.4f}, {_n_permutations} permutations)

    Positive values indicate that mechanism labels have some correspondence with
    morphological clustering. The p-value tests whether this score exceeds what
    would be expected under random label assignment.
    """)
    return (silhouette_obs, silhouette_pval)


# ============================================================================
# Summary
# ============================================================================


@app.cell
def _(
    adata_chem,
    adata_morph,
    dataset_dropdown,
    matched_df,
    mo,
    separation_df,
    silhouette_obs,
    silhouette_pval,
):
    _dataset = dataset_dropdown.value
    _n_sig = int((separation_df["mann_whitney_p"] < 0.05).sum())

    summary_dict = {
        "task": "MitoTox morphological mechanism analysis",
        "research_question": "Can morphology distinguish mitotoxicity submechanisms?",
        "dataset": _dataset,
        "n_matched_to_jump": len(matched_df),
        "n_in_umap_morph": int(adata_morph.n_obs),
        "n_in_umap_chem": int(adata_chem.n_obs),
        "n_toxic": int(matched_df["is_toxic"].sum()),
        "n_nontoxic": int((~matched_df["is_toxic"]).sum()),
        "mechanism_counts": {MECHANISM_SHORT[m]: int(matched_df[f"has_{m}"].sum()) for m in MECHANISMS},
        "silhouette_score": float(silhouette_obs) if not np.isnan(silhouette_obs) else None,
        "silhouette_pvalue": float(silhouette_pval) if not np.isnan(silhouette_pval) else None,
        "mechanism_separation": separation_df.to_dict(orient="records"),
        "n_mechanisms_significant": _n_sig,
        "interpretation": (
            "Positive separation scores indicate mechanisms cluster in morphology space. "
            "Significant p-values (< 0.05) suggest non-random clustering. "
            f"Overall silhouette: {silhouette_obs:.3f} (p={silhouette_pval:.3f})."
        ),
    }

    mo.md(f"""
    ## Summary

    - **{len(matched_df)}** MitoTox compounds matched to JUMP
    - **{adata_morph.n_obs}** in morphology UMAP, **{adata_chem.n_obs}** in chemistry UMAP
    - **{_n_sig} / {len(separation_df)}** mechanisms show significant morphological separation
    - Silhouette: **{silhouette_obs:.3f}** (p={silhouette_pval:.3f})
    """)
    return (summary_dict,)


# ============================================================================
# Save Outputs
# ============================================================================


@app.cell
def _(dataset_dropdown, mo):
    save_button = mo.ui.button(label="Save all outputs", kind="warn")
    _output_path = PROCESSED_DATA_DIR / "mitotox-analysis" / dataset_dropdown.value
    mo.hstack([save_button, mo.md(f"Saves to `{_output_path}`")])
    return (save_button,)


@app.cell
def _(
    dataset_dropdown,
    fig_chem_full,
    fig_chem_mechanisms,
    fig_chem_toxic,
    fig_dendrogram,
    fig_morph_full,
    fig_morph_mechanisms,
    fig_morph_toxic,
    mo,
    save_button,
    separation_df,
    summary_dict,
):
    mo.stop(not save_button.value)

    _output_dir = PROCESSED_DATA_DIR / "mitotox-analysis" / dataset_dropdown.value
    _output_dir.mkdir(parents=True, exist_ok=True)

    _saved = []

    # Save figures
    for _name, _fig in [
        ("umap_morphology_full", fig_morph_full),
        ("umap_morphology_by_mechanism", fig_morph_mechanisms),
        ("umap_morphology_toxic", fig_morph_toxic),
        ("umap_chemistry_full", fig_chem_full),
        ("umap_chemistry_by_mechanism", fig_chem_mechanisms),
        ("umap_chemistry_toxic", fig_chem_toxic),
        ("dendrogram_morphology", fig_dendrogram),
    ]:
        if _fig is not None:
            _fig.savefig(_output_dir / f"{_name}.png", dpi=DEFAULT_DPI, bbox_inches="tight")
            _fig.savefig(_output_dir / f"{_name}.pdf", bbox_inches="tight")
            _saved.append(f"{_name}.png/.pdf")

    # Save CSV and JSON
    separation_df.to_csv(_output_dir / "mechanism_separation.csv", index=False)
    _saved.append("mechanism_separation.csv")

    with open(_output_dir / "summary.json", "w") as _f:
        json.dump(summary_dict, _f, indent=2)
    _saved.append("summary.json")

    logger.success(f"Saved {len(_saved)} outputs to {_output_dir}")
    mo.md(f"**Saved {len(_saved)} outputs** to `{_output_dir}`:\n" + "\n".join(f"- `{f}`" for f in _saved))
    return


@app.function
def run_mitotox_analysis(
    dataset,
    output_dir=None,
) -> str:
    """Run the full MitoTox morphological mechanism analysis.

    Parameters
    ----------
    dataset : str
        Dataset name, e.g. "compound_no_source7".
    output_dir : str or Path, optional
        Output directory. Defaults to PROCESSED_DATA_DIR / "mitotox-analysis" / dataset.

    Returns
    -------
    str
        Path to the output directory.
    """
    dataset = str(dataset)

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "mitotox-analysis" / dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running MitoTox analysis: dataset={dataset}")

    # Step 1: Load MitoTox annotations from database
    con = duckdb.connect(str(METADATA_DB), read_only=True)
    matched_df = con.execute("""
        SELECT
            Metadata_JCP2022,
            Metadata_mitotox_label AS mitotox_label,
            Metadata_mitotox_toxic,
            Metadata_mitotox_mechanisms AS functional_mechanism
        FROM mitotox_annotations
    """).df()
    con.close()
    logger.info(f"Loaded {len(matched_df)} MitoTox compounds from database")

    # Parse mechanisms into binary columns
    for _mech in MECHANISMS:
        matched_df[f"has_{_mech}"] = matched_df["functional_mechanism"].fillna("").str.contains(_mech, regex=False)
    matched_df["is_toxic"] = matched_df["Metadata_mitotox_toxic"] == 1

    # Step 2: Load UMAPs
    matched_jcp = set(matched_df["Metadata_JCP2022"])

    logger.info("Loading precomputed morphology UMAP...")
    adata_morph_full = sc.read_h5ad(MORPH_UMAP_PATH)

    logger.info("Loading precomputed chemistry UMAP...")
    adata_chem_full = sc.read_h5ad(CHEM_UMAP_PATH)
    if "Metadata_JCP2022" in adata_chem_full.obs.columns:
        adata_chem_full.obs["JCP2022"] = adata_chem_full.obs["Metadata_JCP2022"]

    # Find intersection
    _morph_jcp = set(adata_morph_full.obs["JCP2022"])
    _chem_jcp = set(adata_chem_full.obs["JCP2022"])
    common_jcp = _morph_jcp & _chem_jcp

    adata_morph_full = adata_morph_full[adata_morph_full.obs["JCP2022"].isin(common_jcp)].copy()
    adata_chem_full = adata_chem_full[adata_chem_full.obs["JCP2022"].isin(common_jcp)].copy()

    matched_jcp_common = matched_jcp & common_jcp

    # MitoTox subsets with annotations
    _mech_cols = [f"has_{m}" for m in MECHANISMS] + ["is_toxic", "mitotox_label", "functional_mechanism"]
    _matched_dedup = matched_df.drop_duplicates(subset=["Metadata_JCP2022"])[["Metadata_JCP2022"] + _mech_cols]
    _matched_dedup = _matched_dedup.rename(columns={"Metadata_JCP2022": "JCP2022"})

    _morph_mask = adata_morph_full.obs["JCP2022"].isin(matched_jcp_common)
    adata_morph = adata_morph_full[_morph_mask].copy()
    _morph_obs = adata_morph.obs.copy().reset_index()
    _morph_obs = _morph_obs.merge(_matched_dedup, on="JCP2022", how="left")
    _morph_obs = _morph_obs.set_index(adata_morph.obs.index.name or "index")
    adata_morph.obs = _morph_obs

    _chem_mask = adata_chem_full.obs["JCP2022"].isin(matched_jcp_common)
    adata_chem = adata_chem_full[_chem_mask].copy()
    _chem_obs = adata_chem.obs.copy().reset_index()
    _chem_obs = _chem_obs.merge(_matched_dedup, on="JCP2022", how="left")
    _chem_obs = _chem_obs.set_index(adata_chem.obs.index.name or "index")
    adata_chem.obs = _chem_obs

    # Step 3: Generate and save UMAP plots
    fig_morph_full = plot_umap_full_with_mitotox(
        adata_morph_full.obsm["X_umap"], adata_morph.obsm["X_umap"], adata_morph.obs, "Morphology"
    )
    fig_morph_full.savefig(output_dir / "umap_morphology_full.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_morph_full)

    fig_morph_mechanisms = plot_umap_by_mechanisms(adata_morph.obsm["X_umap"], adata_morph.obs, "Morphology UMAP")
    fig_morph_mechanisms.savefig(
        output_dir / "umap_morphology_by_mechanism.png", dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig_morph_mechanisms)

    fig_morph_toxic = plot_umap_toxic_vs_nontoxic(adata_morph.obsm["X_umap"], adata_morph.obs, "Morphology UMAP")
    fig_morph_toxic.savefig(output_dir / "umap_morphology_toxic.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_morph_toxic)

    fig_chem_full = plot_umap_full_with_mitotox(
        adata_chem_full.obsm["X_umap"], adata_chem.obsm["X_umap"], adata_chem.obs, "Chemistry"
    )
    fig_chem_full.savefig(output_dir / "umap_chemistry_full.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_chem_full)

    fig_chem_mechanisms = plot_umap_by_mechanisms(adata_chem.obsm["X_umap"], adata_chem.obs, "Chemistry UMAP")
    fig_chem_mechanisms.savefig(
        output_dir / "umap_chemistry_by_mechanism.png", dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig_chem_mechanisms)

    fig_chem_toxic = plot_umap_toxic_vs_nontoxic(adata_chem.obsm["X_umap"], adata_chem.obs, "Chemistry UMAP")
    fig_chem_toxic.savefig(output_dir / "umap_chemistry_toxic.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_chem_toxic)

    # Step 4: Hierarchical clustering on full profiles
    adata_profiles = load_profiles(dataset, level="perturbation")
    _profile_mask = adata_profiles.obs["JCP2022"].isin(set(matched_df["Metadata_JCP2022"]))
    adata_profiles = adata_profiles[_profile_mask].copy()

    _prof_obs = adata_profiles.obs.copy().reset_index()
    _prof_obs = _prof_obs.merge(_matched_dedup, on="JCP2022", how="left")
    _prof_obs = _prof_obs.set_index(adata_profiles.obs.index.name or "index")
    adata_profiles.obs = _prof_obs

    # L2 normalize for cosine distance
    _X = adata_profiles.X
    _norms = np.linalg.norm(_X, axis=1, keepdims=True)
    X_norm = _X / np.maximum(_norms, 1e-10)

    _distances = pdist(X_norm, metric="cosine")
    dist_matrix = squareform(_distances)

    # Step 5: Mechanism separation statistics
    _results = []
    for _mech in MECHANISMS:
        _mech_mask = adata_profiles.obs[f"has_{_mech}"].values
        _n_mech = _mech_mask.sum()
        if _n_mech < 5:
            continue

        _within_idx = np.where(_mech_mask)[0]
        _within_dists = []
        for _i, _idx_i in enumerate(_within_idx):
            for _idx_j in _within_idx[_i + 1 :]:
                _within_dists.append(dist_matrix[_idx_i, _idx_j])
        _within_dists = np.array(_within_dists)

        _between_idx = np.where(~_mech_mask)[0]
        _between_dists = dist_matrix[np.ix_(_within_idx, _between_idx)].flatten()

        if len(_within_dists) > 0 and len(_between_dists) > 0:
            _stat, _pval = mannwhitneyu(_within_dists, _between_dists, alternative="less")
        else:
            _stat, _pval = np.nan, np.nan

        _results.append(
            {
                "mechanism": _mech,
                "n_compounds": _n_mech,
                "mean_within_dist": float(_within_dists.mean()) if len(_within_dists) > 0 else np.nan,
                "mean_between_dist": float(_between_dists.mean()),
                "separation": float(_between_dists.mean() - _within_dists.mean()) if len(_within_dists) > 0 else np.nan,
                "mann_whitney_p": float(_pval),
            }
        )

    separation_df = pd.DataFrame(_results)
    separation_df.to_csv(output_dir / "mechanism_separation.csv", index=False)

    # Step 6: Silhouette score with permutation test
    def _get_mech_label(row):
        for _i, _m in enumerate(MECHANISMS):
            if row.get(f"has_{_m}", False):
                return _i
        return -1

    _mech_labels = np.array([_get_mech_label(row) for _, row in adata_profiles.obs.iterrows()])
    _valid_mask = _mech_labels >= 0
    _n_permutations = 1000

    if _valid_mask.sum() >= 10:
        _dist_sub = dist_matrix[np.ix_(_valid_mask, _valid_mask)]
        _labels_sub = _mech_labels[_valid_mask]
        _unique_labels = np.unique(_labels_sub)

        if len(_unique_labels) >= 2:
            silhouette_obs = silhouette_score(_dist_sub, _labels_sub, metric="precomputed")
            _null_scores = []
            for _ in range(_n_permutations):
                _shuffled = np.random.permutation(_labels_sub)
                _null_scores.append(silhouette_score(_dist_sub, _shuffled, metric="precomputed"))
            silhouette_pval = (np.sum(np.array(_null_scores) >= silhouette_obs) + 1) / (_n_permutations + 1)
        else:
            silhouette_obs = float("nan")
            silhouette_pval = float("nan")
    else:
        silhouette_obs = float("nan")
        silhouette_pval = float("nan")

    # Step 7: Save summary JSON
    _n_sig = int((separation_df["mann_whitney_p"] < 0.05).sum()) if len(separation_df) > 0 else 0

    summary = {
        "task": "MitoTox morphological mechanism analysis",
        "research_question": "Can morphology distinguish mitotoxicity submechanisms?",
        "dataset": dataset,
        "n_matched_to_jump": len(matched_df),
        "n_in_umap_morph": int(adata_morph.n_obs),
        "n_in_umap_chem": int(adata_chem.n_obs),
        "n_toxic": int(matched_df["is_toxic"].sum()),
        "n_nontoxic": int((~matched_df["is_toxic"]).sum()),
        "mechanism_counts": {MECHANISM_SHORT[m]: int(matched_df[f"has_{m}"].sum()) for m in MECHANISMS},
        "silhouette_score": float(silhouette_obs) if not np.isnan(silhouette_obs) else None,
        "silhouette_pvalue": float(silhouette_pval) if not np.isnan(silhouette_pval) else None,
        "mechanism_separation": separation_df.to_dict(orient="records"),
        "n_mechanisms_significant": _n_sig,
    }

    Path(output_dir, "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"MitoTox analysis complete: {output_dir}")
    return str(output_dir)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
