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
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import (
        COPAIRS_RESULTS_DB,
        DEFAULT_DPI,
        METADATA_DB,
        PROCESSED_DATA_DIR,
    )
    from nb03_ss_profiles import load_profiles, load_umap
    from nb04_ss_visualization import compute_umap_bounds

    DATASET = "compound_no_source7"

    # Additional MMP9 inhibitors found via ChEMBL bioactivity (CHEMBL321, IC50 < 1uM)
    # structure-matched to JUMP by InChIKey. These are not annotated as MMP9 inhibitors
    # in any pipeline annotation source (repurposing hub, chemical probes, motive), so
    # their Metadata_repurposing_target and Metadata_repurposing_moa columns will be empty.
    # Batimastat is in the repurposing hub but annotated as MMP2/8/12/16 (not MMP9).
    EXTRA_MMP9_JCPS = [
        "JCP2022_042691",  # tanomastat (CHEMBL261932)
        "JCP2022_103291",  # batimastat (CHEMBL279786)
        "JCP2022_109077",  # prinomastat (CHEMBL75094)
        "JCP2022_111234",  # S-3304 (CHEMBL297792)
    ]

    # Fallback names for compounds not in the Repurposing Hub
    CHEMBL_NAMES = {
        "JCP2022_042691": "tanomastat",
        "JCP2022_109077": "prinomastat",
        "JCP2022_111234": "S-3304",
    }


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Target Inhibitor Analysis

    Identifies inhibitors of a chosen target gene among JUMP compounds,
    with MoA context and UMAP visualization.

    **Workflow:**

    1. Search all annotation sources (repurposing hub, chemical probes,
       motive/OpenTargets) for compounds targeting the gene
    2. Optionally include extra compounds found via external structure matching
    3. Check phenotypic activity from copairs results
    4. Visualize on UMAP (target vs background, MoA peers)
    5. Compute pairwise profile similarity among target compounds

    **Default target:** MMP9 (with ChEMBL-matched extras). Change the target
    gene and extra JCPs in the controls below.
    """)
    return


# ============================================================================
# Controls
# ============================================================================


@app.cell
def _(mo):
    target_input = mo.ui.text(value="MMP9", label="Target gene")
    extra_jcps_input = mo.ui.text(
        value=", ".join(EXTRA_MMP9_JCPS),
        label="Extra JCP IDs (comma-separated)",
    )
    skip_plots_switch = mo.ui.switch(value=False, label="Skip plots")
    mo.vstack(
        [
            mo.hstack([target_input, skip_plots_switch]),
            extra_jcps_input,
        ]
    )
    return (extra_jcps_input, skip_plots_switch, target_input)


# ============================================================================
# Query Target Compounds
# ============================================================================


@app.function
def get_target_compounds(
    con: duckdb.DuckDBPyConnection,
    target: str,
    extra_jcps: list[str] | None = None,
) -> pd.DataFrame:
    """Query compounds targeting a gene across all annotation sources, plus extras."""
    pattern = f"%{target}%"
    jcp_df = con.execute(
        """
        SELECT DISTINCT Metadata_JCP2022
        FROM compound_metadata
        WHERE Metadata_repurposing_target LIKE ?
           OR Metadata_chmprb_target_genes LIKE ?
           OR Metadata_motive_gene_opentargets LIKE ?
        """,
        [pattern, pattern, pattern],
    ).df()

    all_jcps = set(jcp_df["Metadata_JCP2022"])
    if extra_jcps:
        all_jcps.update(extra_jcps)

    if not all_jcps:
        return pd.DataFrame()

    df = con.execute(
        """
        SELECT
            cm.Metadata_JCP2022,
            r.Metadata_repurposing_name,
            r.Metadata_repurposing_target,
            r.Metadata_repurposing_moa,
            r.Metadata_repurposing_clinical_phase,
            r.Metadata_repurposing_disease_area,
            r.Metadata_repurposing_indication,
            cm.Metadata_SMILES,
            cm.Metadata_InChIKey
        FROM compound_metadata cm
        LEFT JOIN repurposing_hub_annotations r
          ON cm.Metadata_JCP2022 = r.Metadata_JCP2022
        WHERE cm.Metadata_JCP2022 = ANY(?)
        """,
        [list(all_jcps)],
    ).df()

    # Determine annotation sources
    all_jcp_list = list(df["Metadata_JCP2022"])
    other_sources = con.execute(
        """
        SELECT
            jcps.jcp AS Metadata_JCP2022,
            bool_or(cp.Metadata_JCP2022 IS NOT NULL) AS has_chem_probes,
            bool_or(mt.Metadata_JCP2022 IS NOT NULL) AS has_motive
        FROM (SELECT UNNEST(?) AS jcp) jcps
        LEFT JOIN chemical_probes cp
          ON jcps.jcp = cp.Metadata_JCP2022 AND cp.Metadata_chmprb_target_genes LIKE ?
        LEFT JOIN motive_targets_opentargets mt
          ON jcps.jcp = mt.Metadata_JCP2022 AND mt.Metadata_motive_gene_opentargets LIKE ?
        GROUP BY jcps.jcp
        """,
        [all_jcp_list, pattern, pattern],
    ).df()
    other_map = {
        row["Metadata_JCP2022"]: (row["has_chem_probes"], row["has_motive"]) for _, row in other_sources.iterrows()
    }
    extra_set = set(extra_jcps) if extra_jcps else set()
    sources_col = []
    for _, row in df.iterrows():
        sources = []
        if pd.notna(row.get("Metadata_repurposing_target")) and target in str(row["Metadata_repurposing_target"]):
            sources.append("repurposing")
        jcp = row["Metadata_JCP2022"]
        cp_flag, mt_flag = other_map.get(jcp, (False, False))
        if cp_flag:
            sources.append("chem_probes")
        if mt_flag:
            sources.append("motive")
        if jcp in extra_set:
            sources.append("chembl_structure")
        sources_col.append("|".join(sources) if sources else "extra")
    df["annotation_sources"] = sources_col

    df["target_selective"] = df["Metadata_repurposing_target"].apply(
        lambda x: x.strip() == target if pd.notna(x) else False
    )
    df["n_targets"] = df["Metadata_repurposing_target"].apply(lambda x: len(x.split("|")) if pd.notna(x) else 0)
    df = df.sort_values(["target_selective", "n_targets"], ascending=[False, True])
    return df


@app.function
def get_moa_context(con: duckdb.DuckDBPyConnection, target_df: pd.DataFrame) -> pd.DataFrame:
    """Find all compounds sharing the same MoA(s) as the target compounds."""
    all_moas = {m.strip() for s in target_df["Metadata_repurposing_moa"].dropna() for m in s.split("|") if m.strip()}

    if not all_moas:
        return pd.DataFrame()

    moa_patterns = [f"%{moa}%" for moa in all_moas]
    conditions = " OR ".join(["r.Metadata_repurposing_moa LIKE ?" for _ in all_moas])
    moa_df = con.execute(
        f"""
        SELECT
            r.Metadata_JCP2022,
            r.Metadata_repurposing_name,
            r.Metadata_repurposing_target,
            r.Metadata_repurposing_moa,
            r.Metadata_repurposing_clinical_phase
        FROM repurposing_hub_annotations r
        WHERE {conditions}
        """,
        moa_patterns,
    ).df()

    target_jcps = set(target_df["Metadata_JCP2022"])
    moa_df["is_target_compound"] = moa_df["Metadata_JCP2022"].isin(target_jcps)
    moa_df = moa_df.sort_values(["is_target_compound", "Metadata_repurposing_moa"], ascending=[False, True])
    return moa_df


@app.function
def get_activity_status(target_df: pd.DataFrame) -> pd.DataFrame:
    """Join target compounds with phenotypic activity from copairs results."""
    jcp_ids = target_df["Metadata_JCP2022"].tolist()
    act_con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    try:
        activity = act_con.execute(
            """
            SELECT Metadata_JCP2022,
                   mean_normalized_average_precision AS nmAP,
                   corrected_p_value AS fdr_p,
                   below_corrected_p AS active
            FROM activity_results
            WHERE _dataset = ? AND _preprocessing = 'activity_no_target2'
              AND _filter = 'all_sources' AND _activity_params = 'default'
              AND Metadata_JCP2022 = ANY(?)
            """,
            [DATASET, jcp_ids],
        ).df()
    finally:
        act_con.close()
    merged = target_df.merge(activity, on="Metadata_JCP2022", how="left")
    return merged.sort_values("nmAP", ascending=False)


# ============================================================================
# Load Data
# ============================================================================


@app.cell
def _(extra_jcps_input, mo, target_input):
    mo.stop(
        not METADATA_DB.exists(),
        mo.md(f"**Error:** Metadata database not found at `{METADATA_DB}`."),
    )

    target = target_input.value.strip()
    mo.stop(not target, mo.md("**Error:** Please enter a target gene symbol."))

    # Parse extra JCPs
    _extra_text = extra_jcps_input.value.strip()
    extra_jcps = [j.strip() for j in _extra_text.split(",") if j.strip()] if _extra_text else None

    meta_con = duckdb.connect(str(METADATA_DB), read_only=True)

    logger.info(f"Querying compounds targeting {target}...")
    target_df = get_target_compounds(meta_con, target, extra_jcps=extra_jcps)

    # Fill missing names from ChEMBL lookup
    _name_mask = target_df["Metadata_repurposing_name"].isna()
    target_df.loc[_name_mask, "Metadata_repurposing_name"] = target_df.loc[_name_mask, "Metadata_JCP2022"].map(
        CHEMBL_NAMES
    )

    mo.stop(
        len(target_df) == 0,
        mo.md(f"No compounds found for target **{target}**."),
    )

    # Get activity status
    logger.info("Checking phenotypic activity...")
    target_df = get_activity_status(target_df)

    # Get MoA context
    logger.info("Finding compounds with shared MoA...")
    moa_df = get_moa_context(meta_con, target_df)
    meta_con.close()

    _n_active = int(target_df["active"].sum()) if "active" in target_df.columns else 0
    _n_profiled = int(target_df["nmAP"].notna().sum()) if "nmAP" in target_df.columns else 0
    _n_same_moa = len(moa_df[~moa_df["is_target_compound"]]) if len(moa_df) > 0 else 0
    _n_selective = int(target_df["target_selective"].sum())

    mo.md(f"""
    ## Data Loaded

    **Target:** {target} | **Compounds found:** {len(target_df)}

    | Metric | Count |
    |--------|-------|
    | Selective ({target} only) | {_n_selective} |
    | Multi-target | {len(target_df) - _n_selective} |
    | Profiled | {_n_profiled} |
    | Phenotypically active | {_n_active} / {_n_profiled} |
    | Missing profiles | {len(target_df) - _n_profiled} |
    | Shared MoA peers | {_n_same_moa} |
    """)
    return (extra_jcps, moa_df, target, target_df)


# ============================================================================
# Compound Table
# ============================================================================


@app.cell
def _(mo, target_df):
    _display_cols = [
        "Metadata_JCP2022",
        "Metadata_repurposing_name",
        "Metadata_repurposing_target",
        "Metadata_repurposing_moa",
        "annotation_sources",
        "target_selective",
    ]
    if "nmAP" in target_df.columns:
        _display_cols.extend(["nmAP", "fdr_p", "active"])

    _available = [c for c in _display_cols if c in target_df.columns]
    mo.ui.table(target_df[_available], label="Target compounds")
    return


# ============================================================================
# UMAP Visualization
# ============================================================================


@app.function
def plot_target_umap(
    adata: sc.AnnData,
    target_jcps: set[str],
    target: str,
) -> plt.Figure:
    """Plot UMAP with target compounds highlighted over gray background."""
    xlim, ylim = compute_umap_bounds(adata)
    coords = adata.obsm["X_umap"]
    jcp_to_idx = {j: i for i, j in enumerate(adata.obs["JCP2022"].values)}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=0.5, alpha=0.2, rasterized=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    valid = target_jcps & set(jcp_to_idx)
    if valid:
        idx = np.array([jcp_to_idx[j] for j in valid])
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c="#d62728",
            s=40,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label=f"{target} ({len(idx)})",
            rasterized=True,
        )

    ax.set_title(f"{target} inhibitors in morphological space")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig


@app.function
def plot_moa_context_umap(
    adata: sc.AnnData,
    target_jcps: set[str],
    moa_df: pd.DataFrame,
    target: str,
) -> plt.Figure:
    """Plot UMAP with target compounds + MoA peers colored by group."""
    xlim, ylim = compute_umap_bounds(adata)
    coords = adata.obsm["X_umap"]
    jcp_to_idx = {j: i for i, j in enumerate(adata.obs["JCP2022"].values)}

    moa_peer_jcps = set(moa_df[~moa_df["is_target_compound"]]["Metadata_JCP2022"])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=0.5, alpha=0.2, rasterized=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    valid_peers = moa_peer_jcps & set(jcp_to_idx)
    if valid_peers:
        idx = np.array([jcp_to_idx[j] for j in valid_peers])
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c="#1f77b4",
            s=20,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.3,
            zorder=4,
            label=f"Same MoA peers ({len(idx)})",
            rasterized=True,
        )

    valid_tgt = target_jcps & set(jcp_to_idx)
    if valid_tgt:
        idx = np.array([jcp_to_idx[j] for j in valid_tgt])
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c="#d62728",
            s=60,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label=f"{target} ({len(idx)})",
            rasterized=True,
        )

    ax.set_title(f"{target} inhibitors vs MoA peers in morphological space")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig


@app.function
def plot_per_moa_umap(
    adata: sc.AnnData,
    moa_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target: str,
) -> plt.Figure | None:
    """Plot one UMAP per MoA showing target compounds vs peers."""
    all_moas = {m.strip() for s in target_df["Metadata_repurposing_moa"].dropna() for m in s.split("|") if m.strip()}
    if not all_moas:
        return None

    xlim, ylim = compute_umap_bounds(adata)
    coords = adata.obsm["X_umap"]
    jcp_ids = set(adata.obs["JCP2022"].values)
    jcp_to_idx = {j: i for i, j in enumerate(adata.obs["JCP2022"].values)}

    n_moas = len(all_moas)
    cols = min(n_moas, 3)
    rows = (n_moas + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)

    for ax_idx, moa in enumerate(sorted(all_moas)):
        ax = axes[ax_idx // cols][ax_idx % cols]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c="lightgray",
            s=0.5,
            alpha=0.2,
            rasterized=True,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        peer_jcps = (
            set(
                moa_df[
                    (~moa_df["is_target_compound"])
                    & (moa_df["Metadata_repurposing_moa"].fillna("").str.contains(moa, regex=False))
                ]["Metadata_JCP2022"]
            )
            & jcp_ids
        )

        valid_peers = peer_jcps & set(jcp_to_idx)
        if valid_peers:
            idx = np.array([jcp_to_idx[j] for j in valid_peers])
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                c="#1f77b4",
                s=20,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.3,
                zorder=4,
                label=f"Peers ({len(idx)})",
                rasterized=True,
            )

        tgt_with_moa = (
            set(
                target_df[target_df["Metadata_repurposing_moa"].fillna("").str.contains(moa, regex=False)][
                    "Metadata_JCP2022"
                ]
            )
            & jcp_ids
        )

        valid_tgt = tgt_with_moa & set(jcp_to_idx)
        if valid_tgt:
            idx = np.array([jcp_to_idx[j] for j in valid_tgt])
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                c="#d62728",
                s=60,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                label=f"{target} ({len(idx)})",
                rasterized=True,
            )

        ax.set_title(moa, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)

    for ax_idx in range(n_moas, rows * cols):
        axes[ax_idx // cols][ax_idx % cols].set_visible(False)

    fig.suptitle(f"{target} inhibitors by MoA", fontsize=14)
    plt.tight_layout()
    return fig


# ============================================================================
# Similarity Heatmap
# ============================================================================


@app.function
def plot_similarity_heatmap(
    target_df: pd.DataFrame,
) -> plt.Figure | None:
    """Compute and plot pairwise cosine similarity among target compound profiles."""
    adata = load_profiles(DATASET, level="perturbation")
    jcps = list(target_df["Metadata_JCP2022"])
    names_map = {
        jcp: name if pd.notna(name) else jcp
        for jcp, name in zip(target_df["Metadata_JCP2022"], target_df["Metadata_repurposing_name"])
    }

    mask = adata.obs["JCP2022"].isin(jcps)
    sub = adata[mask].copy()
    if sub.n_obs < 2:
        logger.warning(f"Only {sub.n_obs} compounds have profiles - skipping heatmap")
        return None

    logger.info(f"Computing similarity for {sub.n_obs} / {len(jcps)} compounds with profiles")

    X = sub.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    sim = cosine_similarity(X)

    labels = [names_map.get(j, j) for j in sub.obs["JCP2022"]]

    # Cluster
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist), method="ward")
    order = leaves_list(link)
    sim_ordered = sim[np.ix_(order, order)]
    labels_ordered = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim_ordered, cmap="RdBu_r", vmin=-0.5, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(labels_ordered)))
    ax.set_xticklabels(labels_ordered, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels_ordered)))
    ax.set_yticklabels(labels_ordered, fontsize=10)

    for i in range(len(labels_ordered)):
        for j in range(len(labels_ordered)):
            ax.text(
                j,
                i,
                f"{sim_ordered[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(sim_ordered[i, j]) > 0.5 else "black",
            )

    plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)
    ax.set_title("Target inhibitor profile similarity\n(perturbation-level profiles)")
    plt.tight_layout()
    return fig


# ============================================================================
# Plot Cells
# ============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## UMAP Visualizations""")
    return


@app.cell
def _(moa_df, mo, skip_plots_switch, target, target_df):
    mo.stop(skip_plots_switch.value, mo.md("Plots skipped."))

    logger.info("Loading UMAP coordinates...")
    umap_adata = load_umap(DATASET, level="perturbation", metric="cosine", filter_name="all")

    target_jcps = set(target_df["Metadata_JCP2022"])

    # Plot 1: Target vs background
    fig_target_umap = plot_target_umap(umap_adata, target_jcps, target)

    # Plot 2: Target + MoA peers
    fig_moa_umap = None
    if len(moa_df) > 0:
        fig_moa_umap = plot_moa_context_umap(umap_adata, target_jcps, moa_df, target)

    # Plot 3: Per-MoA breakdown
    fig_per_moa = None
    if len(moa_df) > 0:
        fig_per_moa = plot_per_moa_umap(umap_adata, moa_df, target_df, target)

    mo.vstack(
        [
            fig_target_umap,
            fig_moa_umap if fig_moa_umap is not None else mo.md(""),
            fig_per_moa if fig_per_moa is not None else mo.md(""),
        ]
    )
    return (fig_moa_umap, fig_per_moa, fig_target_umap, target_jcps, umap_adata)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Pairwise Profile Similarity""")
    return


@app.cell
def _(mo, skip_plots_switch, target_df):
    mo.stop(skip_plots_switch.value, mo.md("Plots skipped."))

    fig_heatmap = plot_similarity_heatmap(target_df)
    mo.stop(fig_heatmap is None, mo.md("Too few compounds with profiles for heatmap."))
    fig_heatmap
    return (fig_heatmap,)


# ============================================================================
# MoA Context Table
# ============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## MoA Context""")
    return


@app.cell
def _(mo, moa_df, target, target_df):
    if len(moa_df) == 0:
        mo.output.replace(mo.md("No MoA annotations found for target compounds."))
    else:
        _moa_counts = {}
        for _moa_str in target_df["Metadata_repurposing_moa"].dropna():
            for _moa in _moa_str.split("|"):
                _moa = _moa.strip()
                if _moa and _moa not in _moa_counts:
                    _n_total = len(
                        moa_df[moa_df["Metadata_repurposing_moa"].fillna("").str.contains(_moa, regex=False)]
                    )
                    _n_target = len(
                        target_df[target_df["Metadata_repurposing_moa"].fillna("").str.contains(_moa, regex=False)]
                    )
                    _moa_counts[_moa] = {
                        "target_compounds": _n_target,
                        "total_compounds": _n_total,
                    }

        _lines = [
            f"| MoA | {target} compounds | Total in JUMP |",
            "|-----|-------------------|---------------|",
        ]
        for _moa_name, _counts in _moa_counts.items():
            _lines.append(f"| {_moa_name} | {_counts['target_compounds']} | {_counts['total_compounds']} |")

        moa_counts = _moa_counts
        mo.md("\n".join(_lines))
    return (moa_counts,)


# ============================================================================
# Summary & Save
# ============================================================================


@app.cell
def _(mo, target):
    save_button = mo.ui.button(label="Save all outputs", kind="warn")
    _output_path = PROCESSED_DATA_DIR / "mmp9-inhibitors"
    mo.hstack([save_button, mo.md(f"Saves to `{_output_path}` (target: {target})")])
    return (save_button,)


@app.cell
def _(
    mo,
    moa_counts,
    moa_df,
    save_button,
    skip_plots_switch,
    target,
    target_df,
):
    mo.stop(not save_button.value)

    _output_dir = PROCESSED_DATA_DIR / "mmp9-inhibitors"
    _output_dir.mkdir(parents=True, exist_ok=True)

    _saved = []

    # Save target compounds CSV
    target_df.to_csv(_output_dir / "target_inhibitors.csv", index=False)
    _saved.append("target_inhibitors.csv")

    # Save MoA context
    if len(moa_df) > 0:
        moa_df.to_csv(_output_dir / "moa_context.csv", index=False)
        _saved.append("moa_context.csv")

    # Save plots if not skipped
    if not skip_plots_switch.value:
        # We need to re-generate since figure variables may not be in scope
        # depending on cell execution order. Use the @app.function helpers.
        _umap_adata = load_umap(DATASET, level="perturbation", metric="cosine", filter_name="all")
        _target_jcps = set(target_df["Metadata_JCP2022"])

        _fig1 = plot_target_umap(_umap_adata, _target_jcps, target)
        _fig1.savefig(
            _output_dir / "umap_target_vs_background.png",
            dpi=DEFAULT_DPI,
            bbox_inches="tight",
        )
        plt.close(_fig1)
        _saved.append("umap_target_vs_background.png")

        if len(moa_df) > 0:
            _fig2 = plot_moa_context_umap(_umap_adata, _target_jcps, moa_df, target)
            _fig2.savefig(
                _output_dir / "umap_moa_context.png",
                dpi=DEFAULT_DPI,
                bbox_inches="tight",
            )
            plt.close(_fig2)
            _saved.append("umap_moa_context.png")

            _fig3 = plot_per_moa_umap(_umap_adata, moa_df, target_df, target)
            if _fig3 is not None:
                _fig3.savefig(
                    _output_dir / "umap_per_moa.png",
                    dpi=DEFAULT_DPI,
                    bbox_inches="tight",
                )
                plt.close(_fig3)
                _saved.append("umap_per_moa.png")

        _fig4 = plot_similarity_heatmap(target_df)
        if _fig4 is not None:
            _fig4.savefig(
                _output_dir / "similarity_heatmap.png",
                dpi=DEFAULT_DPI,
                bbox_inches="tight",
            )
            plt.close(_fig4)
            _saved.append("similarity_heatmap.png")

    # Summary JSON
    _n_active = int(target_df["active"].sum()) if "active" in target_df.columns else 0
    _n_profiled = int(target_df["nmAP"].notna().sum()) if "nmAP" in target_df.columns else 0
    _n_selective = int(target_df["target_selective"].sum())
    _n_same_moa = len(moa_df[~moa_df["is_target_compound"]]) if len(moa_df) > 0 else 0

    _compound_cols = [
        "Metadata_JCP2022",
        "Metadata_repurposing_name",
        "Metadata_repurposing_target",
        "Metadata_repurposing_moa",
        "annotation_sources",
        "target_selective",
        "nmAP",
        "fdr_p",
        "active",
    ]
    _available_cols = [c for c in _compound_cols if c in target_df.columns]

    _summary = {
        "target_gene": target,
        "total_compounds": len(target_df),
        "target_selective": _n_selective,
        "multi_target": int(len(target_df) - _n_selective),
        "activity": {
            "profiled": _n_profiled,
            "active": _n_active,
            "missing_profiles": int(len(target_df) - _n_profiled),
        },
        "moa_context": {
            "shared_moa_compounds": _n_same_moa,
            "moa_breakdown": moa_counts if moa_counts else {},
        },
        "compounds": target_df[_available_cols].to_dict(orient="records"),
    }

    with open(_output_dir / "summary.json", "w") as _f:
        json.dump(_summary, _f, indent=2, default=str)
    _saved.append("summary.json")

    logger.success(f"Saved {len(_saved)} outputs to {_output_dir}")
    mo.md(f"**Saved {len(_saved)} outputs** to `{_output_dir}`:\n" + "\n".join(f"- `{f}`" for f in _saved))
    return


@app.function
def run_mmp9_analysis(
    output_dir=None,
) -> str:
    """Run the full MMP9 inhibitor analysis pipeline.

    Parameters
    ----------
    output_dir : str or Path, optional
        Output directory. Defaults to PROCESSED_DATA_DIR / "mmp9-inhibitors".

    Returns
    -------
    str
        Path to the output directory.
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "mmp9-inhibitors"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running MMP9 inhibitor analysis")

    target = "MMP9"
    extra_jcps = EXTRA_MMP9_JCPS

    # Step 1: Query target compounds
    meta_con = duckdb.connect(str(METADATA_DB), read_only=True)

    target_df = get_target_compounds(meta_con, target, extra_jcps=extra_jcps)

    # Fill missing names from ChEMBL lookup
    _name_mask = target_df["Metadata_repurposing_name"].isna()
    target_df.loc[_name_mask, "Metadata_repurposing_name"] = target_df.loc[_name_mask, "Metadata_JCP2022"].map(
        CHEMBL_NAMES
    )

    if len(target_df) == 0:
        logger.error("No compounds found for MMP9")
        summary = {"target_gene": target, "error": "No compounds found"}
        Path(output_dir, "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return str(output_dir)

    # Step 2: Get activity status
    target_df = get_activity_status(target_df)

    # Step 3: Get MoA context
    moa_df = get_moa_context(meta_con, target_df)
    meta_con.close()

    # Step 4: Save CSVs
    target_df.to_csv(output_dir / "target_inhibitors.csv", index=False)
    if len(moa_df) > 0:
        moa_df.to_csv(output_dir / "moa_context.csv", index=False)

    # Step 5: Generate and save plots
    _umap_adata = load_umap(DATASET, level="perturbation", metric="cosine", filter_name="all")
    _target_jcps = set(target_df["Metadata_JCP2022"])

    fig1 = plot_target_umap(_umap_adata, _target_jcps, target)
    fig1.savefig(output_dir / "umap_target_vs_background.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig1)

    if len(moa_df) > 0:
        fig2 = plot_moa_context_umap(_umap_adata, _target_jcps, moa_df, target)
        fig2.savefig(output_dir / "umap_moa_context.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig2)

        fig3 = plot_per_moa_umap(_umap_adata, moa_df, target_df, target)
        if fig3 is not None:
            fig3.savefig(output_dir / "umap_per_moa.png", dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig3)

    fig4 = plot_similarity_heatmap(target_df)
    if fig4 is not None:
        fig4.savefig(output_dir / "similarity_heatmap.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig4)

    # Step 6: Compute MoA counts
    moa_counts = {}
    if len(moa_df) > 0:
        for _moa_str in target_df["Metadata_repurposing_moa"].dropna():
            for _moa in _moa_str.split("|"):
                _moa = _moa.strip()
                if _moa and _moa not in moa_counts:
                    _n_total = len(
                        moa_df[moa_df["Metadata_repurposing_moa"].fillna("").str.contains(_moa, regex=False)]
                    )
                    _n_target = len(
                        target_df[target_df["Metadata_repurposing_moa"].fillna("").str.contains(_moa, regex=False)]
                    )
                    moa_counts[_moa] = {
                        "target_compounds": _n_target,
                        "total_compounds": _n_total,
                    }

    # Step 7: Save summary JSON
    _n_active = int(target_df["active"].sum()) if "active" in target_df.columns else 0
    _n_profiled = int(target_df["nmAP"].notna().sum()) if "nmAP" in target_df.columns else 0
    _n_selective = int(target_df["target_selective"].sum())
    _n_same_moa = len(moa_df[~moa_df["is_target_compound"]]) if len(moa_df) > 0 else 0

    _compound_cols = [
        "Metadata_JCP2022",
        "Metadata_repurposing_name",
        "Metadata_repurposing_target",
        "Metadata_repurposing_moa",
        "annotation_sources",
        "target_selective",
        "nmAP",
        "fdr_p",
        "active",
    ]
    _available_cols = [c for c in _compound_cols if c in target_df.columns]

    summary = {
        "target_gene": target,
        "total_compounds": len(target_df),
        "target_selective": _n_selective,
        "multi_target": int(len(target_df) - _n_selective),
        "activity": {
            "profiled": _n_profiled,
            "active": _n_active,
            "missing_profiles": int(len(target_df) - _n_profiled),
        },
        "moa_context": {
            "shared_moa_compounds": _n_same_moa,
            "moa_breakdown": moa_counts,
        },
        "compounds": target_df[_available_cols].to_dict(orient="records"),
    }

    Path(output_dir, "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"MMP9 analysis complete: {output_dir}")
    return str(output_dir)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
