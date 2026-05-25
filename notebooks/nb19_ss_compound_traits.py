# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb==1.5.3",
#     "loguru==0.7.3",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "python-dotenv",
#     "scanpy==1.12.1",
#     "scipy==1.17.1",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    from dataclasses import dataclass
    from pathlib import Path

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from loguru import logger
    from matplotlib.patches import Patch

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import (
        COPAIRS_RESULTS_DB,
        DEFAULT_DPI,
        METADATA_DB,
        PROCESSED_DATA_DIR,
    )
    from nb02_ss_queries import query_activity_results
    from nb04_ss_visualization import (
        MAP_COL,
        NONSIG_COLOR,
        SIG_COL,
        SIG_COLOR,
        compute_partial_correlation,
        plot_categorical_box,
        plot_scatter_with_marginals,
    )

    @dataclass
    class TraitConfig:
        """Configuration for a compound trait in scatter plots."""

        col: str
        label: str
        xlim: tuple[float, float]
        panel_label: str = ""

    CONTINUOUS_TRAITS = [
        TraitConfig("Metadata_LogP", "LogP", (-5, 10), "A. LogP"),
        TraitConfig("Metadata_QED", "QED (Drug-likeness)", (0, 1), "B. QED"),
        TraitConfig("Metadata_MW", "Molecular Weight (Da)", (0, 800), "C. Molecular Weight"),
    ]

    EXTRA_CORRELATION_TRAITS = [
        TraitConfig("Metadata_TPSA", "TPSA", (0, 200), ""),
    ]

    OUTPUT_DIR = PROCESSED_DATA_DIR / "compound-traits"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compound Traits vs Phenotypic Activity

    Which intrinsic compound properties predict higher phenotypic activity?

    **Continuous properties:** LogP, QED, Molecular Weight (scatter + marginal histograms)

    **Categorical properties:** Lipinski violations, PAINS alerts (box plots)

    **Summary statistics:** Spearman correlations, partial correlations controlling for
    confounders, inter-predictor correlations, and activity rates by category.

    Note: Consistency analysis is NOT included here because consistency is a property
    of (compound, target) pairs, not compounds alone. Target-level consistency analysis
    belongs in a separate notebook.

    *Outputs:* `data/processed/compound-traits/{dataset}/{preprocessing}/`
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
    mo.hstack(
        [dataset_dropdown, preprocessing_dropdown, filter_dropdown, activity_params_dropdown],
        justify="start",
    )
    return (
        activity_params_dropdown,
        dataset_dropdown,
        filter_dropdown,
        preprocessing_dropdown,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@app.function
def load_compound_properties() -> pd.DataFrame:
    """Load compound properties from metadata database."""
    _con = duckdb.connect(str(METADATA_DB), read_only=True)
    _query = """
    SELECT
        Metadata_JCP2022,
        Metadata_SMILES,
        Metadata_MW,
        Metadata_LogP,
        Metadata_TPSA,
        Metadata_QED,
        Metadata_HBD,
        Metadata_HBA,
        Metadata_RotatableBonds,
        Metadata_NumRings,
        Metadata_Lipinski_Violations,
        Metadata_HasPAINS,
        Metadata_MurckoScaffold,
        CASE WHEN Metadata_repurposing_target IS NOT NULL THEN 1 ELSE 0 END as has_repurposing_target,
        CASE WHEN Metadata_Uniprot_target IS NOT NULL THEN 1 ELSE 0 END as has_uniprot_target,
        CASE WHEN Metadata_chmprb_target_genes IS NOT NULL THEN 1 ELSE 0 END as has_probe_target
    FROM compound_metadata
    WHERE Metadata_SMILES IS NOT NULL
    """
    _df = _con.execute(_query).fetchdf()
    _con.close()
    return _df


@app.function
def load_activity_with_properties(
    dataset: str,
    preprocessing: str,
    filter_name: str,
    activity_params: str = "default",
) -> pd.DataFrame:
    """Load activity results joined with compound properties."""
    _activity_df = query_activity_results(dataset, preprocessing, filter_name, activity_params)
    logger.info(f"Loaded {len(_activity_df):,} activity results")

    _props_df = load_compound_properties()
    logger.info(f"Loaded {len(_props_df):,} compounds with properties")

    _merged = _activity_df.merge(_props_df, on="Metadata_JCP2022", how="left")
    logger.info(f"Merged: {len(_merged):,} rows with properties")
    return _merged


@app.cell
def _(
    activity_params_dropdown,
    dataset_dropdown,
    filter_dropdown,
    mo,
    preprocessing_dropdown,
):
    mo.stop(
        not COPAIRS_RESULTS_DB.exists(),
        mo.md(f"**Copairs database not found:** `{COPAIRS_RESULTS_DB}`"),
    )
    mo.stop(
        not METADATA_DB.exists(),
        mo.md(f"**Metadata database not found:** `{METADATA_DB}`"),
    )

    df = load_activity_with_properties(
        dataset_dropdown.value,
        preprocessing_dropdown.value,
        filter_dropdown.value,
        activity_params_dropdown.value,
    )

    _n_active = df[SIG_COL].sum()
    mo.md(f"""
    ## Data loaded

    **Dataset:** {dataset_dropdown.value} | **Preprocessing:** {preprocessing_dropdown.value}
    | **Filter:** {filter_dropdown.value} | **Activity params:** {activity_params_dropdown.value}

    **Total compounds:** {len(df):,} | **Active:** {_n_active:,} ({100 * _n_active / len(df):.1f}%)
    """)
    return (df,)


# ---------------------------------------------------------------------------
# Individual scatter plots (continuous traits)
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Continuous traits vs activity

    Scatter plots with marginal histograms showing the distribution shift
    between significant (blue) and non-significant (gray) compounds.
    """)
    return


@app.cell
def _(df):
    _trait = CONTINUOUS_TRAITS[0]
    fig_logp, _stats_logp = plot_scatter_with_marginals(
        df,
        x_col=_trait.col,
        x_label=_trait.label,
        x_lim=_trait.xlim,
        title=f"{_trait.label} vs Phenotypic Activity",
    )
    fig_logp
    return (fig_logp,)


@app.cell
def _(df):
    _trait = CONTINUOUS_TRAITS[1]
    fig_qed, _stats_qed = plot_scatter_with_marginals(
        df,
        x_col=_trait.col,
        x_label=_trait.label,
        x_lim=_trait.xlim,
        title=f"{_trait.label} vs Phenotypic Activity",
    )
    fig_qed
    return (fig_qed,)


@app.cell
def _(df):
    _trait = CONTINUOUS_TRAITS[2]
    fig_mw, _stats_mw = plot_scatter_with_marginals(
        df,
        x_col=_trait.col,
        x_label=_trait.label,
        x_lim=_trait.xlim,
        title=f"{_trait.label} vs Phenotypic Activity",
    )
    fig_mw
    return (fig_mw,)


# ---------------------------------------------------------------------------
# Categorical traits
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Categorical traits vs activity

    Box plots of normalized mAP by Lipinski violation count and PAINS alert status.
    """)
    return


@app.cell
def _(df):
    _valid_lip = df[df["Metadata_Lipinski_Violations"] <= 3].copy()
    fig_lipinski = plot_categorical_box(
        _valid_lip,
        cat_col="Metadata_Lipinski_Violations",
        order=[0, 1, 2, 3],
        xlabel="Lipinski Violations",
    )
    fig_lipinski
    return (fig_lipinski,)


@app.cell
def _(df):
    _df_pains = df.copy()
    _df_pains["PAINS"] = _df_pains["Metadata_HasPAINS"].map({False: "Clean", True: "PAINS"})
    fig_pains = plot_categorical_box(
        _df_pains,
        cat_col="PAINS",
        order=["Clean", "PAINS"],
        colors={"Clean": "#3498db", "PAINS": "#e74c3c"},
    )
    fig_pains
    return (fig_pains,)


# ---------------------------------------------------------------------------
# Combined publication figure
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Combined publication figure

    Two-row layout:
    - **Row 1** (taller): A. LogP, B. QED, C. Molecular Weight (scatter + marginals)
    - **Row 2** (shorter): D. Lipinski Violations, E. PAINS Alerts, legend panel
    """)
    return


@app.function
def plot_combined_figure(df: pd.DataFrame) -> plt.Figure:
    """Create combined publication figure with intrinsic compound traits."""
    fig = plt.figure(figsize=(14, 10))

    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.25, wspace=0.25)

    # Row 1: Scatter plots with marginals
    for _col_idx, _trait in enumerate(CONTINUOUS_TRAITS):
        plot_scatter_with_marginals(
            df,
            x_col=_trait.col,
            x_label=_trait.label,
            x_lim=_trait.xlim,
            panel_label=_trait.panel_label,
            fig=fig,
            gs_parent=gs[0, _col_idx],
        )

    # Row 2: Categorical box plots

    # D. Lipinski violations
    _ax_d = fig.add_subplot(gs[1, 0])
    _valid_lip = df[df["Metadata_Lipinski_Violations"] <= 3].copy()
    plot_categorical_box(
        _valid_lip,
        cat_col="Metadata_Lipinski_Violations",
        order=[0, 1, 2, 3],
        xlabel="Lipinski Violations",
        ax=_ax_d,
    )
    _ax_d.set_title("D. Lipinski Violations", fontsize=10, fontweight="bold", loc="left")

    # E. PAINS alerts
    _ax_e = fig.add_subplot(gs[1, 1])
    if "Metadata_HasPAINS" in df.columns:
        _df_pains = df.copy()
        _df_pains["PAINS"] = _df_pains["Metadata_HasPAINS"].map({False: "Clean", True: "PAINS"})
        plot_categorical_box(
            _df_pains,
            cat_col="PAINS",
            order=["Clean", "PAINS"],
            colors={"Clean": "#3498db", "PAINS": "#e74c3c"},
            ax=_ax_e,
        )
    _ax_e.set_title("E. PAINS Alerts", fontsize=10, fontweight="bold", loc="left")

    # Legend panel
    _ax_legend = fig.add_subplot(gs[1, 2])
    _ax_legend.set_xlim(0, 1)
    _ax_legend.set_ylim(0, 1)
    _ax_legend.axis("off")

    _legend_elements = [
        Patch(facecolor=SIG_COLOR, alpha=0.7, label="Significant (p < 0.05, corrected)"),
        Patch(facecolor=NONSIG_COLOR, alpha=0.5, label="Not significant"),
    ]
    _ax_legend.legend(handles=_legend_elements, loc="center", fontsize=9, frameon=False)
    _ax_legend.text(
        0.5,
        0.25,
        f"n = {len(df):,} compounds\n{df[SIG_COL].sum():,} significant ({100 * df[SIG_COL].mean():.1f}%)",
        ha="center",
        va="center",
        fontsize=9,
        transform=_ax_legend.transAxes,
    )

    fig.suptitle(
        "Compound Traits vs Phenotypic Activity",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    return fig


@app.cell
def _(df):
    fig_combined = plot_combined_figure(df)
    fig_combined
    return (fig_combined,)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary statistics

    Spearman correlations, partial correlations controlling for confounders,
    inter-predictor correlations, and activity rates by categorical trait.
    """)
    return


@app.function
def compute_correlation_stats(
    df: pd.DataFrame,
    property_col: str,
    activity_col: str = MAP_COL,
) -> dict:
    """Compute Spearman correlation between property and activity."""
    from scipy import stats as scipy_stats

    _valid = df[[property_col, activity_col]].dropna()
    if len(_valid) < 10:
        return {"spearman_r": np.nan, "n": len(_valid)}

    _r, _ = scipy_stats.spearmanr(_valid[property_col], _valid[activity_col])
    return {"spearman_r": _r, "n": len(_valid)}


@app.function
def compute_confounding_analysis(df: pd.DataFrame) -> dict:
    """Analyze confounding between LogP, MW, and QED.

    Returns inter-predictor correlations and partial correlations
    controlling for confounders.
    """
    from scipy import stats as scipy_stats

    _results = {
        "inter_predictor_correlations": {},
        "partial_correlations": {},
        "lipinski_by_properties": {},
    }

    # Inter-predictor correlations
    _predictors = [
        ("Metadata_LogP", "Metadata_MW"),
        ("Metadata_LogP", "Metadata_QED"),
        ("Metadata_MW", "Metadata_QED"),
    ]
    for _col1, _col2 in _predictors:
        _valid = df[[_col1, _col2]].dropna()
        _r, _ = scipy_stats.spearmanr(_valid[_col1], _valid[_col2])
        _key = f"{_col1.replace('Metadata_', '').lower()}_vs_{_col2.replace('Metadata_', '').lower()}"
        _results["inter_predictor_correlations"][_key] = round(_r, 3)

    # Partial correlations
    _pc = compute_partial_correlation(df, "Metadata_LogP", MAP_COL, ["Metadata_MW"])
    _results["partial_correlations"]["logp_controlling_mw"] = (
        round(_pc["partial_r"], 3) if not np.isnan(_pc["partial_r"]) else None
    )

    _pc = compute_partial_correlation(df, "Metadata_MW", MAP_COL, ["Metadata_LogP"])
    _results["partial_correlations"]["mw_controlling_logp"] = (
        round(_pc["partial_r"], 3) if not np.isnan(_pc["partial_r"]) else None
    )

    _pc = compute_partial_correlation(df, "Metadata_QED", MAP_COL, ["Metadata_LogP", "Metadata_MW"])
    _results["partial_correlations"]["qed_controlling_logp_mw"] = (
        round(_pc["partial_r"], 3) if not np.isnan(_pc["partial_r"]) else None
    )

    # Mean properties by Lipinski violations
    for _lip in [0, 1, 2, 3]:
        _subset = df[df["Metadata_Lipinski_Violations"] == _lip]
        if len(_subset) > 0:
            _results["lipinski_by_properties"][str(_lip)] = {
                "n": len(_subset),
                "mean_logp": round(_subset["Metadata_LogP"].mean(), 2),
                "mean_mw": round(_subset["Metadata_MW"].mean(), 0),
                "mean_qed": round(_subset["Metadata_QED"].mean(), 2),
            }

    return _results


@app.function
def generate_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for all trait-activity analyses."""
    _n_total = len(df)
    _n_active = df[SIG_COL].sum()

    _summary = {
        "dataset_stats": {
            "n_compounds": _n_total,
            "n_active": int(_n_active),
            "activity_rate_pct": round(100 * _n_active / _n_total, 2),
        },
        "continuous_correlations": {},
        "categorical_stats": {},
        "confounding_analysis": {},
        "statistical_notes": {
            "p_values": "With n=112k, all p-values are at machine precision (<1e-100). Effect sizes (rho) are the relevant metric.",
            "partial_correlations": "Computed via residualization to disentangle confounded predictors",
        },
    }

    # Continuous property correlations
    _all_traits = CONTINUOUS_TRAITS + EXTRA_CORRELATION_TRAITS
    for _trait in _all_traits:
        if _trait.col in df.columns:
            _stats_dict = compute_correlation_stats(df, _trait.col)
            _name = _trait.col.replace("Metadata_", "").lower()
            _summary["continuous_correlations"][_name] = {
                "spearman_r": round(_stats_dict["spearman_r"], 3) if not np.isnan(_stats_dict["spearman_r"]) else None,
                "n": _stats_dict["n"],
            }

    # Confounding analysis
    logger.info("Computing confounding analysis...")
    _summary["confounding_analysis"] = compute_confounding_analysis(df)

    # Lipinski violations
    if "Metadata_Lipinski_Violations" in df.columns:
        _lip_stats = df.groupby("Metadata_Lipinski_Violations")[SIG_COL].agg(["count", "sum", "mean"])
        _summary["categorical_stats"]["lipinski_violations"] = {
            int(_v): {
                "n": int(_lip_stats.loc[_v, "count"]) if _v in _lip_stats.index else 0,
                "activity_rate": round(_lip_stats.loc[_v, "mean"] * 100, 2) if _v in _lip_stats.index else 0,
            }
            for _v in range(5)
        }

    # PAINS
    if "Metadata_HasPAINS" in df.columns:
        _pains_stats = df.groupby("Metadata_HasPAINS")[SIG_COL].agg(["count", "sum", "mean"])
        _summary["categorical_stats"]["pains"] = {
            "clean": {
                "n": int(_pains_stats.loc[False, "count"]) if False in _pains_stats.index else 0,
                "activity_rate": round(_pains_stats.loc[False, "mean"] * 100, 2) if False in _pains_stats.index else 0,
            },
            "pains_flagged": {
                "n": int(_pains_stats.loc[True, "count"]) if True in _pains_stats.index else 0,
                "activity_rate": round(_pains_stats.loc[True, "mean"] * 100, 2) if True in _pains_stats.index else 0,
            },
        }

    return _summary


@app.cell
def _(df, mo):
    summary = generate_summary(df)

    # Format correlation table
    _corr_rows = []
    for _name, _vals in summary["continuous_correlations"].items():
        _corr_rows.append(f"| {_name} | {_vals['spearman_r']} | {_vals['n']:,} |")
    _corr_table = "\n".join(_corr_rows)

    # Format partial correlations
    _partial = summary["confounding_analysis"].get("partial_correlations", {})
    _partial_rows = []
    for _key, _val in _partial.items():
        _partial_rows.append(f"| {_key} | {_val} |")
    _partial_table = "\n".join(_partial_rows)

    # Format inter-predictor correlations
    _inter = summary["confounding_analysis"].get("inter_predictor_correlations", {})
    _inter_rows = []
    for _key, _val in _inter.items():
        _inter_rows.append(f"| {_key} | {_val} |")
    _inter_table = "\n".join(_inter_rows)

    mo.md(f"""
    ### Spearman correlations (trait vs normalized mAP)

    | Trait | Spearman rho | n |
    |-------|-------------|---|
    {_corr_table}

    ### Partial correlations (controlling for confounders)

    | Comparison | Partial rho |
    |-----------|------------|
    {_partial_table}

    ### Inter-predictor correlations

    | Pair | Spearman rho |
    |------|-------------|
    {_inter_table}

    **Note:** {summary["statistical_notes"]["p_values"]}
    """)
    return (summary,)


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save outputs

    Saves the combined publication figure and summary JSON to
    `data/processed/compound-traits/{dataset}/{preprocessing}/`.
    """)
    return


@app.cell
def _(
    dataset_dropdown,
    df,
    fig_combined,
    fig_lipinski,
    fig_logp,
    fig_mw,
    fig_pains,
    fig_qed,
    mo,
    preprocessing_dropdown,
    summary,
):
    _output_dir = OUTPUT_DIR / dataset_dropdown.value / preprocessing_dropdown.value
    _output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined figure
    fig_combined.savefig(_output_dir / "compound_traits_combined.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    logger.info(f"Saved combined figure to {_output_dir / 'compound_traits_combined.png'}")

    # Save individual scatter plots
    fig_logp.savefig(_output_dir / "scatter_logp_marginals.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_qed.savefig(_output_dir / "scatter_qed_marginals.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_mw.savefig(_output_dir / "scatter_mw_marginals.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    # Save categorical box plots
    fig_lipinski.savefig(_output_dir / "lipinski_box.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_pains.savefig(_output_dir / "pains_box.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    # Save summary JSON
    _summary_path = _output_dir / "compound_traits_summary.json"
    with open(_summary_path, "w") as _f:
        json.dump(summary, _f, indent=2)
    logger.info(f"Saved summary to {_summary_path}")

    mo.md(f"""
    **Saved outputs to** `{_output_dir}`

    - `compound_traits_combined.png` - combined publication figure
    - `scatter_logp_marginals.png`, `scatter_qed_marginals.png`, `scatter_mw_marginals.png`
    - `lipinski_box.png`, `pains_box.png`
    - `compound_traits_summary.json` - correlation and categorical statistics
    """)
    return


@app.function
def run_compound_traits(
    dataset: str,
    preprocessing: str,
    filter_name: str = "all_sources",
    activity_params: str = "default",
    output_dir: str | Path | None = None,
) -> str:
    """Run compound traits vs phenotypic activity analysis.

    Loads activity results joined with compound properties, generates the
    combined publication figure, computes summary statistics (Spearman
    correlations, partial correlations, categorical stats), and saves
    everything plus a .complete marker.

    Parameters
    ----------
    dataset : str
        e.g. "compound_no_source7"
    preprocessing : str
        e.g. "activity_no_target2"
    filter_name : str
        e.g. "all_sources"
    activity_params : str
        e.g. "default", "withinsource", "crosssource"
    output_dir : str | Path | None
        Output directory. Defaults to
        PROCESSED_DATA_DIR / "compound-traits" / dataset / preprocessing

    Returns
    -------
    str
        Path to the output directory.
    """
    if output_dir is None:
        _out = OUTPUT_DIR / dataset / preprocessing
    else:
        _out = Path(output_dir)
    _out.mkdir(parents=True, exist_ok=True)

    # Load data
    _df = load_activity_with_properties(dataset, preprocessing, filter_name, activity_params)
    logger.info(f"Loaded {len(_df):,} compounds for compound traits analysis")

    # Combined publication figure
    _fig_combined = plot_combined_figure(_df)
    _fig_combined.savefig(
        _out / "compound_traits_combined.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white"
    )
    plt.close(_fig_combined)

    # Individual scatter plots
    for _trait in CONTINUOUS_TRAITS:
        _fig, _stats = plot_scatter_with_marginals(
            _df,
            x_col=_trait.col,
            x_label=_trait.label,
            x_lim=_trait.xlim,
            title=f"{_trait.label} vs Phenotypic Activity",
        )
        _fname = f"scatter_{_trait.col.replace('Metadata_', '').lower()}_marginals.png"
        _fig.savefig(_out / _fname, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
        plt.close(_fig)

    # Categorical box plots
    if "Metadata_Lipinski_Violations" in _df.columns:
        _valid_lip = _df[_df["Metadata_Lipinski_Violations"] <= 3].copy()
        _fig_lip = plot_categorical_box(
            _valid_lip,
            cat_col="Metadata_Lipinski_Violations",
            order=[0, 1, 2, 3],
            xlabel="Lipinski Violations",
        )
        _fig_lip.savefig(_out / "lipinski_box.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
        plt.close(_fig_lip)

    if "Metadata_HasPAINS" in _df.columns:
        _df_pains = _df.copy()
        _df_pains["PAINS"] = _df_pains["Metadata_HasPAINS"].map({False: "Clean", True: "PAINS"})
        _fig_pains = plot_categorical_box(
            _df_pains,
            cat_col="PAINS",
            order=["Clean", "PAINS"],
            colors={"Clean": "#3498db", "PAINS": "#e74c3c"},
        )
        _fig_pains.savefig(_out / "pains_box.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
        plt.close(_fig_pains)

    # Summary JSON
    _summary = generate_summary(_df)
    _summary_path = _out / "compound_traits_summary.json"
    with open(_summary_path, "w") as _f:
        json.dump(_summary, _f, indent=2)
    logger.info(f"Saved summary to {_summary_path}")

    # .complete marker
    (_out / ".complete").touch()

    logger.success(f"run_compound_traits: saved all outputs to {_out}")
    return str(_out)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
