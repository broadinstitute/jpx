# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb==1.5.2",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
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

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    from nb00_ss_config import COPAIRS_RESULTS_DB


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cosine vs Absolute Cosine Distance

    JUMP consistency analysis computes how well compounds sharing the same annotation
    (e.g., MOA or disease area) cluster in morphological space using mean Average Precision (nMAP).
    The distance metric used to measure profile similarity matters - cosine distance preserves
    sign information (opposing morphological effects appear dissimilar), while absolute cosine
    collapses sign and only measures magnitude of effect.

    This notebook compares the two metrics side by side.
    Points above the diagonal gained nMAP by switching to absolute cosine;
    points below lost it. Color encodes whether statistical significance changed.

    **Verdict:** cosine is the better default. Most groups lose nMAP under absolute cosine,
    and very few gain significance. The sign of morphological perturbation carries real signal.

    *Related: [GitHub Issue #22](https://github.com/broadinstitute/jpx/issues/22)*
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
    dataset_dropdown
    return (dataset_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method

    Both metrics are computed by copairs-runner across all consistency target sources.
    The query below pulls `disease_area` and `moa` group types, excluding sweep runs,
    and joins the cosine and absolute cosine results on matching group/preprocessing/dataset keys.
    """)
    return


@app.function
def query_cosine_comparison(dataset: str):
    """Query cosine vs abs_cosine consistency results for a dataset."""
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    query = """
    WITH cosine_results AS (
        SELECT
            _group_type,
            _preprocessing,
            _dataset,
            group_value,
            mean_normalized_average_precision as cosine_nmap,
            corrected_p_value as cosine_pval,
            n_perturbations
        FROM consistency_results
        WHERE _distance = 'cosine'
          AND _dataset = ?
          AND _group_type IN ('disease_area', 'moa')
          AND _preprocessing NOT LIKE '%_sweep'
    ),
    abs_cosine_results AS (
        SELECT
            _group_type,
            _preprocessing,
            _dataset,
            group_value,
            mean_normalized_average_precision as abs_cosine_nmap,
            corrected_p_value as abs_cosine_pval
        FROM consistency_results
        WHERE _distance = 'abs_cosine'
          AND _dataset = ?
          AND _group_type IN ('disease_area', 'moa')
          AND _preprocessing NOT LIKE '%_sweep'
    )
    SELECT
        c._group_type,
        c._preprocessing,
        c.group_value,
        c.n_perturbations as n,
        c.cosine_nmap,
        a.abs_cosine_nmap,
        c.cosine_pval < 0.05 as cosine_sig,
        a.abs_cosine_pval < 0.05 as abs_cosine_sig
    FROM cosine_results c
    JOIN abs_cosine_results a
        ON c._group_type = a._group_type
        AND c._preprocessing = a._preprocessing
        AND c._dataset = a._dataset
        AND c.group_value = a.group_value
    """
    df = con.execute(query, [dataset, dataset]).fetchdf()
    con.close()
    return df


@app.cell
def _(dataset_dropdown):
    df = query_cosine_comparison(dataset_dropdown.value)
    return (df,)


@app.function
def plot_cosine_comparison(df, dataset: str):
    """Scatter plot comparing cosine vs abs_cosine nMAP, colored by significance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, group_type in zip(axes, ["disease_area", "moa"]):
        subset = df[df["_group_type"] == group_type]

        colors = []
        for _, row in subset.iterrows():
            if row["cosine_sig"] and row["abs_cosine_sig"]:
                colors.append("#2ecc71")
            elif row["cosine_sig"] and not row["abs_cosine_sig"]:
                colors.append("#e74c3c")
            elif not row["cosine_sig"] and row["abs_cosine_sig"]:
                colors.append("#3498db")
            else:
                colors.append("#95a5a6")

        sizes = np.clip(subset["n"] * 5, 20, 200)

        ax.scatter(
            subset["cosine_nmap"],
            subset["abs_cosine_nmap"],
            c=colors,
            s=sizes,
            alpha=0.6,
            edgecolors="white",
            linewidth=0.5,
        )

        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.3, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel("Cosine nMAP")
        ax.set_ylabel("Abs Cosine nMAP")
        ax.set_title(f"{group_type.replace('_', ' ').title()}\n(n={len(subset)} groups)")
        ax.set_aspect("equal")

        corr = np.corrcoef(subset["cosine_nmap"], subset["abs_cosine_nmap"])[0, 1]
        ax.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="Both sig"),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="Lost sig (cosine->abs)"
        ),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="Gained sig"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6", markersize=10, label="Neither sig"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle("Cosine vs Absolute Cosine: Consistency nMAP Comparison", y=1.02)
    plt.tight_layout()
    return fig


@app.cell
def _(dataset_dropdown, df):
    fig = plot_cosine_comparison(df, dataset_dropdown.value)
    fig
    return (fig,)


@app.cell
def _(dataset_dropdown, fig, mo):
    outdir = Path("data/processed/exploration/0.01") / dataset_dropdown.value
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "cosine_vs_abs_cosine_scatter.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    mo.md(f"Saved: `{outpath}`")
    return


@app.cell
def _(df, mo):
    lines = ["## Summary\n"]
    for group_type in ["disease_area", "moa"]:
        subset = df[df["_group_type"] == group_type]
        diff = subset["abs_cosine_nmap"] - subset["cosine_nmap"]

        both_sig = ((subset["cosine_sig"]) & (subset["abs_cosine_sig"])).sum()
        lost_sig = ((subset["cosine_sig"]) & (~subset["abs_cosine_sig"])).sum()
        gained_sig = ((~subset["cosine_sig"]) & (subset["abs_cosine_sig"])).sum()

        lines.append(f"### {group_type.replace('_', ' ').title()}\n")
        lines.append(f"- **N groups:** {len(subset)}")
        lines.append(f"- **Mean diff (abs - cos):** {diff.mean():.4f}")
        lines.append(f"- **Groups improved:** {(diff > 0).sum()} ({100 * (diff > 0).mean():.1f}%)")
        lines.append(f"- **Groups worse:** {(diff < 0).sum()} ({100 * (diff < 0).mean():.1f}%)")
        lines.append(f"- **Significance:** both={both_sig}, lost={lost_sig}, gained={gained_sig}\n")

    mo.md("\n".join(lines))
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
