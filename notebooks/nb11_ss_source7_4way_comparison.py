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
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from collections import OrderedDict
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from loguru import logger
    from matplotlib.lines import Line2D

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import DEFAULT_DPI, PROCESSED_DATA_DIR
    from nb02_ss_queries import query_activity_results, query_consistency_results

    # Activity uses original datasets (per-compound, no shared pool needed)
    ACTIVITY_CONFIGS = OrderedDict(
        [
            ("CP_no_s7", "compound_no_source7"),
            ("CP_with_s7", "compound_with_source7"),
            ("DL_no_s7", "compound_DL_CPCNN_no_source7"),
            ("DL_with_s7", "compound_DL_CPCNN_with_source7"),
        ]
    )

    # Consistency uses active_union datasets (shared pool of actives from CP|DL, per issue #21)
    CONSISTENCY_CONFIGS = OrderedDict(
        [
            ("CP_no_s7", "compound_no_source7_active_union"),
            ("CP_with_s7", "compound_with_source7_active_union"),
            ("DL_no_s7", "compound_DL_CPCNN_no_source7_active_union"),
            ("DL_with_s7", "compound_DL_CPCNN_with_source7_active_union"),
        ]
    )

    TARGETS = [
        "repurposing",
        "uniprot",
        "chemical_probes",
        "moa",
        "disease_area",
        "motive_biokg",
        "motive_opentargets",
        "motive_primekg",
        "toxcast_assay",
    ]

    # Colors: CP = blue tones, DL = orange tones; lighter = no_s7, darker = with_s7
    CONFIG_COLORS = {
        "CP_no_s7": "#6baed6",
        "CP_with_s7": "#2171b5",
        "DL_no_s7": "#fdae6b",
        "DL_with_s7": "#e6550d",
    }

    # Markers: circle = no_s7, triangle = with_s7
    CONFIG_MARKERS = {
        "CP_no_s7": "o",
        "CP_with_s7": "^",
        "DL_no_s7": "o",
        "DL_with_s7": "^",
    }

    OUTPUT_DIR = PROCESSED_DATA_DIR / "source7-comparison"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Source 7 4-Way Comparison

    Compares activity and consistency results across 4 configurations:

    - **CP no_s7**: compound_no_source7
    - **CP with_s7**: compound_with_source7
    - **DL no_s7**: compound_DL_CPCNN_no_source7
    - **DL with_s7**: compound_DL_CPCNN_with_source7

    Source 7 contributes 3,191 compounds at 0.625 uM (vs 5 uM for other sources).
    Consistency uses union-based evaluation (issue #21): both methods evaluated on
    the shared pool of perturbations active in either CP or DL.

    Produces:
    1. Activity grouped bar chart
    2. Consistency heatmaps (% significant, mean nMAP)
    3. Delta significant targets bar chart
    4. Per-target 4-way dot plots with status change CSVs
    5. Signal dilution analysis
    6. Head-to-head CP vs DL comparison
    """)
    return


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    top_n_slider = mo.ui.slider(
        start=10,
        stop=50,
        step=5,
        value=30,
        label="Top N targets in dot plots",
    )
    top_n_slider
    return (top_n_slider,)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


@app.function
def get_activity_summary() -> pd.DataFrame:
    """Query activity results for all 4 configs and return summary rows."""
    _rows = []
    for _label, _dataset in ACTIVITY_CONFIGS.items():
        _df = query_activity_results(
            _dataset,
            preprocessing="activity_no_target2",
            filter_name="all_sources",
            activity_params="default",
        )
        if len(_df) == 0:
            logger.warning(f"No activity results for dataset '{_dataset}'")
        _n_total = len(_df)
        _n_active = int(_df["below_corrected_p"].sum()) if _n_total else 0
        _pct = 100 * _n_active / _n_total if _n_total else 0
        _mean_nmap = _df["mean_normalized_average_precision"].mean() if _n_total else 0
        _rows.append(
            {
                "config": _label,
                "dataset": _dataset,
                "n_compounds": _n_total,
                "n_active": _n_active,
                "pct_active": round(_pct, 2),
                "mean_nmap": round(_mean_nmap, 4),
            }
        )
        logger.info(f"  {_label}: {_n_total:,} compounds, {_n_active:,} active ({_pct:.1f}%)")
    return pd.DataFrame(_rows)


@app.function
def get_consistency_summary() -> pd.DataFrame:
    """Query consistency results for all 4 configs x 9 targets (union-based)."""
    _rows = []
    for _label, _dataset in CONSISTENCY_CONFIGS.items():
        for _target in TARGETS:
            _df = query_consistency_results(
                _dataset,
                preprocessing="consistency_prefiltered",
                filter_name="all_sources",
                group_type=_target,
                distance="cosine",
            )
            _n_total = len(_df)
            _n_sig = int(_df["below_corrected_p"].sum()) if _n_total else 0
            _pct = 100 * _n_sig / _n_total if _n_total else 0
            _mean_nmap = _df["mean_normalized_average_precision"].mean() if _n_total else 0
            _rows.append(
                {
                    "config": _label,
                    "dataset": _dataset,
                    "target": _target,
                    "n_targets": _n_total,
                    "n_significant": _n_sig,
                    "pct_significant": round(_pct, 1),
                    "mean_nmap": round(_mean_nmap, 4),
                }
            )
    return pd.DataFrame(_rows)


@app.function
def get_per_target_results(target: str) -> dict[str, pd.DataFrame]:
    """Query per-target consistency results for all 4 configs (union-based)."""
    _results = {}
    for _label, _dataset in CONSISTENCY_CONFIGS.items():
        _df = query_consistency_results(
            _dataset,
            preprocessing="consistency_prefiltered",
            filter_name="all_sources",
            group_type=target,
            distance="cosine",
        )
        _results[_label] = _df
    return _results


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------


@app.function
def classify_status(row: pd.Series, profile: str) -> str:
    """Classify a target's status change between no_s7 and with_s7.

    Categories:
    - stable_sig: significant in both
    - stable_nonsig: not significant in both
    - gained: not sig in no_s7, sig in with_s7
    - lost: sig in no_s7, not sig in with_s7
    - new_sig: only evaluable in with_s7 and significant
    - new_nonsig: only evaluable in with_s7 and not significant
    - dropped: only evaluable in no_s7 (target lost)
    """
    _no_s7 = f"{profile}_no_s7"
    _with_s7 = f"{profile}_with_s7"
    _sig_no = row[f"sig_{_no_s7}"]
    _sig_with = row[f"sig_{_with_s7}"]

    _has_no = pd.notna(row[f"nmap_{_no_s7}"])
    _has_with = pd.notna(row[f"nmap_{_with_s7}"])

    # After outer join, sig columns can be NaN - coerce to bool safely
    _sig_no = bool(_sig_no) if pd.notna(_sig_no) else False
    _sig_with = bool(_sig_with) if pd.notna(_sig_with) else False

    if _has_no and _has_with:
        if _sig_no and _sig_with:
            return "stable_sig"
        elif not _sig_no and not _sig_with:
            return "stable_nonsig"
        elif not _sig_no and _sig_with:
            return "gained"
        else:
            return "lost"
    elif not _has_no and _has_with:
        return "new_sig" if _sig_with else "new_nonsig"
    elif _has_no and not _has_with:
        return "dropped"
    else:
        return "absent"


@app.function
def build_status_changes(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """4-way outer join on group_value with status classification."""
    if not results or all(len(df) == 0 for df in results.values()):
        return pd.DataFrame()
    _merged = None
    for _config, _df in results.items():
        _subset = _df[
            ["group_value", "mean_normalized_average_precision", "below_corrected_p", "n_perturbations"]
        ].copy()
        _subset = _subset.rename(
            columns={
                "mean_normalized_average_precision": f"nmap_{_config}",
                "below_corrected_p": f"sig_{_config}",
                "n_perturbations": f"n_{_config}",
            }
        )
        if _merged is None:
            _merged = _subset
        else:
            _merged = _merged.merge(_subset, on="group_value", how="outer")

    # Classify status for CP and DL
    _merged["status_CP"] = _merged.apply(classify_status, axis=1, profile="CP")
    _merged["status_DL"] = _merged.apply(classify_status, axis=1, profile="DL")

    # Sort by max nMAP across configs
    _nmap_cols = [c for c in _merged.columns if c.startswith("nmap_")]
    _merged["_max_nmap"] = _merged[_nmap_cols].max(axis=1)
    _merged = _merged.sort_values("_max_nmap", ascending=False).drop(columns=["_max_nmap"])

    return _merged


# ---------------------------------------------------------------------------
# Plot functions (all return fig, never close)
# ---------------------------------------------------------------------------


@app.function
def plot_activity_grouped_bar(summary: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: % phenotypically active across 4 configs."""
    fig, ax = plt.subplots(figsize=(7, 5))

    _groups = ["CP", "DL"]
    _variants = ["no_s7", "with_s7"]
    _x = np.arange(len(_groups))
    _width = 0.3

    for _i, _variant in enumerate(_variants):
        _labels = [f"{g}_{_variant}" for g in _groups]
        _values = [summary.loc[summary.config == lbl, "pct_active"].values[0] for lbl in _labels]
        _n_vals = [summary.loc[summary.config == lbl, "n_active"].values[0] for lbl in _labels]
        _colors = [CONFIG_COLORS[lbl] for lbl in _labels]
        _offset = (_i - 0.5) * _width
        _bars = ax.bar(_x + _offset, _values, _width * 0.9, color=_colors, edgecolor="white", linewidth=0.5)

        for _bar, _pct, _n in zip(_bars, _values, _n_vals):
            ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                _bar.get_height() + 0.3,
                f"{_pct:.1f}%\nn={_n:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(_x)
    ax.set_xticklabels(_groups, fontsize=11)
    ax.set_ylabel("% Phenotypically Active (corrected p < 0.05)", fontsize=10)
    ax.set_title("Activity: Source 7 Impact on CP vs DL", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=CONFIG_COLORS["CP_no_s7"], label="CP no_s7"),
        plt.Rectangle((0, 0), 1, 1, fc=CONFIG_COLORS["CP_with_s7"], label="CP with_s7"),
        plt.Rectangle((0, 0), 1, 1, fc=CONFIG_COLORS["DL_no_s7"], label="DL no_s7"),
        plt.Rectangle((0, 0), 1, 1, fc=CONFIG_COLORS["DL_with_s7"], label="DL with_s7"),
    ]
    ax.legend(handles=_legend_elements, loc="upper right", framealpha=0.9, ncol=2)

    fig.tight_layout()
    return fig


@app.function
def plot_consistency_heatmap(
    summary: pd.DataFrame,
    value_col: str = "pct_significant",
    cell_fmt: str = "pct",
    cmap: str = "Blues",
    title: str = "Consistency: % Significant Targets",
    cbar_label: str = "% Significant",
) -> plt.Figure:
    """Heatmap with rows=targets, cols=configs."""
    _target_order = summary.groupby("target")[value_col].mean().sort_values(ascending=False).index.tolist()
    _config_order = list(CONFIG_COLORS.keys())

    _pivot = summary.pivot(index="target", columns="config", values=value_col)
    _pivot = _pivot.reindex(index=_target_order, columns=_config_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        _pivot,
        annot=True,
        fmt=".1f" if value_col != "mean_nmap" else ".3f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": cbar_label},
    )

    # Add N_sig/N_total annotations for pct heatmap
    if cell_fmt == "pct":
        for _i, _target in enumerate(_target_order):
            for _j, _config in enumerate(_config_order):
                _row = summary[(summary.target == _target) & (summary.config == _config)]
                if len(_row) == 1:
                    _n_sig = _row.iloc[0]["n_significant"]
                    _n_total = _row.iloc[0]["n_targets"]
                    ax.text(
                        _j + 0.5,
                        _i + 0.75,
                        f"{_n_sig}/{_n_total}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="gray",
                    )

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("")
    ax.set_xlabel("")

    fig.tight_layout()
    return fig


@app.function
def plot_delta_bar(summary: pd.DataFrame) -> plt.Figure:
    """Grouped bar: delta significant targets (with_s7 - no_s7) for CP and DL."""
    fig, ax = plt.subplots(figsize=(10, 5))

    _profiles = ["CP", "DL"]
    _x = np.arange(len(TARGETS))
    _width = 0.35

    _profile_colors = {"CP": CONFIG_COLORS["CP_with_s7"], "DL": CONFIG_COLORS["DL_with_s7"]}
    for _i, _profile in enumerate(_profiles):
        _no_s7 = f"{_profile}_no_s7"
        _with_s7 = f"{_profile}_with_s7"
        _deltas = []
        for _target in TARGETS:
            _n_no = summary.loc[(summary.config == _no_s7) & (summary.target == _target), "n_significant"]
            _n_with = summary.loc[(summary.config == _with_s7) & (summary.target == _target), "n_significant"]
            if len(_n_no) == 0 or len(_n_with) == 0:
                logger.warning(f"Missing data for {_profile}/{_target} in delta bar")
                _deltas.append(float("nan"))
            else:
                _deltas.append(int(_n_with.values[0]) - int(_n_no.values[0]))

        _offset = (_i - 0.5) * _width
        _bars = ax.bar(
            _x + _offset,
            _deltas,
            _width * 0.9,
            color=_profile_colors[_profile],
            edgecolor="white",
            alpha=0.8,
        )

        for _bar, _delta in zip(_bars, _deltas):
            if pd.notna(_delta) and _delta != 0:
                _va = "bottom" if _delta > 0 else "top"
                ax.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _delta,
                    f"{int(_delta):+d}",
                    ha="center",
                    va=_va,
                    fontsize=7,
                )

    ax.set_xticks(_x)
    ax.set_xticklabels(TARGETS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Delta Significant Targets (with_s7 - no_s7)", fontsize=10)
    ax.set_title("Impact of Source 7 on Target Consistency", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=CONFIG_COLORS["CP_with_s7"], label="CP"),
        plt.Rectangle((0, 0), 1, 1, fc=CONFIG_COLORS["DL_with_s7"], label="DL"),
    ]
    ax.legend(handles=_legend_elements, loc="upper right")

    fig.tight_layout()
    return fig


@app.function
def plot_4way_dotplot(
    merged: pd.DataFrame,
    target_source: str,
    top_n: int = 30,
) -> plt.Figure:
    """Cleveland dot plot with 4 markers per target row.

    CP = blue tones, DL = orange tones.
    Circle = no_s7, Triangle = with_s7.
    Filled = significant, Hollow = not significant.
    Thin gray connecting line per target showing spread.
    """
    _nmap_cols = [c for c in merged.columns if c.startswith("nmap_")]
    _plot_merged = merged.copy()
    _plot_merged["_max_nmap"] = _plot_merged[_nmap_cols].max(axis=1)
    _plot_df = _plot_merged.nlargest(top_n, "_max_nmap").copy()
    _plot_df = _plot_df.sort_values("_max_nmap", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(_plot_df) * 0.3)))

    _y_positions = range(len(_plot_df))
    _config_labels = list(CONFIG_COLORS.keys())

    for _y_idx, (_, _row) in enumerate(_plot_df.iterrows()):
        _x_vals = []

        for _config in _config_labels:
            _nmap_val = _row[f"nmap_{_config}"]
            _sig_val = _row[f"sig_{_config}"]
            if pd.isna(_nmap_val):
                continue

            _x_vals.append(_nmap_val)
            _color = CONFIG_COLORS[_config]
            _marker = CONFIG_MARKERS[_config]

            _is_sig = bool(_sig_val) if pd.notna(_sig_val) else False
            if _is_sig:
                ax.scatter(
                    _nmap_val,
                    _y_idx,
                    marker=_marker,
                    s=70,
                    c=_color,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )
            else:
                ax.scatter(
                    _nmap_val,
                    _y_idx,
                    marker=_marker,
                    s=70,
                    facecolors="none",
                    edgecolors=_color,
                    linewidths=1.5,
                    zorder=3,
                )

        if len(_x_vals) >= 2:
            ax.plot(
                [min(_x_vals), max(_x_vals)],
                [_y_idx, _y_idx],
                color="#cccccc",
                linewidth=0.8,
                zorder=1,
            )

    for _y in _y_positions:
        ax.axhline(y=_y, color="#f0f0f0", linewidth=0.5, zorder=0)

    ax.set_yticks(list(_y_positions))
    ax.set_yticklabels(_plot_df["group_value"], fontsize=8)
    ax.set_xlabel("Normalized mAP", fontsize=11)
    ax.set_title(f"4-Way Comparison: {target_source}", fontsize=12)
    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#6baed6",
            markersize=9,
            label="CP no_s7 (sig)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#6baed6",
            markersize=9,
            markeredgewidth=1.5,
            label="CP no_s7 (n.s.)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#2171b5",
            markersize=9,
            label="CP with_s7 (sig)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#2171b5",
            markersize=9,
            markeredgewidth=1.5,
            label="CP with_s7 (n.s.)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#fdae6b",
            markersize=9,
            label="DL no_s7 (sig)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#fdae6b",
            markersize=9,
            markeredgewidth=1.5,
            label="DL no_s7 (n.s.)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#e6550d",
            markersize=9,
            label="DL with_s7 (sig)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#e6550d",
            markersize=9,
            markeredgewidth=1.5,
            label="DL with_s7 (n.s.)",
        ),
    ]
    ax.legend(handles=_legend_elements, loc="lower right", fontsize=8, framealpha=0.9, ncol=2)

    fig.tight_layout()
    return fig


@app.function
def plot_head_to_head(h2h: pd.DataFrame) -> plt.Figure:
    """Paired bar chart: CP vs DL significant targets per annotation source."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, _s7_label in zip(axes, ["no_s7", "with_s7"]):
        _subset = h2h[h2h.source7 == _s7_label]
        _x = np.arange(len(TARGETS))
        _width = 0.35

        _cp_vals = [_subset.loc[_subset.target == t, "cp_sig"].values[0] for t in TARGETS]
        _dl_vals = [_subset.loc[_subset.target == t, "dl_sig"].values[0] for t in TARGETS]

        ax.barh(_x - _width / 2, _cp_vals, _width, color=CONFIG_COLORS["CP_with_s7"], label="CP")
        ax.barh(_x + _width / 2, _dl_vals, _width, color=CONFIG_COLORS["DL_with_s7"], label="DL")

        ax.set_yticks(_x)
        ax.set_yticklabels(TARGETS, fontsize=9)
        ax.set_xlabel("# Significant Targets", fontsize=10)
        ax.set_title(f"CP vs DL ({_s7_label})", fontsize=12)
        ax.legend(loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


@app.function
def plot_dilution_bar(dilution: pd.DataFrame) -> plt.Figure:
    """Grouped bar: net gained/lost on existing targets per annotation source."""
    fig, ax = plt.subplots(figsize=(10, 5))

    _x = np.arange(len(TARGETS))
    _width = 0.35
    _profile_colors = {"CP": CONFIG_COLORS["CP_with_s7"], "DL": CONFIG_COLORS["DL_with_s7"]}

    for _i, _profile in enumerate(["CP", "DL"]):
        _pdf = dilution[dilution.profile == _profile].set_index("target")
        _nets = [_pdf.loc[t, "net"] if t in _pdf.index else 0 for t in TARGETS]
        _offset = (_i - 0.5) * _width
        _bars = ax.bar(
            _x + _offset,
            _nets,
            _width * 0.9,
            color=_profile_colors[_profile],
            edgecolor="white",
            alpha=0.8,
        )
        for _bar, _net in zip(_bars, _nets):
            if _net != 0:
                _va = "bottom" if _net > 0 else "top"
                ax.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _net,
                    f"{int(_net):+d}",
                    ha="center",
                    va=_va,
                    fontsize=7,
                )

    ax.set_xticks(_x)
    ax.set_xticklabels(TARGETS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Net Delta on Existing Targets (gained - lost)", fontsize=10)
    ax.set_title("Signal Dilution: Does Source 7 Help or Hurt Existing Targets?", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=_profile_colors["CP"], label="CP"),
        plt.Rectangle((0, 0), 1, 1, fc=_profile_colors["DL"], label="DL"),
    ]
    ax.legend(handles=_legend_elements, loc="upper right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Head-to-head CP vs DL helper
# ---------------------------------------------------------------------------


@app.function
def compute_head_to_head() -> pd.DataFrame:
    """CP vs DL on the same union pool, for both no_s7 and with_s7."""
    _rows = []
    for _s7_label, _cp_ds, _dl_ds in [
        (
            "no_s7",
            "compound_no_source7_active_union",
            "compound_DL_CPCNN_no_source7_active_union",
        ),
        (
            "with_s7",
            "compound_with_source7_active_union",
            "compound_DL_CPCNN_with_source7_active_union",
        ),
    ]:
        for _target in TARGETS:
            _cp = query_consistency_results(
                _cp_ds,
                preprocessing="consistency_prefiltered",
                filter_name="all_sources",
                group_type=_target,
                distance="cosine",
            )
            _dl = query_consistency_results(
                _dl_ds,
                preprocessing="consistency_prefiltered",
                filter_name="all_sources",
                group_type=_target,
                distance="cosine",
            )
            _n = len(_cp)
            _cp_sig = int(_cp["below_corrected_p"].sum())
            _dl_sig = int(_dl["below_corrected_p"].sum())
            _cp_nmap = _cp["mean_normalized_average_precision"].mean()
            _dl_nmap = _dl["mean_normalized_average_precision"].mean()
            _winner_sig = "CP" if _cp_sig > _dl_sig else ("DL" if _dl_sig > _cp_sig else "tie")
            _winner_nmap = "CP" if _cp_nmap > _dl_nmap else "DL"

            _rows.append(
                {
                    "source7": _s7_label,
                    "target": _target,
                    "n_targets": _n,
                    "cp_sig": _cp_sig,
                    "dl_sig": _dl_sig,
                    "cp_nmap": round(_cp_nmap, 4),
                    "dl_nmap": round(_dl_nmap, 4),
                    "winner_sig": _winner_sig,
                    "winner_nmap": _winner_nmap,
                }
            )
    return pd.DataFrame(_rows)


# ---------------------------------------------------------------------------
# Signal dilution helper
# ---------------------------------------------------------------------------


@app.function
def compute_dilution_stats(
    all_status_changes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute signal dilution stats from per-target status change DataFrames.

    For targets evaluable in both no_s7 and with_s7, counts how many
    stayed significant (stable_sig), lost significance (lost/diluted),
    gained significance, or stayed non-significant.
    """
    _rows = []
    for _target, _df in all_status_changes.items():
        if _df is None or len(_df) == 0:
            continue

        for _profile in ["CP", "DL"]:
            _col = f"status_{_profile}"
            _existing = _df[_df[_col].isin(["stable_sig", "stable_nonsig", "gained", "lost"])]
            _new = _df[_df[_col].isin(["new_sig", "new_nonsig"])]

            _stable_sig = (_existing[_col] == "stable_sig").sum()
            _stable_nonsig = (_existing[_col] == "stable_nonsig").sum()
            _gained = (_existing[_col] == "gained").sum()
            _lost = (_existing[_col] == "lost").sum()
            _new_sig = (_new[_col] == "new_sig").sum()
            _new_nonsig = (_new[_col] == "new_nonsig").sum()
            _previously_sig = _stable_sig + _lost

            # Mean nMAP change on matched targets
            _no_col = f"nmap_{_profile}_no_s7"
            _with_col = f"nmap_{_profile}_with_s7"
            _has_both = _df[_no_col].notna() & _df[_with_col].notna()
            _nmap_no = _df.loc[_has_both, _no_col].mean() if _has_both.any() else float("nan")
            _nmap_with = _df.loc[_has_both, _with_col].mean() if _has_both.any() else float("nan")

            _rows.append(
                {
                    "target": _target,
                    "profile": _profile,
                    "stable_sig": _stable_sig,
                    "stable_nonsig": _stable_nonsig,
                    "gained": _gained,
                    "lost": _lost,
                    "net": _gained - _lost,
                    "new_sig": _new_sig,
                    "new_nonsig": _new_nonsig,
                    "previously_sig": _previously_sig,
                    "dilution_rate": round(100 * _lost / _previously_sig, 1) if _previously_sig > 0 else 0,
                    "nmap_no_s7": round(_nmap_no, 4),
                    "nmap_with_s7": round(_nmap_with, 4),
                    "nmap_delta": round(_nmap_with - _nmap_no, 4) if pd.notna(_nmap_no) else float("nan"),
                }
            )

    return pd.DataFrame(_rows)


# ===========================================================================
# 1. Activity Summary
# ===========================================================================


@app.cell
def _(mo):
    mo.md("## 1. Activity Summary")
    return


@app.cell
def _():
    activity_summary = get_activity_summary()
    activity_summary
    return (activity_summary,)


@app.cell
def _(activity_summary):
    fig_activity = plot_activity_grouped_bar(activity_summary)
    fig_activity
    return (fig_activity,)


@app.cell
def _(activity_summary, mo):
    _lines = []
    for _profile in ["CP", "DL"]:
        _no = activity_summary[activity_summary.config == f"{_profile}_no_s7"].iloc[0]
        _ws = activity_summary[activity_summary.config == f"{_profile}_with_s7"].iloc[0]
        _delta_n = _ws["n_active"] - _no["n_active"]
        _delta_pct = _ws["pct_active"] - _no["pct_active"]
        _lines.append(f"- **{_profile}**: Delta active = {_delta_n:+,} ({_delta_pct:+.1f} pp)")
    mo.md("### Activity Delta (with_s7 - no_s7)\n\n" + "\n".join(_lines))
    return


# ===========================================================================
# 2. Consistency Summary
# ===========================================================================


@app.cell
def _(mo):
    mo.md("## 2. Consistency Summary")
    return


@app.cell
def _():
    consistency_summary = get_consistency_summary()
    consistency_summary
    return (consistency_summary,)


@app.cell
def _(consistency_summary):
    fig_heatmap_pct = plot_consistency_heatmap(
        consistency_summary,
        value_col="pct_significant",
        cell_fmt="pct",
        cmap="Blues",
        title="Consistency: % Significant Targets",
        cbar_label="% Significant",
    )
    fig_heatmap_pct
    return (fig_heatmap_pct,)


@app.cell
def _(consistency_summary):
    fig_heatmap_nmap = plot_consistency_heatmap(
        consistency_summary,
        value_col="mean_nmap",
        cell_fmt="value",
        cmap="YlGnBu",
        title="Consistency: Mean Normalized mAP",
        cbar_label="Mean nMAP",
    )
    fig_heatmap_nmap
    return (fig_heatmap_nmap,)


@app.cell
def _(consistency_summary):
    fig_delta = plot_delta_bar(consistency_summary)
    fig_delta
    return (fig_delta,)


# ===========================================================================
# 3. Per-Target Analysis (dot plots + status changes)
# ===========================================================================


@app.cell
def _(mo):
    mo.md("## 3. Per-Target 4-Way Dot Plots")
    return


@app.cell
def _(mo):
    target_dropdown = mo.ui.dropdown(
        options=TARGETS,
        value="repurposing",
        label="Annotation source",
    )
    target_dropdown
    return (target_dropdown,)


@app.cell
def _(target_dropdown, top_n_slider):
    _results = get_per_target_results(target_dropdown.value)
    selected_merged = build_status_changes(_results)
    fig_dotplot = plot_4way_dotplot(selected_merged, target_dropdown.value, top_n=top_n_slider.value)
    fig_dotplot
    return fig_dotplot, selected_merged


@app.cell
def _(mo, selected_merged, target_dropdown):
    _n_gained_cp = (selected_merged["status_CP"] == "gained").sum()
    _n_lost_cp = (selected_merged["status_CP"] == "lost").sum()
    _n_gained_dl = (selected_merged["status_DL"] == "gained").sum()
    _n_lost_dl = (selected_merged["status_DL"] == "lost").sum()
    mo.md(f"""
    **{target_dropdown.value}** status changes:
    - CP: gained={_n_gained_cp}, lost={_n_lost_cp}, net={_n_gained_cp - _n_lost_cp:+d}
    - DL: gained={_n_gained_dl}, lost={_n_lost_dl}, net={_n_gained_dl - _n_lost_dl:+d}
    """)
    return


# ===========================================================================
# 4. Signal Dilution Analysis
# ===========================================================================


@app.cell
def _(mo):
    mo.md("## 4. Signal Dilution Analysis")
    return


@app.cell
def _():
    # Build status changes for all targets (needed for dilution stats)
    all_status_changes = {}
    for _target in TARGETS:
        _results = get_per_target_results(_target)
        all_status_changes[_target] = build_status_changes(_results)
    dilution = compute_dilution_stats(all_status_changes)
    dilution
    return all_status_changes, dilution


@app.cell
def _(dilution):
    fig_dilution = plot_dilution_bar(dilution)
    fig_dilution
    return (fig_dilution,)


@app.cell
def _(dilution, mo):
    _lines = []
    for _profile in ["CP", "DL"]:
        _pdf = dilution[dilution.profile == _profile]
        _total_lost = _pdf["lost"].sum()
        _total_gained = _pdf["gained"].sum()
        _total_stable_sig = _pdf["stable_sig"].sum()
        _total_prev_sig = _pdf["previously_sig"].sum()
        _total_new_sig = _pdf["new_sig"].sum()
        _dilution_rate = 100 * _total_lost / _total_prev_sig if _total_prev_sig else 0
        _lines.append(f"**{_profile}** across {len(TARGETS)} annotation sources:")
        _lines.append(f"- Previously significant: {_total_prev_sig}")
        _lines.append(f"- Stable (still significant): {_total_stable_sig}")
        _lines.append(f"- Lost significance (diluted): {_total_lost} ({_dilution_rate:.1f}%)")
        _lines.append(f"- Gained significance: {_total_gained}")
        _lines.append(f"- Net on existing: {_total_gained - _total_lost:+d}")
        _lines.append(f"- New targets only evaluable with s7: {_total_new_sig} significant")
        _lines.append("")
    mo.md("### Aggregate Dilution\n\n" + "\n".join(_lines))
    return


# ===========================================================================
# 5. Head-to-Head CP vs DL
# ===========================================================================


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Head-to-Head: CP vs DL

    Both methods evaluated on the union of actives (identical compound set per row).
    """)
    return


@app.cell
def _():
    h2h = compute_head_to_head()
    h2h
    return (h2h,)


@app.cell
def _(h2h):
    fig_h2h = plot_head_to_head(h2h)
    fig_h2h
    return (fig_h2h,)


@app.cell
def _(h2h, mo):
    _lines = []
    for _s7_label in ["no_s7", "with_s7"]:
        _subset = h2h[h2h.source7 == _s7_label]
        _cp_wins = (_subset.winner_sig == "CP").sum()
        _dl_wins = (_subset.winner_sig == "DL").sum()
        _ties = (_subset.winner_sig == "tie").sum()
        _cp_nmap_wins = (_subset.winner_nmap == "CP").sum()
        _dl_nmap_wins = (_subset.winner_nmap == "DL").sum()
        _lines.append(
            f"- **{_s7_label}**: # sig targets - CP {_cp_wins}, DL {_dl_wins}, "
            f"ties {_ties} | mean nMAP - CP {_cp_nmap_wins}, DL {_dl_nmap_wins}"
        )
    mo.md("### Score Summary\n\n" + "\n".join(_lines))
    return


# ===========================================================================
# 6. Save outputs
# ===========================================================================


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save all outputs")
    save_button
    return (save_button,)


@app.cell
def _(
    activity_summary,
    all_status_changes,
    consistency_summary,
    dilution,
    fig_activity,
    fig_delta,
    fig_dilution,
    fig_h2h,
    fig_heatmap_nmap,
    fig_heatmap_pct,
    h2h,
    mo,
    save_button,
    top_n_slider,
):
    mo.stop(not save_button.value, "Click 'Save all outputs' to write files.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSVs
    activity_summary.to_csv(OUTPUT_DIR / "activity_summary.csv", index=False)
    consistency_summary.to_csv(OUTPUT_DIR / "consistency_summary.csv", index=False)
    dilution.to_csv(OUTPUT_DIR / "dilution_summary.csv", index=False)
    h2h.to_csv(OUTPUT_DIR / "head_to_head.csv", index=False)

    # Aggregate figures
    fig_activity.savefig(OUTPUT_DIR / "activity_grouped_bar.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_heatmap_pct.savefig(OUTPUT_DIR / "consistency_heatmap.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_heatmap_nmap.savefig(OUTPUT_DIR / "consistency_nmap_heatmap.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_delta.savefig(OUTPUT_DIR / "delta_significant_bar.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_dilution.savefig(OUTPUT_DIR / "dilution_bar.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_h2h.savefig(OUTPUT_DIR / "head_to_head.png", dpi=DEFAULT_DPI, bbox_inches="tight")

    # Per-target dot plots and status CSVs
    _per_target_dir = OUTPUT_DIR / "per_target"
    for _target in TARGETS:
        _target_dir = _per_target_dir / _target
        _target_dir.mkdir(parents=True, exist_ok=True)

        _merged = all_status_changes[_target]
        _merged.to_csv(_target_dir / "status_changes.csv", index=False)

        if len(_merged) > 0:
            _fig = plot_4way_dotplot(_merged, _target, top_n=top_n_slider.value)
            _fig.savefig(_target_dir / "4way_dotplot.png", dpi=DEFAULT_DPI, bbox_inches="tight")
            del _fig

    logger.info(f"All outputs saved to {OUTPUT_DIR}")
    mo.md(f"All outputs saved to `{OUTPUT_DIR}`")
    return


# ---------------------------------------------------------------------------
# marimo boilerplate
# ---------------------------------------------------------------------------


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
