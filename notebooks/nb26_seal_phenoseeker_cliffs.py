# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anndata",
#     "duckdb==1.5.3",
#     "loguru==0.7.3",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==2.3.3",
#     "python-dotenv",
#     "scanpy",
#     "scipy==1.17.1",
#     "seaborn",
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
    from loguru import logger
    from scipy import stats as scipy_stats
    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
    from scipy.spatial.distance import pdist, squareform

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import DEFAULT_DPI, INTERIM_DATA_DIR, METADATA_DB, PROCESSED_DATA_DIR
    from nb03_ss_profiles import load_profiles


# -- Inlined helpers from src/jump_production/statistics.py --


@app.function
def scaffold_null_separation(
    X_morph: np.ndarray,
    scaffold_sizes: list[int],
    n_samples_per_size: int = 200,
    n_clusters: int = 2,
    random_state: int = 42,
) -> dict[int, np.ndarray]:
    """Generate null separation score distributions by scaffold size.

    For each scaffold size, randomly samples compounds from the full morphology
    matrix, clusters them, and computes separation scores.
    """
    rng = np.random.default_rng(random_state)
    n_total = X_morph.shape[0]

    norms = np.linalg.norm(X_morph, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_norm = X_morph / norms

    null_distributions = {}

    for size in sorted(set(scaffold_sizes)):
        if size > n_total or size < 2:
            continue

        null_scores = np.zeros(n_samples_per_size)

        for i in range(n_samples_per_size):
            idx = rng.choice(n_total, size=size, replace=False)
            X_sample = X_norm[idx]

            distances = pdist(X_sample, metric="cosine")
            Z = linkage(distances, method="average")
            labels = fcluster(Z, n_clusters, criterion="maxclust")

            dist_matrix = squareform(distances)
            intra_dists = []
            inter_dists = []

            for ii in range(size):
                for jj in range(ii + 1, size):
                    if labels[ii] == labels[jj]:
                        intra_dists.append(dist_matrix[ii, jj])
                    else:
                        inter_dists.append(dist_matrix[ii, jj])

            mean_intra = np.mean(intra_dists) if intra_dists else 0.0
            mean_inter = np.mean(inter_dists) if inter_dists else 0.0
            null_scores[i] = mean_inter - mean_intra

        null_distributions[size] = null_scores

    return null_distributions


@app.function
def compute_scaffold_pvalue(
    observed_separation: float,
    scaffold_size: int,
    null_distributions: dict[int, np.ndarray],
) -> float:
    """Compute empirical p-value for a scaffold's separation score."""
    if not null_distributions:
        return 1.0

    available_sizes = sorted(null_distributions.keys())
    if scaffold_size in available_sizes:
        size_key = scaffold_size
    else:
        size_key = min(available_sizes, key=lambda s: abs(s - scaffold_size))

    null_scores = null_distributions[size_key]
    p_value = float(np.mean(null_scores >= observed_separation))

    return p_value


@app.function
def compute_binomial_pvalue(
    n_significant: int,
    n_total: int,
    expected_rate: float = 0.05,
) -> float:
    """Compute binomial p-value for enrichment of significant results."""
    p_value = scipy_stats.binom.sf(n_significant - 1, n_total, expected_rate)
    return float(p_value)


# -- Configuration constants --

OUTPUT_DIR = PROCESSED_DATA_DIR / "activity-cliffs-phenoseeker"

MIN_COMPOUNDS = 6
N_CLUSTERS = 2
SEPARATION_THRESHOLD = 0.5
RANDOM_STATE = 42

# -- Data loading functions --


@app.function
def load_scaffolds_with_counts(
    con: duckdb.DuckDBPyConnection,
    min_compounds: int = MIN_COMPOUNDS,
) -> pd.DataFrame:
    """Load scaffolds with N>=min_compounds."""
    df = con.execute(
        """
        SELECT
            Metadata_MurckoScaffold as scaffold,
            COUNT(*) as n_compounds,
            LIST(Metadata_JCP2022) as compound_ids
        FROM compound_metadata
        WHERE Metadata_ValidMol = TRUE
          AND Metadata_MurckoScaffold IS NOT NULL
        GROUP BY Metadata_MurckoScaffold
        HAVING COUNT(*) >= ?
        ORDER BY n_compounds DESC
    """,
        [min_compounds],
    ).df()

    logger.info(f"Loaded {len(df):,} scaffolds with N>={min_compounds} compounds")
    logger.info(f"Total compounds in these scaffolds: {df['n_compounds'].sum():,}")

    return df


@app.function
def load_fingerprints() -> tuple[np.ndarray, np.ndarray]:
    """Load Morgan fingerprints (2048-bit)."""
    fp_data = np.load(INTERIM_DATA_DIR / "compound_featurization" / "morgan_fp.npz", allow_pickle=True)
    fp_array = fp_data["fingerprints"]
    fp_jcp = fp_data["jcp2022"]
    logger.info(f"Morgan fingerprints: {fp_array.shape[0]:,} compounds x {fp_array.shape[1]} bits")
    return fp_array, fp_jcp


@app.function
def align_morphology_to_scaffolds(
    adata,
    scaffold_df: pd.DataFrame,
    min_compounds: int = MIN_COMPOUNDS,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Align morphology profiles to scaffold compound lists."""
    logger.info("Aligning morphology profiles to scaffolds...")

    profile_jcp = adata.obs["JCP2022"].values
    profile_idx = {k: i for i, k in enumerate(profile_jcp)}
    profile_set = set(profile_jcp)

    scaffold_data = {}
    n_scaffolds_with_data = 0
    n_compounds_aligned = 0

    for _, row in scaffold_df.iterrows():
        _scaffold = row["scaffold"]
        _compound_ids = row["compound_ids"]

        _valid_ids = [jcp for jcp in _compound_ids if jcp in profile_set]

        if len(_valid_ids) < min_compounds:
            continue

        _indices = [profile_idx[jcp] for jcp in _valid_ids]
        _X_morph = adata.X[_indices].astype(np.float32)

        _norms = np.linalg.norm(_X_morph, axis=1)
        _valid_mask = (_norms > 0) & ~np.isnan(_norms)

        if _valid_mask.sum() < min_compounds:
            continue

        _X_morph = _X_morph[_valid_mask]
        _valid_ids = [_valid_ids[i] for i in range(len(_valid_ids)) if _valid_mask[i]]

        scaffold_data[_scaffold] = (_X_morph, _valid_ids)
        n_scaffolds_with_data += 1
        n_compounds_aligned += len(_valid_ids)

    logger.info(f"Scaffolds with valid morphology data: {n_scaffolds_with_data:,}")
    logger.info(f"Total compounds aligned: {n_compounds_aligned:,}")

    return scaffold_data


# -- Hierarchical clustering analysis --


@app.function
def analyze_scaffold_series(
    X_morph: np.ndarray,
    jcp_ids: list[str],
    n_clusters: int = N_CLUSTERS,
) -> dict:
    """Perform hierarchical clustering on morphology within a scaffold."""
    n_compounds = len(jcp_ids)

    norms = np.linalg.norm(X_morph, axis=1, keepdims=True)
    X_norm = X_morph / norms

    distances = pdist(X_norm, metric="cosine")
    Z = linkage(distances, method="average")
    labels = fcluster(Z, n_clusters, criterion="maxclust")

    dist_matrix = squareform(distances)
    intra_dists = []
    inter_dists = []

    for i in range(n_compounds):
        for j in range(i + 1, n_compounds):
            if labels[i] == labels[j]:
                intra_dists.append(dist_matrix[i, j])
            else:
                inter_dists.append(dist_matrix[i, j])

    mean_intra_dist = float(np.mean(intra_dists)) if intra_dists else 0.0
    mean_inter_dist = float(np.mean(inter_dists)) if inter_dists else 0.0
    separation = mean_inter_dist - mean_intra_dist

    mean_intra_sim = 1.0 - mean_intra_dist
    mean_inter_sim = 1.0 - mean_inter_dist

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels.tolist(), counts.tolist()))

    return {
        "n_compounds": n_compounds,
        "labels": labels.tolist(),
        "jcp_ids": jcp_ids,
        "mean_intra_distance": mean_intra_dist,
        "mean_inter_distance": mean_inter_dist,
        "mean_intra_similarity": mean_intra_sim,
        "mean_inter_similarity": mean_inter_sim,
        "separation_score": separation,
        "cluster_sizes": cluster_sizes,
        "linkage_matrix": Z,
        "distance_matrix": dist_matrix,
    }


@app.function
def analyze_all_scaffolds(
    scaffold_data: dict[str, tuple[np.ndarray, list[str]]],
    n_clusters: int = N_CLUSTERS,
) -> list[dict]:
    """Analyze all scaffolds and compute separation scores."""
    logger.info(f"Analyzing {len(scaffold_data):,} scaffolds...")

    results = []
    for _scaffold, (_X_morph, _jcp_ids) in scaffold_data.items():
        _result = analyze_scaffold_series(_X_morph, _jcp_ids, n_clusters)
        _result["scaffold"] = _scaffold
        results.append(_result)

    results.sort(key=lambda x: -x["separation_score"])

    logger.info(f"Analysis complete for {len(results):,} scaffolds")

    return results


@app.function
def add_structural_similarities(
    scaffold_results: list[dict],
    fp_array: np.ndarray,
    fp_jcp: np.ndarray,
) -> None:
    """Add structural (Tanimoto) intra/inter-cluster similarities to scaffold results.

    Modifies scaffold_results in place.
    """
    logger.info("Computing structural similarities for scaffolds...")

    fp_idx = {k: i for i, k in enumerate(fp_jcp)}
    fp_set = set(fp_jcp)

    for result in scaffold_results:
        _jcp_ids = result["jcp_ids"]
        _labels = result["labels"]

        _valid_indices = []
        _valid_labels = []
        _fp_indices = []
        for i, jcp in enumerate(_jcp_ids):
            if jcp in fp_set:
                _valid_indices.append(i)
                _valid_labels.append(_labels[i])
                _fp_indices.append(fp_idx[jcp])

        if len(_valid_indices) < 2:
            result["struct_mean_intra_similarity"] = None
            result["struct_mean_inter_similarity"] = None
            continue

        _X_fp = fp_array[_fp_indices].astype(np.float32)
        _norms = np.linalg.norm(_X_fp, axis=1, keepdims=True)
        _norms = np.where(_norms == 0, 1, _norms)
        _X_fp_norm = _X_fp / _norms

        _sim_matrix = _X_fp_norm @ _X_fp_norm.T

        _intra_sims = []
        _inter_sims = []
        _n = len(_valid_labels)
        for i in range(_n):
            for j in range(i + 1, _n):
                if _valid_labels[i] == _valid_labels[j]:
                    _intra_sims.append(_sim_matrix[i, j])
                else:
                    _inter_sims.append(_sim_matrix[i, j])

        result["struct_mean_intra_similarity"] = float(np.mean(_intra_sims)) if _intra_sims else None
        result["struct_mean_inter_similarity"] = float(np.mean(_inter_sims)) if _inter_sims else None

    _n_valid = sum(1 for r in scaffold_results if r["struct_mean_intra_similarity"] is not None)
    logger.info(f"Structural similarities computed for {_n_valid:,} scaffolds")


# -- Activity cliff detection --


@app.function
def detect_cliff_scaffolds(
    scaffold_results: list[dict],
    threshold: float = SEPARATION_THRESHOLD,
) -> list[dict]:
    """Flag scaffold series with significant cluster separation (activity cliffs)."""
    cliff_scaffolds = [r for r in scaffold_results if r["separation_score"] > threshold]
    logger.info(f"Cliff scaffolds (separation > {threshold}): {len(cliff_scaffolds):,}")
    return cliff_scaffolds


@app.function
def identify_cliff_pairs_in_scaffold(
    scaffold_result: dict,
    top_n: int = 5,
) -> list[dict]:
    """For a cliff scaffold, identify the most extreme cliff pairs."""
    _labels = scaffold_result["labels"]
    _jcp_ids = scaffold_result["jcp_ids"]
    _dist_matrix = scaffold_result["distance_matrix"]
    _scaffold = scaffold_result["scaffold"]

    cliff_pairs = []
    _n_compounds = len(_labels)

    for i in range(_n_compounds):
        for j in range(i + 1, _n_compounds):
            if _labels[i] != _labels[j]:
                cliff_pairs.append(
                    {
                        "scaffold": _scaffold,
                        "jcp_i": _jcp_ids[i],
                        "jcp_j": _jcp_ids[j],
                        "morph_distance": float(_dist_matrix[i, j]),
                        "cluster_i": int(_labels[i]),
                        "cluster_j": int(_labels[j]),
                    }
                )

    cliff_pairs.sort(key=lambda x: -x["morph_distance"])
    return cliff_pairs[:top_n]


@app.function
def collect_all_cliff_pairs(
    cliff_scaffolds: list[dict],
    top_per_scaffold: int = 3,
) -> pd.DataFrame:
    """Collect top cliff pairs from all cliff scaffolds."""
    all_pairs = []
    for _result in cliff_scaffolds:
        _pairs = identify_cliff_pairs_in_scaffold(_result, top_n=top_per_scaffold)
        all_pairs.extend(_pairs)

    if not all_pairs:
        return pd.DataFrame()

    df = pd.DataFrame(all_pairs)
    df = df.sort_values("morph_distance", ascending=False).reset_index(drop=True)

    logger.info(f"Collected {len(df):,} cliff pairs from {len(cliff_scaffolds):,} scaffolds")

    return df


# -- Visualization functions --


@app.function
def plot_scaffold_size_distribution(scaffold_df: pd.DataFrame) -> plt.Figure:
    """Plot histogram of compounds per scaffold."""
    fig, ax = plt.subplots(figsize=(8, 5))

    _sizes = scaffold_df["n_compounds"].values
    ax.hist(_sizes, bins=50, color="#1f77b4", edgecolor="black", linewidth=0.5, alpha=0.8)

    ax.set_xlabel("Compounds per Scaffold")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title(f"Scaffold Size Distribution (N >= {_sizes.min()})")

    _stats_text = (
        f"n = {len(_sizes):,}\nMean = {_sizes.mean():.1f}\nMedian = {np.median(_sizes):.0f}\nMax = {_sizes.max()}"
    )
    ax.text(
        0.95,
        0.95,
        _stats_text,
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    return fig


@app.function
def plot_separation_score_distribution(
    scaffold_results: list[dict],
    threshold: float,
) -> plt.Figure:
    """Plot distribution of separation scores across scaffolds."""
    _scores = [r["separation_score"] for r in scaffold_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(_scores, bins=50, color="#1f77b4", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")

    ax.set_xlabel("Separation Score (inter - intra cluster distance)")
    ax.set_ylabel("Count")
    ax.set_title("Separation Score Distribution Across Scaffolds")
    ax.legend()

    _n_cliffs = sum(1 for s in _scores if s > threshold)
    _stats_text = f"n = {len(_scores):,}\nCliffs: {_n_cliffs:,} ({_n_cliffs / len(_scores) * 100:.1f}%)"
    ax.text(
        0.95,
        0.95,
        _stats_text,
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    return fig


@app.function
def plot_top_cliff_scaffolds(
    cliff_scaffolds: list[dict],
    top_n: int = 20,
) -> plt.Figure:
    """Plot bar chart of scaffolds with highest separation scores."""
    _top_scaffolds = cliff_scaffolds[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))

    _labels = [r["scaffold"][:30] + "..." if len(r["scaffold"]) > 30 else r["scaffold"] for r in _top_scaffolds]
    _scores = [r["separation_score"] for r in _top_scaffolds]
    _n_compounds = [r["n_compounds"] for r in _top_scaffolds]

    _y_pos = np.arange(len(_labels))
    _bars = ax.barh(_y_pos, _scores, color="#ff7f0e", edgecolor="black", linewidth=0.5)

    for i, (bar, n) in enumerate(zip(_bars, _n_compounds)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"n={n}", va="center", fontsize=8)

    ax.set_yticks(_y_pos)
    ax.set_yticklabels(_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Separation Score")
    ax.set_title(f"Top {len(_top_scaffolds)} Scaffold Series with Activity Cliffs")

    plt.tight_layout()
    return fig


@app.function
def plot_example_dendrograms(
    cliff_scaffolds: list[dict],
    top_n: int = 4,
) -> plt.Figure:
    """Plot dendrograms for top cliff scaffolds."""
    _n_plots = min(top_n, len(cliff_scaffolds))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    _axes_flat = axes.flatten()

    for i, _result in enumerate(cliff_scaffolds[:_n_plots]):
        _ax = _axes_flat[i]
        _scaffold = _result["scaffold"]
        _Z = _result["linkage_matrix"]
        _jcp_ids = _result["jcp_ids"]

        _short_labels = [jcp[-6:] for jcp in _jcp_ids]

        dendrogram(
            _Z,
            ax=_ax,
            labels=_short_labels,
            leaf_rotation=90,
            leaf_font_size=8,
        )

        _scaffold_short = _scaffold[:40] + "..." if len(_scaffold) > 40 else _scaffold
        _ax.set_title(f"{_scaffold_short}\nSep={_result['separation_score']:.3f}", fontsize=9)
        _ax.set_xlabel("")

    for i in range(_n_plots, 4):
        _axes_flat[i].set_visible(False)

    plt.suptitle("Hierarchical Clustering Dendrograms for Top Cliff Scaffolds", fontsize=11)
    plt.tight_layout()
    return fig


@app.function
def plot_intra_inter_phenotypic(
    scaffold_results: list[dict],
    cliff_scaffolds: list[dict],
) -> plt.Figure:
    """Plot intra vs inter-cluster phenotypic similarity."""
    _cliff_scaffold_set = {r["scaffold"] for r in cliff_scaffolds}

    _cliff_intra, _cliff_inter = [], []
    _noncliff_intra, _noncliff_inter = [], []

    for r in scaffold_results:
        _intra = r["mean_intra_similarity"]
        _inter = r["mean_inter_similarity"]
        if r["scaffold"] in _cliff_scaffold_set:
            _cliff_intra.append(_intra)
            _cliff_inter.append(_inter)
        else:
            _noncliff_intra.append(_intra)
            _noncliff_inter.append(_inter)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(_noncliff_intra, _noncliff_inter, c="gray", alpha=0.4, s=20, label="Other scaffolds")
    ax.scatter(
        _cliff_intra, _cliff_inter, c="red", alpha=0.7, s=30, label=f"Activity cliffs (n={len(cliff_scaffolds)})"
    )

    _all_intra = _cliff_intra + _noncliff_intra
    _all_inter = _cliff_inter + _noncliff_inter
    _min_val = min(min(_all_intra), min(_all_inter)) - 0.05
    _max_val = max(max(_all_intra), max(_all_inter)) + 0.05
    ax.plot([_min_val, _max_val], [_min_val, _max_val], "k--", alpha=0.5, label="Intra = Inter")

    ax.set_xlabel("Intra-cluster Phenotypic Similarity", fontsize=11)
    ax.set_ylabel("Inter-cluster Phenotypic Similarity", fontsize=11)
    ax.set_title(
        f"Intra vs Inter-cluster Phenotypic Similarity\n(Activity cliff scaffolds in red, n={len(_cliff_intra)})",
        fontsize=12,
    )
    ax.legend(loc="upper left")
    ax.set_xlim(_min_val, _max_val)
    ax.set_ylim(_min_val, _max_val)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


@app.function
def plot_intra_inter_structural(
    scaffold_results: list[dict],
    cliff_scaffolds: list[dict],
) -> plt.Figure:
    """Plot intra vs inter-cluster structural similarity."""
    _cliff_scaffold_set = {r["scaffold"] for r in cliff_scaffolds}

    _cliff_intra, _cliff_inter = [], []
    _noncliff_intra, _noncliff_inter = [], []

    for r in scaffold_results:
        _intra = r.get("struct_mean_intra_similarity")
        _inter = r.get("struct_mean_inter_similarity")
        if _intra is None or _inter is None:
            continue
        if r["scaffold"] in _cliff_scaffold_set:
            _cliff_intra.append(_intra)
            _cliff_inter.append(_inter)
        else:
            _noncliff_intra.append(_intra)
            _noncliff_inter.append(_inter)

    fig, ax = plt.subplots(figsize=(8, 8))

    if not _cliff_intra and not _noncliff_intra:
        ax.text(0.5, 0.5, "No structural similarity data available", transform=ax.transAxes, ha="center")
        return fig

    ax.scatter(_noncliff_intra, _noncliff_inter, c="gray", alpha=0.4, s=20, label="Other scaffolds")
    ax.scatter(_cliff_intra, _cliff_inter, c="blue", alpha=0.7, s=30, label=f"Activity cliffs (n={len(_cliff_intra)})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Intra = Inter")

    ax.set_xlabel("Intra-cluster Structural Similarity (Tanimoto)", fontsize=11)
    ax.set_ylabel("Inter-cluster Structural Similarity (Tanimoto)", fontsize=11)
    _n_cliff_total = len(cliff_scaffolds)
    _n_cliff_with_struct = len(_cliff_intra)
    ax.set_title(
        f"Intra vs Inter-cluster Structural Similarity\n"
        f"(Phenotypic activity cliff scaffolds in blue: {_n_cliff_with_struct}/{_n_cliff_total})",
        fontsize=12,
    )
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


# -- Summary generation --


@app.function
def generate_summary(
    scaffold_df: pd.DataFrame,
    scaffold_results: list[dict],
    cliff_scaffolds: list[dict],
    cliff_pairs_df: pd.DataFrame,
    threshold: float,
    null_model_stats: dict | None = None,
) -> dict:
    """Generate summary dict with all results."""
    _scores = [r["separation_score"] for r in scaffold_results]

    summary = {
        "task": "PhenoSeeker-style activity cliffs analysis",
        "reference": "Sanchez et al. 2025 (PhenoSeeker)",
        "research_question": "Identify scaffold series with high morphological heterogeneity",
        "method": "Hierarchical clustering within Bemis-Murcko scaffolds",
        "parameters": {
            "min_compounds_per_scaffold": int(scaffold_df["n_compounds"].min()),
            "n_clusters": N_CLUSTERS,
            "separation_threshold": threshold,
        },
        "scaffold_statistics": {
            "n_scaffolds_analyzed": len(scaffold_results),
            "total_compounds": int(scaffold_df["n_compounds"].sum()),
            "mean_compounds_per_scaffold": float(scaffold_df["n_compounds"].mean()),
            "max_compounds_per_scaffold": int(scaffold_df["n_compounds"].max()),
        },
        "separation_scores": {
            "mean": float(np.mean(_scores)),
            "std": float(np.std(_scores)),
            "median": float(np.median(_scores)),
            "max": float(np.max(_scores)),
            "min": float(np.min(_scores)),
        },
        "activity_cliffs": {
            "n_cliff_scaffolds": len(cliff_scaffolds),
            "pct_cliff_scaffolds": float(len(cliff_scaffolds) / len(scaffold_results) * 100),
            "n_cliff_pairs": len(cliff_pairs_df),
            "comparison_to_phenoseeker": "PhenoSeeker found 81 scaffold series with activity cliffs",
        },
        "top_cliff_scaffolds": [
            {
                "scaffold": r["scaffold"],
                "n_compounds": r["n_compounds"],
                "separation_score": r["separation_score"],
                "p_value": r.get("p_value"),
            }
            for r in cliff_scaffolds[:10]
        ],
        "interpretation": "",
    }

    if null_model_stats is not None:
        summary["null_model"] = null_model_stats

    n_cliffs = len(cliff_scaffolds)
    if null_model_stats is not None:
        _enrichment = null_model_stats.get("enrichment_p05", 1.0)
        _binomial_p = null_model_stats.get("binomial_pvalue", 1.0)
        if _binomial_p < 0.001 and _enrichment > 2:
            _interpretation = (
                f"Found {n_cliffs} cliff scaffolds with {_enrichment:.1f}x enrichment of significant "
                f"separation scores (p < {_binomial_p:.2e}), confirming systematic activity cliffs"
            )
        elif _binomial_p < 0.05:
            _interpretation = (
                f"Found {n_cliffs} cliff scaffolds with modest {_enrichment:.1f}x enrichment "
                f"(p = {_binomial_p:.3f}), suggesting some activity cliff signal"
            )
        else:
            _interpretation = (
                f"Found {n_cliffs} cliff scaffolds but enrichment ({_enrichment:.1f}x) is not "
                f"significant (p = {_binomial_p:.3f}), activity cliffs may be chance occurrences"
            )
    else:
        if n_cliffs > 100:
            _interpretation = f"Found {n_cliffs} scaffold series with activity cliffs (more than PhenoSeeker's 81)"
        elif n_cliffs > 50:
            _interpretation = f"Found {n_cliffs} scaffold series with activity cliffs (comparable to PhenoSeeker's 81)"
        elif n_cliffs > 20:
            _interpretation = f"Found {n_cliffs} scaffold series with activity cliffs (fewer than PhenoSeeker's 81)"
        else:
            _interpretation = (
                f"Found only {n_cliffs} scaffold series with activity cliffs (much fewer than PhenoSeeker's 81)"
            )

    summary["interpretation"] = _interpretation

    return summary


# =====================================================================
# Notebook cells
# =====================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PhenoSeeker-Style Activity Cliffs Analysis

    Reproduces PhenoSeeker's (Sanchez et al. 2025) activity cliff detection approach
    using CellProfiler features instead of DINOv2-Giant embeddings.

    **Research question:** Can we identify scaffold series where structurally similar
    compounds (same Bemis-Murcko scaffold) exhibit dramatically different morphological
    phenotypes?

    **Approach:**

    1. Group compounds by Bemis-Murcko scaffold
    2. Focus on scaffold series with N>=6 compounds
    3. Perform hierarchical clustering by morphology within each scaffold
    4. Flag scaffolds where mean(inter-cluster) - mean(intra-cluster) > threshold

    Reference: PhenoSeeker found 81 scaffold series with activity cliffs.
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
    min_compounds_slider = mo.ui.slider(
        start=4,
        stop=20,
        step=1,
        value=MIN_COMPOUNDS,
        label="Min compounds per scaffold",
    )
    threshold_slider = mo.ui.slider(
        start=0.1,
        stop=1.0,
        step=0.05,
        value=SEPARATION_THRESHOLD,
        label="Separation threshold",
    )
    mo.hstack([dataset_dropdown, min_compounds_slider, threshold_slider])
    return (dataset_dropdown, min_compounds_slider, threshold_slider)


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run analysis")
    run_button
    return (run_button,)


@app.cell
def _(dataset_dropdown, min_compounds_slider, mo, run_button, threshold_slider):
    mo.stop(not run_button.value)

    mo.stop(
        not METADATA_DB.exists(),
        mo.md(f"**Error:** Metadata database not found at `{METADATA_DB}`. Run `just run` first."),
    )

    _dataset = dataset_dropdown.value
    _min_compounds = min_compounds_slider.value
    separation_threshold = threshold_slider.value

    _metadata_con = duckdb.connect(str(METADATA_DB), read_only=True)

    # Load scaffold data
    logger.info("Loading scaffold data...")
    scaffold_df = load_scaffolds_with_counts(_metadata_con, _min_compounds)

    _metadata_con.close()

    # Load morphology profiles
    logger.info("Loading morphology profiles...")
    adata = load_profiles(_dataset, level="perturbation")

    # Load fingerprints for structural similarity
    logger.info("Loading fingerprints...")
    fp_array, fp_jcp = load_fingerprints()

    # Align morphology to scaffolds
    scaffold_data = align_morphology_to_scaffolds(adata, scaffold_df, _min_compounds)

    mo.md(f"""
    ## Data loaded

    **Dataset:** {_dataset} | **Min compounds:** {_min_compounds} | **Threshold:** {separation_threshold}

    - Scaffolds with N>={_min_compounds}: {len(scaffold_df):,}
    - Total compounds in scaffolds: {scaffold_df["n_compounds"].sum():,}
    - Scaffolds with valid morphology data: {len(scaffold_data):,}
    """)
    return (adata, fp_array, fp_jcp, scaffold_data, scaffold_df, separation_threshold)


# -- Analyze all scaffolds --


@app.cell
def _(fp_array, fp_jcp, scaffold_data):
    scaffold_results = analyze_all_scaffolds(scaffold_data, N_CLUSTERS)
    add_structural_similarities(scaffold_results, fp_array, fp_jcp)
    logger.info(f"Analyzed {len(scaffold_results):,} scaffolds")
    return (scaffold_results,)


# -- Null model analysis --


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Null Model Analysis

    Generates null separation score distributions by randomly sampling compounds
    (ignoring scaffold membership) and computing separation scores. This provides
    a baseline for what separation scores we'd expect by chance.
    """)
    return


@app.cell
def _(adata, mo, scaffold_results):
    logger.info("Computing null separation distributions by scaffold size...")
    _unique_sizes = sorted(set(r["n_compounds"] for r in scaffold_results))
    null_distributions = scaffold_null_separation(
        adata.X.astype(np.float32),
        _unique_sizes,
        n_samples_per_size=200,
        n_clusters=N_CLUSTERS,
    )
    logger.info(f"  Generated null distributions for {len(null_distributions)} scaffold sizes")

    # Add p-values to scaffold results
    logger.info("Computing p-values for scaffold separation scores...")
    for _result in scaffold_results:
        _result["p_value"] = compute_scaffold_pvalue(
            _result["separation_score"],
            _result["n_compounds"],
            null_distributions,
        )

    # Compute enrichment statistics
    _n_total = len(scaffold_results)
    _n_significant_p05 = sum(1 for r in scaffold_results if r["p_value"] < 0.05)
    _expected_p05 = _n_total * 0.05
    _enrichment_p05 = _n_significant_p05 / _expected_p05 if _expected_p05 > 0 else 0.0
    _binomial_pvalue = compute_binomial_pvalue(_n_significant_p05, _n_total, expected_rate=0.05)

    null_model_stats = {
        "method": "random_compound_sampling",
        "n_samples_per_size": 200,
        "n_significant_p05": _n_significant_p05,
        "expected_p05": _expected_p05,
        "enrichment_p05": _enrichment_p05,
        "binomial_pvalue": _binomial_pvalue,
    }

    mo.md(f"""
    ### Null model results

    - Significant scaffolds (p<0.05): {_n_significant_p05} (expected: {_expected_p05:.0f})
    - Enrichment: {_enrichment_p05:.2f}x
    - Binomial p-value: {_binomial_pvalue:.2e}
    """)
    return (null_distributions, null_model_stats)


# -- Detect cliff scaffolds --


@app.cell
def _(mo, scaffold_results, separation_threshold):
    cliff_scaffolds = detect_cliff_scaffolds(scaffold_results, separation_threshold)

    mo.md(f"""
    ## Activity Cliff Detection

    - Scaffolds analyzed: {len(scaffold_results):,}
    - Cliff scaffolds (separation > {separation_threshold}): {len(cliff_scaffolds):,}
      ({len(cliff_scaffolds) / len(scaffold_results) * 100:.1f}%)
    """)
    return (cliff_scaffolds,)


@app.cell
def _(cliff_scaffolds):
    cliff_pairs_df = collect_all_cliff_pairs(cliff_scaffolds[:100])
    logger.info(f"Collected {len(cliff_pairs_df):,} cliff pairs")
    return (cliff_pairs_df,)


# -- Visualization cells --


@app.cell
def _(scaffold_df):
    fig_scaffold_sizes = plot_scaffold_size_distribution(scaffold_df)
    fig_scaffold_sizes
    return (fig_scaffold_sizes,)


@app.cell
def _(scaffold_results, separation_threshold):
    fig_separation = plot_separation_score_distribution(scaffold_results, separation_threshold)
    fig_separation
    return (fig_separation,)


@app.cell
def _(cliff_scaffolds):
    fig_top_cliffs = plot_top_cliff_scaffolds(cliff_scaffolds, top_n=20)
    fig_top_cliffs
    return (fig_top_cliffs,)


@app.cell
def _(cliff_scaffolds):
    fig_dendrograms = plot_example_dendrograms(cliff_scaffolds, top_n=4)
    fig_dendrograms
    return (fig_dendrograms,)


@app.cell
def _(cliff_scaffolds, scaffold_results):
    fig_intra_inter_pheno = plot_intra_inter_phenotypic(scaffold_results, cliff_scaffolds)
    fig_intra_inter_pheno
    return (fig_intra_inter_pheno,)


@app.cell
def _(cliff_scaffolds, scaffold_results):
    fig_intra_inter_struct = plot_intra_inter_structural(scaffold_results, cliff_scaffolds)
    fig_intra_inter_struct
    return (fig_intra_inter_struct,)


# -- Top cliff pairs table --


@app.cell
def _(cliff_pairs_df, mo):
    if len(cliff_pairs_df) > 0:
        mo.md(f"""
        ## Top Cliff Pairs

        Top 20 most extreme cliff pairs across all cliff scaffolds:

        {cliff_pairs_df.head(20).to_markdown(index=False)}
        """)
    else:
        mo.md("No cliff pairs found with current thresholds.")
    return


# -- Save outputs --


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save all outputs")
    save_button
    return (save_button,)


@app.cell
def _(
    cliff_pairs_df,
    cliff_scaffolds,
    dataset_dropdown,
    fig_dendrograms,
    fig_intra_inter_pheno,
    fig_intra_inter_struct,
    fig_scaffold_sizes,
    fig_separation,
    fig_top_cliffs,
    mo,
    null_model_stats,
    save_button,
    scaffold_df,
    scaffold_results,
    separation_threshold,
):
    mo.stop(not save_button.value)

    _dataset = dataset_dropdown.value
    _outdir = OUTPUT_DIR / _dataset
    _outdir.mkdir(parents=True, exist_ok=True)

    # Save figures
    fig_scaffold_sizes.savefig(_outdir / "scaffold_size_distribution.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_scaffold_sizes.savefig(_outdir / "scaffold_size_distribution.pdf", bbox_inches="tight")

    fig_separation.savefig(_outdir / "separation_score_distribution.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_separation.savefig(_outdir / "separation_score_distribution.pdf", bbox_inches="tight")

    if cliff_scaffolds:
        fig_top_cliffs.savefig(_outdir / "top_cliff_scaffolds.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        fig_top_cliffs.savefig(_outdir / "top_cliff_scaffolds.pdf", bbox_inches="tight")

        fig_dendrograms.savefig(_outdir / "example_dendrograms.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        fig_dendrograms.savefig(_outdir / "example_dendrograms.pdf", bbox_inches="tight")

    fig_intra_inter_pheno.savefig(_outdir / "intra_inter_phenotypic.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_intra_inter_pheno.savefig(_outdir / "intra_inter_phenotypic.pdf", bbox_inches="tight")

    fig_intra_inter_struct.savefig(_outdir / "intra_inter_structural.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    fig_intra_inter_struct.savefig(_outdir / "intra_inter_structural.pdf", bbox_inches="tight")

    # Save data
    _scaffold_analysis_df = pd.DataFrame(
        [
            {
                "scaffold": r["scaffold"],
                "n_compounds": r["n_compounds"],
                "mean_intra_similarity": r["mean_intra_similarity"],
                "mean_inter_similarity": r["mean_inter_similarity"],
                "separation_score": r["separation_score"],
                "p_value": r.get("p_value"),
                "struct_mean_intra_similarity": r.get("struct_mean_intra_similarity"),
                "struct_mean_inter_similarity": r.get("struct_mean_inter_similarity"),
                "cluster_sizes": str(r["cluster_sizes"]),
            }
            for r in scaffold_results
        ]
    )
    _scaffold_analysis_df.to_parquet(_outdir / "scaffold_analysis.parquet", index=False)

    if cliff_scaffolds:
        _cliff_scaffolds_df = pd.DataFrame(
            [
                {
                    "scaffold": r["scaffold"],
                    "n_compounds": r["n_compounds"],
                    "separation_score": r["separation_score"],
                    "jcp_ids": "|".join(r["jcp_ids"]),
                }
                for r in cliff_scaffolds[:100]
            ]
        )
        _cliff_scaffolds_df.to_csv(_outdir / "cliff_scaffolds.csv", index=False)

    if len(cliff_pairs_df) > 0:
        cliff_pairs_df.to_csv(_outdir / "cliff_pairs.csv", index=False)

    _summary = generate_summary(
        scaffold_df,
        scaffold_results,
        cliff_scaffolds,
        cliff_pairs_df,
        separation_threshold,
        null_model_stats=null_model_stats,
    )
    with open(_outdir / "summary.json", "w") as _f:
        json.dump(_summary, _f, indent=2)

    mo.md(f"""
    **Saved all outputs to:** `{_outdir}`

    - `scaffold_size_distribution.png/.pdf`
    - `separation_score_distribution.png/.pdf`
    - `top_cliff_scaffolds.png/.pdf`
    - `example_dendrograms.png/.pdf`
    - `intra_inter_phenotypic.png/.pdf`
    - `intra_inter_structural.png/.pdf`
    - `scaffold_analysis.parquet`
    - `cliff_scaffolds.csv`
    - `cliff_pairs.csv`
    - `summary.json`
    """)
    return


@app.function
def run_phenoseeker(
    dataset: str = "compound_no_source7",
    output_dir=None,
) -> str:
    """Run the full PhenoSeeker-style activity cliff analysis.

    Composes load_scaffolds_with_counts, load_profiles, load_fingerprints,
    align_morphology_to_scaffolds, analyze_all_scaffolds,
    add_structural_similarities, scaffold_null_separation,
    detect_cliff_scaffolds, collect_all_cliff_pairs,
    plot_separation_score_distribution, plot_top_cliff_scaffolds,
    and generate_summary. Saves plots, data files, summary.json,
    and .complete marker.

    Called from workflow.py via run_task.py.

    Args:
        dataset: Dataset name (e.g., "compound_no_source7")
        output_dir: Output directory (default: data/processed/activity-cliffs-phenoseeker/...)

    Returns:
        Path to .complete marker file.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / dataset
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    _metadata_con = duckdb.connect(str(METADATA_DB), read_only=True)
    _scaffold_df = load_scaffolds_with_counts(_metadata_con, MIN_COMPOUNDS)
    _metadata_con.close()

    _adata = load_profiles(dataset, level="perturbation")
    _fp_array, _fp_jcp = load_fingerprints()

    # Align morphology to scaffolds
    _scaffold_data = align_morphology_to_scaffolds(_adata, _scaffold_df, MIN_COMPOUNDS)

    # Analyze all scaffolds
    _scaffold_results = analyze_all_scaffolds(_scaffold_data, N_CLUSTERS)
    add_structural_similarities(_scaffold_results, _fp_array, _fp_jcp)

    # Null model analysis
    _unique_sizes = sorted(set(r["n_compounds"] for r in _scaffold_results))
    _null_distributions = scaffold_null_separation(
        _adata.X.astype(np.float32), _unique_sizes, n_samples_per_size=200, n_clusters=N_CLUSTERS
    )

    # Add p-values
    for _result in _scaffold_results:
        _result["p_value"] = compute_scaffold_pvalue(
            _result["separation_score"], _result["n_compounds"], _null_distributions
        )

    # Compute enrichment statistics
    _n_total = len(_scaffold_results)
    _n_significant_p05 = sum(1 for r in _scaffold_results if r["p_value"] < 0.05)
    _expected_p05 = _n_total * 0.05
    _enrichment_p05 = _n_significant_p05 / _expected_p05 if _expected_p05 > 0 else 0.0
    _binomial_pvalue = compute_binomial_pvalue(_n_significant_p05, _n_total, expected_rate=0.05)

    _null_model_stats = {
        "method": "random_compound_sampling",
        "n_samples_per_size": 200,
        "n_significant_p05": _n_significant_p05,
        "expected_p05": _expected_p05,
        "enrichment_p05": _enrichment_p05,
        "binomial_pvalue": _binomial_pvalue,
    }

    # Detect cliff scaffolds
    _cliff_scaffolds = detect_cliff_scaffolds(_scaffold_results, SEPARATION_THRESHOLD)
    _cliff_pairs_df = collect_all_cliff_pairs(_cliff_scaffolds[:100])

    # Save figures
    _fig_separation = plot_separation_score_distribution(_scaffold_results, SEPARATION_THRESHOLD)
    _fig_separation.savefig(
        output_dir / "separation_score_distribution.png", dpi=150, bbox_inches="tight", facecolor="white"
    )
    _fig_separation.savefig(output_dir / "separation_score_distribution.pdf", bbox_inches="tight")
    plt.close(_fig_separation)

    if _cliff_scaffolds:
        _fig_top_cliffs = plot_top_cliff_scaffolds(_cliff_scaffolds, top_n=20)
        _fig_top_cliffs.savefig(output_dir / "top_cliff_scaffolds.png", dpi=150, bbox_inches="tight", facecolor="white")
        _fig_top_cliffs.savefig(output_dir / "top_cliff_scaffolds.pdf", bbox_inches="tight")
        plt.close(_fig_top_cliffs)

    _fig_intra_inter_pheno = plot_intra_inter_phenotypic(_scaffold_results, _cliff_scaffolds)
    _fig_intra_inter_pheno.savefig(
        output_dir / "intra_inter_phenotypic.png", dpi=150, bbox_inches="tight", facecolor="white"
    )
    _fig_intra_inter_pheno.savefig(output_dir / "intra_inter_phenotypic.pdf", bbox_inches="tight")
    plt.close(_fig_intra_inter_pheno)

    _fig_intra_inter_struct = plot_intra_inter_structural(_scaffold_results, _cliff_scaffolds)
    _fig_intra_inter_struct.savefig(
        output_dir / "intra_inter_structural.png", dpi=150, bbox_inches="tight", facecolor="white"
    )
    _fig_intra_inter_struct.savefig(output_dir / "intra_inter_structural.pdf", bbox_inches="tight")
    plt.close(_fig_intra_inter_struct)

    # Save data
    _scaffold_analysis_df = pd.DataFrame(
        [
            {
                "scaffold": r["scaffold"],
                "n_compounds": r["n_compounds"],
                "mean_intra_similarity": r["mean_intra_similarity"],
                "mean_inter_similarity": r["mean_inter_similarity"],
                "separation_score": r["separation_score"],
                "p_value": r.get("p_value"),
                "struct_mean_intra_similarity": r.get("struct_mean_intra_similarity"),
                "struct_mean_inter_similarity": r.get("struct_mean_inter_similarity"),
                "cluster_sizes": str(r["cluster_sizes"]),
            }
            for r in _scaffold_results
        ]
    )
    _scaffold_analysis_df.to_parquet(output_dir / "scaffold_analysis.parquet", index=False)

    if _cliff_scaffolds:
        _cliff_scaffolds_df = pd.DataFrame(
            [
                {
                    "scaffold": r["scaffold"],
                    "n_compounds": r["n_compounds"],
                    "separation_score": r["separation_score"],
                    "jcp_ids": "|".join(r["jcp_ids"]),
                }
                for r in _cliff_scaffolds[:100]
            ]
        )
        _cliff_scaffolds_df.to_csv(output_dir / "cliff_scaffolds.csv", index=False)

    if len(_cliff_pairs_df) > 0:
        _cliff_pairs_df.to_csv(output_dir / "cliff_pairs.csv", index=False)

    # Generate and save summary
    _summary = generate_summary(
        _scaffold_df,
        _scaffold_results,
        _cliff_scaffolds,
        _cliff_pairs_df,
        SEPARATION_THRESHOLD,
        null_model_stats=_null_model_stats,
    )
    Path(output_dir, "summary.json").write_text(json.dumps(_summary, indent=2))

    # Write .complete marker
    _marker = output_dir / ".complete"
    _marker.touch()
    logger.success(f"Saved PhenoSeeker activity cliff outputs to {output_dir}")
    return str(_marker)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
