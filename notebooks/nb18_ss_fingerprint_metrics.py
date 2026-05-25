# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "loguru",
#     "matplotlib",
#     "numpy",
#     "python-dotenv",
#     "scikit-learn",
#     "scipy",
# ]
# NOTE: Run with pixi run -e cheminformatics marimo edit/run
# rdkit is conda-only and comes from the pixi cheminformatics env.
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import json
    import sys
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from loguru import logger
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neighbors import NearestNeighbors

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from nb00_ss_config import DEFAULT_DPI, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

    MORGAN_FP_FILE = INTERIM_DATA_DIR / "compound_featurization" / "morgan_fp.npz"
    OUTPUT_DIR = PROCESSED_DATA_DIR / "fingerprint-metrics"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fingerprint Metrics

    Evaluate different transformations of Morgan fingerprints to find methods
    where standard metrics (Euclidean, cosine) approximate Tanimoto/Jaccard
    similarity.

    **Motivation:**
    - Tanimoto/Jaccard is the chemically correct metric for binary fingerprints
    - Many ML algorithms require Euclidean or inner product spaces
    - We want fast transformations that preserve similarity relationships

    **Transformations tested:**
    1. L2 normalization (baseline)
    2. Popcount normalization (divide by sqrt of bit count)
    3. Concatenated features (add bit count info)
    4. MinHash signatures (fast vectorized)
    5. IDF-weighted features
    6. Tanimoto Random Features (simplified)

    **Key finding:** Simple L2 normalization achieves rho=0.998 correlation with
    Tanimoto for JUMP compounds due to homogeneous bit densities (CV=20.4%).
    See Issue #26.

    **Environment:** Requires `pixi run -e cheminformatics marimo edit/run`
    (rdkit is conda-only).
    """)
    return


@app.cell
def _(mo):
    sample_size_input = mo.ui.number(value=3000, start=500, stop=10000, step=500, label="Sample size")
    n_minhash_input = mo.ui.number(value=256, start=64, stop=1024, step=64, label="MinHash functions")
    skip_slow_input = mo.ui.checkbox(value=False, label="Skip slow methods (MinHash)")
    run_umap_input = mo.ui.checkbox(value=False, label="Run UMAP comparison")
    seed_input = mo.ui.number(value=42, start=0, stop=9999, step=1, label="Seed")

    mo.vstack(
        [
            mo.md("### Parameters"),
            mo.hstack([sample_size_input, n_minhash_input, seed_input]),
            mo.hstack([skip_slow_input, run_umap_input]),
        ]
    )
    return (
        n_minhash_input,
        run_umap_input,
        sample_size_input,
        seed_input,
        skip_slow_input,
    )


@app.function
def tanimoto_similarity(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Compute pairwise Tanimoto similarity for binary fingerprints.

    Tanimoto(A,B) = |A intersection B| / |A union B| = (A.B) / (|A| + |B| - A.B)
    """
    if Y is None:
        Y = X

    intersection = X @ Y.T

    X_bits = X.sum(axis=1, keepdims=True)
    Y_bits = Y.sum(axis=1, keepdims=True)

    union = X_bits + Y_bits.T - intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = intersection / union
        similarity = np.nan_to_num(similarity, nan=0.0)

    return similarity


@app.function
def tanimoto_distance(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Compute pairwise Tanimoto distance (1 - similarity)."""
    return 1.0 - tanimoto_similarity(X, Y)


@app.function
def transform_l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2 normalize fingerprints to unit sphere.

    After transformation: inner product = cosine similarity.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return X / norms


@app.function
def transform_popcount_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize by sqrt of popcount (number of set bits).

    After transformation: inner product = (A.B) / sqrt(|A|.|B|)
    This equals cosine similarity for the original binary vectors.
    """
    popcounts = X.sum(axis=1, keepdims=True)
    sqrt_popcounts = np.sqrt(np.maximum(popcounts, 1.0))
    return X / sqrt_popcounts


@app.function
def transform_concatenate_popcount(X: np.ndarray) -> np.ndarray:
    """L2 normalize and concatenate popcount features.

    Appends sqrt(popcount) and 1/sqrt(popcount) to capture
    the bit count information that Tanimoto uses.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    _X_norm = X / norms

    popcounts = X.sum(axis=1, keepdims=True)
    _sqrt_pop = np.sqrt(np.maximum(popcounts, 1.0))
    _inv_sqrt_pop = 1.0 / _sqrt_pop

    _sqrt_pop_scaled = _sqrt_pop / 10.0
    _inv_sqrt_pop_scaled = _inv_sqrt_pop * 10.0

    return np.hstack([_X_norm, _sqrt_pop_scaled, _inv_sqrt_pop_scaled])


@app.function
def transform_minhash_fast(X: np.ndarray, n_hashes: int = 256, seed: int = 42) -> np.ndarray:
    """Fast vectorized MinHash using random hash values.

    Instead of permutations, assigns random hash values to each bit position
    and takes the minimum hash value among set bits.
    """
    _n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)

    signatures = np.zeros((_n_samples, n_hashes), dtype=np.float32)

    for h in range(n_hashes):
        hash_vals = rng.random(n_features).astype(np.float32)
        masked = np.where(X == 1, hash_vals, np.inf)
        signatures[:, h] = masked.min(axis=1)

    signatures[signatures == np.inf] = 1.0
    return signatures


@app.function
def transform_idf_weighted(X: np.ndarray, smooth_idf: bool = True) -> np.ndarray:
    """Weight bits by inverse document frequency and L2 normalize.

    Rare bits (present in few compounds) get higher weights.
    Common bits get lower weights.
    """
    n_samples, _n_features = X.shape

    df = X.sum(axis=0)

    if smooth_idf:
        idf = np.log((n_samples + 1) / (df + 1)) + 1
    else:
        idf = np.log(n_samples / np.maximum(df, 1))

    _X_weighted = X * idf

    norms = np.linalg.norm(_X_weighted, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)

    return _X_weighted / norms


@app.function
def transform_tanimoto_random_features(X: np.ndarray, n_features_out: int = 512, seed: int = 42) -> np.ndarray:
    """Simplified Tanimoto random features based on power series expansion.

    Uses random projections with popcount-based scaling.
    """
    _n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)

    W = rng.choice([-1, 1], size=(n_features, n_features_out)).astype(np.float32)
    W /= np.sqrt(n_features)

    Z = X @ W

    popcounts = X.sum(axis=1, keepdims=True)
    scale = 1.0 / np.sqrt(np.maximum(popcounts, 1.0))

    return Z * scale


@app.function
def compute_knn_recall(
    X_transformed: np.ndarray,
    X_original: np.ndarray,
    k_values: list[int],
    metric_transformed: str = "cosine",
) -> dict[int, float]:
    """Compute k-NN recall: fraction of true Tanimoto neighbors found."""
    n_samples = X_original.shape[0]

    logger.info("Computing ground truth Tanimoto k-NN...")
    _tanimoto_dist = tanimoto_distance(X_original)
    np.fill_diagonal(_tanimoto_dist, np.inf)

    max_k = max(k_values)

    true_neighbors = np.argsort(_tanimoto_dist, axis=1)[:, :max_k]

    logger.info(f"Computing predicted k-NN ({metric_transformed})...")
    if metric_transformed == "cosine":
        sim = cosine_similarity(X_transformed)
        np.fill_diagonal(sim, -np.inf)
        pred_neighbors = np.argsort(-sim, axis=1)[:, :max_k]
    elif metric_transformed == "euclidean":
        nn = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
        nn.fit(X_transformed)
        _, indices = nn.kneighbors(X_transformed)
        pred_neighbors = indices[:, 1:]
    else:
        raise ValueError(f"Unknown metric: {metric_transformed}")

    recalls = {}
    for k in k_values:
        true_k = true_neighbors[:, :k]
        pred_k = pred_neighbors[:, :k]

        recall_per_sample = []
        for i in range(n_samples):
            intersection = len(set(true_k[i]) & set(pred_k[i]))
            recall_per_sample.append(intersection / k)

        recalls[k] = np.mean(recall_per_sample)

    return recalls


@app.function
def evaluate_transformation(
    name: str,
    transform_fn,
    X_sample: np.ndarray,
    X_full: np.ndarray,
    tanimoto_flat: np.ndarray,
    k_values: list[int],
    metric: str = "cosine",
) -> dict:
    """Evaluate a single transformation method.

    Returns dict with timing and accuracy metrics.
    """
    result = {"name": name, "metric": metric}

    logger.info(f"Timing {name} on full dataset ({X_full.shape[0]:,} samples)...")
    t0 = time.perf_counter()
    _ = transform_fn(X_full)
    result["transform_time_full_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _X_transformed = transform_fn(X_sample)
    result["transform_time_sample_s"] = time.perf_counter() - t0
    result["output_dim"] = _X_transformed.shape[1]

    logger.info(f"Computing {metric} on transformed space...")
    if metric == "cosine":
        _sim_transformed = cosine_similarity(_X_transformed)
        _dist_transformed_flat = 1.0 - _sim_transformed[np.triu_indices(len(X_sample), k=1)]
    elif metric == "euclidean":
        _dist_transformed = cdist(_X_transformed, _X_transformed, metric="euclidean")
        _dist_transformed_flat = _dist_transformed[np.triu_indices(len(X_sample), k=1)]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    tanimoto_dist_flat = 1.0 - tanimoto_flat
    correlation, p_value = spearmanr(_dist_transformed_flat, tanimoto_dist_flat)
    result["spearman_correlation"] = correlation
    result["spearman_p_value"] = p_value

    logger.info(f"Computing k-NN recall for {name}...")
    recalls = compute_knn_recall(_X_transformed, X_sample, k_values, metric_transformed=metric)
    for k, recall in recalls.items():
        result[f"recall_at_{k}"] = recall

    return result


@app.cell
def _(mo):
    mo.stop(
        not MORGAN_FP_FILE.exists(),
        mo.md(f"**Morgan fingerprint file not found:** `{MORGAN_FP_FILE}`\n\nRun the featurization step first."),
    )

    logger.info(f"Loading Morgan fingerprints from {MORGAN_FP_FILE}...")
    _fp_data = np.load(MORGAN_FP_FILE, allow_pickle=True)
    X_full = _fp_data["fingerprints"].astype(np.float32)
    _valid_mask = _fp_data["valid_mask"]

    X_full = X_full[_valid_mask]
    logger.info(f"Loaded {X_full.shape[0]:,} valid Morgan fingerprints, {X_full.shape[1]} bits")
    mo.md(f"Loaded **{X_full.shape[0]:,}** valid Morgan fingerprints ({X_full.shape[1]} bits each)")
    return (X_full,)


@app.cell
def _(X_full, sample_size_input, seed_input):
    _sample_size = sample_size_input.value
    _seed = seed_input.value
    _rng = np.random.default_rng(_seed)

    if _sample_size < len(X_full):
        _sample_idx = _rng.choice(len(X_full), size=_sample_size, replace=False)
        X_sample = X_full[_sample_idx]
    else:
        X_sample = X_full

    logger.info(f"Using {len(X_sample):,} samples for accuracy evaluation")

    logger.info("Computing ground truth Tanimoto similarities...")
    _tanimoto_sim = tanimoto_similarity(X_sample)
    _triu_idx = np.triu_indices(len(X_sample), k=1)
    tanimoto_flat = _tanimoto_sim[_triu_idx]
    return X_sample, tanimoto_flat


@app.cell
def _(
    X_full,
    X_sample,
    n_minhash_input,
    seed_input,
    skip_slow_input,
    tanimoto_flat,
):
    _seed = seed_input.value
    _n_minhash = n_minhash_input.value
    _skip_slow = skip_slow_input.value
    _k_values = [10, 50, 100]

    _transformations = {
        "L2 Normalize": (transform_l2_normalize, "cosine"),
        "Popcount Normalize": (transform_popcount_normalize, "cosine"),
        "Concat Popcount": (transform_concatenate_popcount, "euclidean"),
        "IDF Weighted": (transform_idf_weighted, "cosine"),
        "Random Features": (
            lambda X: transform_tanimoto_random_features(X, n_features_out=512, seed=_seed),
            "cosine",
        ),
    }

    if not _skip_slow:
        _transformations["MinHash (fast)"] = (
            lambda X: transform_minhash_fast(X, n_hashes=_n_minhash, seed=_seed),
            "euclidean",
        )

    results = []
    for _name, (_transform_fn, _metric) in _transformations.items():
        logger.info(f"Evaluating: {_name}")
        _result = evaluate_transformation(
            name=_name,
            transform_fn=_transform_fn,
            X_sample=X_sample,
            X_full=X_full,
            tanimoto_flat=tanimoto_flat,
            k_values=_k_values,
            metric=_metric,
        )
        results.append(_result)
        logger.info(
            f"  rho={_result['spearman_correlation']:.4f}  "
            f"R@10={_result.get('recall_at_10', 0):.4f}  "
            f"time={_result['transform_time_full_s']:.3f}s"
        )

    transformations_dict = _transformations
    return results, transformations_dict


@app.cell
def _(X_sample, transformations_dict):
    _tanimoto_sim = tanimoto_similarity(X_sample)
    _tanimoto_dist = 1.0 - _tanimoto_sim
    _n = len(X_sample)
    _triu_idx = np.triu_indices(_n, k=1)
    _tanimoto_flat = _tanimoto_dist[_triu_idx]

    _n_pairs = 10000
    _rng = np.random.default_rng(42)
    if len(_tanimoto_flat) > _n_pairs:
        _sample_idx = _rng.choice(len(_tanimoto_flat), size=_n_pairs, replace=False)
    else:
        _sample_idx = np.arange(len(_tanimoto_flat))

    _tanimoto_sample = _tanimoto_flat[_sample_idx]

    _n_transforms = len(transformations_dict)
    _n_cols = 3
    _n_rows = (_n_transforms + _n_cols - 1) // _n_cols

    fig_correlation, _axes = plt.subplots(_n_rows, _n_cols, figsize=(4 * _n_cols, 4 * _n_rows))
    _axes = _axes.flatten() if _n_transforms > 1 else [_axes]

    for _ax, (_name, (_transform_fn, _metric)) in zip(_axes, transformations_dict.items()):
        _X_t = _transform_fn(X_sample)

        if _metric == "cosine":
            _sim = cosine_similarity(_X_t)
            _dist_flat = 1.0 - _sim[_triu_idx]
        else:
            _dist = cdist(_X_t, _X_t, metric="euclidean")
            _dist_flat = _dist[_triu_idx]

        _dist_sample = _dist_flat[_sample_idx]
        _corr, _ = spearmanr(_dist_sample, _tanimoto_sample)

        _ax.scatter(_tanimoto_sample, _dist_sample, s=1, alpha=0.1, rasterized=True)
        _ax.set_xlabel("Tanimoto Distance", fontsize=9)
        _ax.set_ylabel(f"{_metric.title()} Distance", fontsize=9)
        _ax.set_title(f"{_name}\nSpearman rho = {_corr:.3f}", fontsize=10)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

    for _ax in _axes[_n_transforms:]:
        _ax.set_visible(False)

    plt.tight_layout()
    fig_correlation
    return (fig_correlation,)


@app.cell
def _(results):
    _names = [r["name"] for r in results]
    _x = np.arange(len(_names))

    fig_summary, _axes = plt.subplots(1, 3, figsize=(12, 4))

    _correlations = [r["spearman_correlation"] for r in results]
    _axes[0].bar(_x, _correlations, color="#3498db", edgecolor="none")
    _axes[0].set_xticks(_x)
    _axes[0].set_xticklabels(_names, rotation=45, ha="right", fontsize=8)
    _axes[0].set_ylabel("Spearman Correlation")
    _axes[0].set_title("Correlation with Tanimoto", fontweight="bold")
    _axes[0].set_ylim(0, 1)
    _axes[0].spines["top"].set_visible(False)
    _axes[0].spines["right"].set_visible(False)

    _recalls = [r.get("recall_at_10", 0) for r in results]
    _axes[1].bar(_x, _recalls, color="#2ecc71", edgecolor="none")
    _axes[1].set_xticks(_x)
    _axes[1].set_xticklabels(_names, rotation=45, ha="right", fontsize=8)
    _axes[1].set_ylabel("Recall@10")
    _axes[1].set_title("k-NN Recall (k=10)", fontweight="bold")
    _axes[1].set_ylim(0, 1)
    _axes[1].spines["top"].set_visible(False)
    _axes[1].spines["right"].set_visible(False)

    _times = [r["transform_time_full_s"] for r in results]
    _axes[2].bar(_x, _times, color="#e74c3c", edgecolor="none")
    _axes[2].set_xticks(_x)
    _axes[2].set_xticklabels(_names, rotation=45, ha="right", fontsize=8)
    _axes[2].set_ylabel("Time (seconds)")
    _axes[2].set_title("Transform Time (Full Dataset)", fontweight="bold")
    _axes[2].spines["top"].set_visible(False)
    _axes[2].spines["right"].set_visible(False)

    plt.tight_layout()
    fig_summary
    return (fig_summary,)


@app.cell
def _(X_full, mo, run_umap_input, seed_input):
    _run_umap = run_umap_input.value
    _seed = seed_input.value

    if _run_umap:
        import umap

        _umap_sample_size = 5000
        _rng = np.random.default_rng(_seed)
        if _umap_sample_size < len(X_full):
            _umap_idx = _rng.choice(len(X_full), size=_umap_sample_size, replace=False)
            _X_umap = X_full[_umap_idx]
        else:
            _X_umap = X_full

        logger.info("Running UMAP comparison (Jaccard vs Cosine on L2-normalized)...")

        logger.info("  Computing UMAP with Jaccard metric...")
        _reducer_jaccard = umap.UMAP(metric="jaccard", n_neighbors=15, min_dist=0.1, random_state=_seed)
        _embedding_jaccard = _reducer_jaccard.fit_transform(_X_umap)

        logger.info("  Computing UMAP with cosine on L2-normalized...")
        _X_l2 = transform_l2_normalize(_X_umap)
        _reducer_cosine = umap.UMAP(metric="cosine", n_neighbors=15, min_dist=0.1, random_state=_seed)
        _embedding_cosine = _reducer_cosine.fit_transform(_X_l2)

        fig_umap, _axes = plt.subplots(1, 2, figsize=(12, 5))

        _axes[0].scatter(
            _embedding_jaccard[:, 0],
            _embedding_jaccard[:, 1],
            s=1,
            alpha=0.3,
            c="#3498db",
            rasterized=True,
        )
        _axes[0].set_title("UMAP with Jaccard (Ground Truth)", fontsize=11, fontweight="bold")
        _axes[0].set_xlabel("UMAP 1")
        _axes[0].set_ylabel("UMAP 2")
        _axes[0].set_aspect("equal")

        _axes[1].scatter(
            _embedding_cosine[:, 0],
            _embedding_cosine[:, 1],
            s=1,
            alpha=0.3,
            c="#e74c3c",
            rasterized=True,
        )
        _axes[1].set_title("UMAP with Cosine (L2 Normalized)", fontsize=11, fontweight="bold")
        _axes[1].set_xlabel("UMAP 1")
        _axes[1].set_ylabel("UMAP 2")
        _axes[1].set_aspect("equal")

        fig_umap.suptitle(
            f"UMAP Comparison (n={len(_X_umap):,} compounds)",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
    else:
        fig_umap = None
        mo.md("*UMAP comparison disabled. Check the box above to enable.*")

    fig_umap
    return (fig_umap,)


@app.cell
def _(mo, results):
    _header = f"{'Method':<25} {'rho':>8} {'R@10':>8} {'R@50':>8} {'Time(s)':>10}"
    _sep = "-" * 65
    _rows = []
    for _r in sorted(results, key=lambda x: -x["spearman_correlation"]):
        _rows.append(
            f"{_r['name']:<25} "
            f"{_r['spearman_correlation']:>8.4f} "
            f"{_r.get('recall_at_10', 0):>8.4f} "
            f"{_r.get('recall_at_50', 0):>8.4f} "
            f"{_r['transform_time_full_s']:>10.3f}"
        )

    _best_corr = max(results, key=lambda x: x["spearman_correlation"])
    _best_recall = max(results, key=lambda x: x.get("recall_at_10", 0))
    _fastest = min(results, key=lambda x: x["transform_time_full_s"])

    mo.md(f"""
    ### Results Summary

    ```
    {_header}
    {_sep}
    {"    ".join("") + chr(10).join(_rows)}
    ```

    **Recommendations:**
    - Best correlation: **{_best_corr["name"]}** (rho = {_best_corr["spearman_correlation"]:.4f})
    - Best recall@10: **{_best_recall["name"]}** (R@10 = {_best_recall.get("recall_at_10", 0):.4f})
    - Fastest: **{_fastest["name"]}** ({_fastest["transform_time_full_s"]:.3f}s)
    """)
    return


@app.cell
def _(fig_correlation, fig_summary, fig_umap, mo, results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_correlation.savefig(
        OUTPUT_DIR / "fingerprint_metrics_correlation.png",
        dpi=DEFAULT_DPI,
        bbox_inches="tight",
    )
    logger.info("Saved correlation scatter plots")

    fig_summary.savefig(
        OUTPUT_DIR / "fingerprint_metrics_summary.png",
        dpi=DEFAULT_DPI,
        bbox_inches="tight",
    )
    logger.info("Saved results summary plot")

    if fig_umap is not None:
        fig_umap.savefig(
            OUTPUT_DIR / "fingerprint_metrics_umap_comparison.png",
            dpi=DEFAULT_DPI,
            bbox_inches="tight",
        )
        logger.info("Saved UMAP comparison plot")

    _results_file = OUTPUT_DIR / "fingerprint_metrics_results.json"
    with open(_results_file, "w") as _f:
        json.dump(results, _f, indent=2)
    logger.info(f"Saved results to {_results_file}")

    mo.md(f"Outputs saved to `{OUTPUT_DIR}`")
    return


@app.function
def run_fingerprint_metrics(output_dir: str | Path | None = None, sample_size: int = 3000) -> str:
    """Run the L2-norm cosine vs Tanimoto fingerprint metrics analysis.

    Evaluates multiple fingerprint transformations, computes Spearman
    correlations and k-NN recall against Tanimoto ground truth, and saves
    correlation plots, summary bar chart, results JSON, and a .complete
    marker.

    Parameters
    ----------
    output_dir : str | Path | None
        Output directory. Defaults to PROCESSED_DATA_DIR / "fingerprint-metrics".
    sample_size : int
        Number of compounds to subsample for pairwise accuracy evaluation.

    Returns
    -------
    str
        Path to the output directory.
    """
    if output_dir is None:
        _out = OUTPUT_DIR
    else:
        _out = Path(output_dir)
    _out.mkdir(parents=True, exist_ok=True)

    _seed = 42
    _k_values = [10, 50, 100]

    # Load fingerprints
    logger.info(f"Loading Morgan fingerprints from {MORGAN_FP_FILE}...")
    _fp_data = np.load(MORGAN_FP_FILE, allow_pickle=True)
    _X_full = _fp_data["fingerprints"].astype(np.float32)
    _valid_mask = _fp_data["valid_mask"]
    _X_full = _X_full[_valid_mask]
    logger.info(f"Loaded {_X_full.shape[0]:,} valid fingerprints, {_X_full.shape[1]} bits")

    # Subsample for accuracy evaluation
    _rng = np.random.default_rng(_seed)
    if sample_size < len(_X_full):
        _sample_idx = _rng.choice(len(_X_full), size=sample_size, replace=False)
        _X_sample = _X_full[_sample_idx]
    else:
        _X_sample = _X_full

    logger.info(f"Using {len(_X_sample):,} samples for accuracy evaluation")

    # Ground truth Tanimoto
    logger.info("Computing ground truth Tanimoto similarities...")
    _tanimoto_sim = tanimoto_similarity(_X_sample)
    _triu_idx = np.triu_indices(len(_X_sample), k=1)
    _tanimoto_flat = _tanimoto_sim[_triu_idx]

    # Define transformations
    _transformations = {
        "L2 Normalize": (transform_l2_normalize, "cosine"),
        "Popcount Normalize": (transform_popcount_normalize, "cosine"),
        "Concat Popcount": (transform_concatenate_popcount, "euclidean"),
        "IDF Weighted": (transform_idf_weighted, "cosine"),
        "Random Features": (
            lambda X: transform_tanimoto_random_features(X, n_features_out=512, seed=_seed),
            "cosine",
        ),
    }

    # Evaluate all transformations
    _results = []
    for _name, (_transform_fn, _metric) in _transformations.items():
        logger.info(f"Evaluating: {_name}")
        _result = evaluate_transformation(
            name=_name,
            transform_fn=_transform_fn,
            X_sample=_X_sample,
            X_full=_X_full,
            tanimoto_flat=_tanimoto_flat,
            k_values=_k_values,
            metric=_metric,
        )
        _results.append(_result)
        logger.info(
            f"  rho={_result['spearman_correlation']:.4f}  "
            f"R@10={_result.get('recall_at_10', 0):.4f}  "
            f"time={_result['transform_time_full_s']:.3f}s"
        )

    # Correlation scatter plots
    _tanimoto_dist_flat = 1.0 - _tanimoto_flat
    _n_pairs = 10000
    _rng2 = np.random.default_rng(42)
    if len(_tanimoto_dist_flat) > _n_pairs:
        _scatter_idx = _rng2.choice(len(_tanimoto_dist_flat), size=_n_pairs, replace=False)
    else:
        _scatter_idx = np.arange(len(_tanimoto_dist_flat))
    _tanimoto_sample = _tanimoto_dist_flat[_scatter_idx]

    _n_transforms = len(_transformations)
    _n_cols = 3
    _n_rows = (_n_transforms + _n_cols - 1) // _n_cols

    _fig_corr, _axes_corr = plt.subplots(_n_rows, _n_cols, figsize=(4 * _n_cols, 4 * _n_rows))
    _axes_corr = _axes_corr.flatten() if _n_transforms > 1 else [_axes_corr]

    for _ax, (_name, (_transform_fn, _metric)) in zip(_axes_corr, _transformations.items()):
        _X_t = _transform_fn(_X_sample)
        if _metric == "cosine":
            _sim = cosine_similarity(_X_t)
            _dist_flat = 1.0 - _sim[_triu_idx]
        else:
            _dist_flat = cdist(_X_t, _X_t, metric="euclidean")[_triu_idx]
        _dist_sample = _dist_flat[_scatter_idx]
        _corr, _ = spearmanr(_dist_sample, _tanimoto_sample)
        _ax.scatter(_tanimoto_sample, _dist_sample, s=1, alpha=0.1, rasterized=True)
        _ax.set_xlabel("Tanimoto Distance", fontsize=9)
        _ax.set_ylabel(f"{_metric.title()} Distance", fontsize=9)
        _ax.set_title(f"{_name}\nSpearman rho = {_corr:.3f}", fontsize=10)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

    for _ax in _axes_corr[_n_transforms:]:
        _ax.set_visible(False)
    plt.tight_layout()
    _fig_corr.savefig(
        _out / "fingerprint_metrics_correlation.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white"
    )
    plt.close(_fig_corr)

    # Summary bar chart
    _names = [r["name"] for r in _results]
    _x = np.arange(len(_names))
    _fig_summary, _axes_sum = plt.subplots(1, 3, figsize=(12, 4))

    _correlations = [r["spearman_correlation"] for r in _results]
    _axes_sum[0].bar(_x, _correlations, color="#3498db", edgecolor="none")
    _axes_sum[0].set_xticks(_x)
    _axes_sum[0].set_xticklabels(_names, rotation=45, ha="right", fontsize=8)
    _axes_sum[0].set_ylabel("Spearman Correlation")
    _axes_sum[0].set_title("Correlation with Tanimoto", fontweight="bold")
    _axes_sum[0].set_ylim(0, 1)
    _axes_sum[0].spines["top"].set_visible(False)
    _axes_sum[0].spines["right"].set_visible(False)

    _recalls = [r.get("recall_at_10", 0) for r in _results]
    _axes_sum[1].bar(_x, _recalls, color="#2ecc71", edgecolor="none")
    _axes_sum[1].set_xticks(_x)
    _axes_sum[1].set_xticklabels(_names, rotation=45, ha="right", fontsize=8)
    _axes_sum[1].set_ylabel("Recall@10")
    _axes_sum[1].set_title("k-NN Recall (k=10)", fontweight="bold")
    _axes_sum[1].set_ylim(0, 1)
    _axes_sum[1].spines["top"].set_visible(False)
    _axes_sum[1].spines["right"].set_visible(False)

    _times = [r["transform_time_full_s"] for r in _results]
    _axes_sum[2].bar(_x, _times, color="#e74c3c", edgecolor="none")
    _axes_sum[2].set_xticks(_x)
    _axes_sum[2].set_xticklabels(_names, rotation=45, ha="right", fontsize=8)
    _axes_sum[2].set_ylabel("Time (seconds)")
    _axes_sum[2].set_title("Transform Time (Full Dataset)", fontweight="bold")
    _axes_sum[2].spines["top"].set_visible(False)
    _axes_sum[2].spines["right"].set_visible(False)

    plt.tight_layout()
    _fig_summary.savefig(
        _out / "fingerprint_metrics_summary.png", dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white"
    )
    plt.close(_fig_summary)

    # Results JSON
    _results_file = _out / "fingerprint_metrics_results.json"
    with open(_results_file, "w") as _f:
        json.dump(_results, _f, indent=2)
    logger.info(f"Saved results to {_results_file}")

    # .complete marker
    (_out / ".complete").touch()

    logger.success(f"run_fingerprint_metrics: saved all outputs to {_out}")
    return str(_out)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
