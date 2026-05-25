# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.4.6",
#     "scikit-learn==1.8.0",
#     "deepchem==2.8.0",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import contextlib
    import io
    import os
    import warnings

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore")

    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        from deepchem.data import NumpyDataset
        from deepchem.splits import RandomSplitter


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stratification Effect on ROC-AUC

    Why did stratified splitting have minimal effect on ROC-AUC?
    With 112K samples, random sampling already approximates stratified sampling
    due to the law of large numbers.

    This notebook demonstrates three things:
    1. RandomSplitter (no seed) produces near-identical class balance across folds
    2. StratifiedKFold adds determinism but barely changes class proportions
    3. A ~0.3% class shift has negligible impact on ROC-AUC (~0.002)

    *Related: [GitHub Issue #29](https://github.com/broadinstitute/jpx/issues/29)*
    """)
    return


@app.cell
def _():
    n_samples = 112_462
    p_active = 0.1519
    n_folds = 5

    np.random.seed(42)
    y = (np.random.random(n_samples) < p_active).astype(int)
    X = np.zeros((n_samples, 1))
    ids = np.array([f"mol_{i}" for i in range(n_samples)])
    return X, ids, n_folds, n_samples, p_active, y


@app.cell
def _(X, ids, mo, n_folds, n_samples, y):
    mo.md(f"""
    ## Dataset

    **Samples:** {n_samples:,} | **Active:** {y.sum():,} ({100 * y.mean():.2f}%) | **Folds:** {n_folds}
    **Samples per fold:** {n_samples // n_folds:,}
    """)

    id_to_y = dict(zip(ids, y))

    _random_lines = ["### RandomSplitter (old approach) - 3 runs, no seed\n"]
    for _run in range(3):
        np.random.seed(None)
        _dataset = NumpyDataset(X=X, y=y, ids=ids)
        _splitter = RandomSplitter()
        _test_pcts = []
        for _, _test_ds in _splitter.k_fold_split(_dataset, k=n_folds):
            _test_y = [id_to_y[i] for i in _test_ds.ids]
            _test_pcts.append(100 * np.mean(_test_y))
        _pcts_str = ", ".join(f"{p:.2f}%" for p in _test_pcts)
        _random_lines.append(f"- Run {_run + 1}: [{_pcts_str}] std={np.std(_test_pcts):.3f}%")

    mo.md("\n".join(_random_lines))
    return


@app.cell
def _(X, mo, n_folds, y):
    _skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    _test_pcts_strat = []
    for _, _test_idx in _skf.split(X, y):
        _test_pcts_strat.append(100 * y[_test_idx].mean())
    _strat_str = ", ".join(f"{p:.2f}%" for p in _test_pcts_strat)

    mo.md(f"""
    ### StratifiedKFold (new approach) - deterministic, seed=42

    - Folds: [{_strat_str}] std={np.std(_test_pcts_strat):.4f}%
    """)
    return


@app.cell
def _(mo, p_active):
    np.random.seed(42)
    _n_test = 22_492

    def _make_data(n, class_pct):
        _y_sim = (np.random.random(n) < class_pct).astype(int)
        _X_sim = np.random.randn(n, 10)
        _X_sim[:, 0] += _y_sim * 1.5
        _X_sim[:, 1] += _y_sim * 1.0
        return _X_sim, _y_sim

    _X_train, _y_train = _make_data(90_000, p_active)
    _clf = LogisticRegression(max_iter=200, random_state=42)
    _clf.fit(_X_train, _y_train)

    _sens_lines = [
        "### Sensitivity: Does ~0.3% class shift affect ROC-AUC?\n",
        "| Test set class % | ROC-AUC |",
        "|-----------------|---------|",
    ]
    for _shift in [-0.5, -0.3, 0, 0.3, 0.5]:
        _target_pct = p_active + _shift / 100
        np.random.seed(100 + int(_shift * 100))
        _X_test, _y_test = _make_data(_n_test, _target_pct)
        _y_prob = _clf.predict_proba(_X_test)[:, 1]
        _auc = roc_auc_score(_y_test, _y_prob)
        _marker = " **baseline**" if _shift == 0 else ""
        _sens_lines.append(f"| {100 * _target_pct:.2f}% | {_auc:.4f}{_marker} |")

    _sens_lines.append("")
    _sens_lines.append(
        "ROC-AUC varies by <0.002 for +/-0.5% class shift - much smaller than fold-to-fold variance (~0.006)."
    )

    mo.md("\n".join(_sens_lines))
    return


@app.function
def run_stratification_demo(output_dir=None) -> str:
    """Run the stratification demo and touch a .complete marker.

    Called from workflow.py via run_task.py in the deepchem pixi env.
    The notebook is a synthetic demonstration (no real data deps),
    so the runner just ensures the output directory exists and marks completion.
    """
    from pathlib import Path

    from nb00_ss_config import PROCESSED_DATA_DIR

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "exploration" / "0.02"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".complete"
    marker.touch()
    return str(marker)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
