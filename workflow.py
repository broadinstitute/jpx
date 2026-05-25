"""jpx pipeline - redun workflow.

Processing logic lives in marimo notebooks (nb32-nb42); analysis tasks
import from existing notebooks (nb01-nb31).

Usage:
    pixi run redun run workflow.py core           # Database + copairs
    pixi run redun run workflow.py main            # Full pipeline
    pixi run redun run workflow.py explore         # Phase 0 explorations
    pixi run redun run workflow.py srijit          # Srijit's analyses
"""

import itertools
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import yaml
from omegaconf import OmegaConf
from redun import File, task

redun_namespace = "jpx"

PROJ_ROOT = Path(__file__).parent
NOTEBOOKS_DIR = PROJ_ROOT / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

_NUM_GPUS = int(os.environ.get("NUM_GPUS", 4))
_gpu_cycle = itertools.cycle(range(_NUM_GPUS))
_gpu_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Section 1: Configuration and matrix expansion
# ---------------------------------------------------------------------------

DEFAULT_DATASET = "compound_no_source7"
DEFAULT_PREPROCESSING = "activity_no_target2"
DEFAULT_FILTER = "all_sources"
DEFAULT_ACTIVITY_PARAMS = "default"

COMPARISON_PROFILES_ACTIVITY = [
    ("compound_no_source7", "compound_DL_CPCNN_no_source7"),
]
COMPARISON_PROFILES_CONSISTENCY = [
    ("compound_no_source7_active_union", "compound_DL_CPCNN_no_source7_active_union"),
]

TARGET_CONSISTENCY_DATASETS = [
    "compound_no_source7",
    "compound_DL_CPCNN_no_source7",
    "compound_DL_CPCNN_with_source7",
    "compound_with_source7",
]

CROSS_SOURCE_COMBOS = [
    ("activity_only_target2", "all_sources", "activity_only_target2"),
    ("activity_only_target2", "no_source9", "activity_only_target2_no_source9"),
    ("activity_poscon_only", "all_sources", "activity_poscon_only"),
]

UMAP_COMBOS = [
    ("compound_no_source7", "cosine", "all"),
    ("compound_no_source7", "cosine", "active"),
    ("compound_no_source7", "euclidean", "all"),
    ("compound_DL_CPCNN_no_source7", "cosine", "all"),
    ("compound_with_source7", "cosine", "all"),
    ("compound_DL_CPCNN_with_source7", "cosine", "all"),
]


def _load_copairs_matrix():
    """Load and expand copairs run matrix from YAML config."""
    config_path = PROJ_ROOT / "configs" / "copairs" / "copairs_runs.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def _expand_consistency_matrix(matrix_config):
    """Expand dimensional consistency config into flat list."""
    if not isinstance(matrix_config, list):
        matrix_config = [matrix_config]
    results = []
    for item in matrix_config:
        if "dataset" in item:
            results.append(item)
            continue
        for dataset in item["datasets"]:
            for target in item["targets"]:
                for preproc in item.get("preprocessings", [{}]):
                    for distance in item.get("distances", ["cosine"]):
                        results.append(
                            {
                                "dataset": dataset,
                                "columns": item.get("columns", "feat_all"),
                                "preprocessing_02_core_cons": preproc.get("cons"),
                                "preprocessing_02_core_act": preproc.get("act"),
                                "filter": preproc.get("filter"),
                                "target_column": target["column"] if isinstance(target, dict) else target,
                                "target_table": target.get("table") if isinstance(target, dict) else None,
                                "distance": distance,
                            }
                        )
    return results


def _expand_activity_matrix(matrix_config):
    """Expand dimensional activity config into flat list."""
    if not isinstance(matrix_config, list):
        matrix_config = [matrix_config]
    results = []
    for item in matrix_config:
        if "dataset" in item:
            results.append(item)
            continue
        for dataset in item["datasets"]:
            for preproc in item["preprocessings"]:
                results.append(
                    {
                        "dataset": dataset,
                        "columns": item.get("columns", "feat_all"),
                        "preprocessing_02_core": preproc["preprocessing"],
                        "filter": preproc["filter"],
                    }
                )
    return results


def _expand_prefiltered_matrix(matrix_config):
    """Expand prefiltered consistency config into flat list."""
    if not isinstance(matrix_config, list):
        matrix_config = [matrix_config]
    results = []
    for item in matrix_config:
        if "dataset" in item:
            results.append(item)
            continue
        for dataset in item["datasets"]:
            for target in item["targets"]:
                results.append(
                    {
                        "dataset": dataset,
                        "columns": item.get("columns", "feat_all"),
                        "preprocessing_02_core_cons": item.get("preprocessing_02_core_cons", "consistency_prefiltered"),
                        "filter": item.get("filter", "all_sources"),
                        "target_column": target,
                        "distance": item.get("distance", "cosine"),
                    }
                )
    return results


def _expand_sweep_matrix(matrix_config):
    """Expand consistency sweep config with threshold dimension."""
    if not isinstance(matrix_config, list):
        matrix_config = [matrix_config]
    results = []
    for item in matrix_config:
        for dataset in item["datasets"]:
            for target in item["targets"]:
                for preproc in item.get("preprocessings", [{}]):
                    for distance in item.get("distances", ["cosine"]):
                        for threshold in item["thresholds"]:
                            results.append(
                                {
                                    "dataset": dataset,
                                    "columns": item.get("columns", "feat_all"),
                                    "preprocessing_02_core_cons": preproc.get("cons"),
                                    "preprocessing_02_core_act": preproc.get("act"),
                                    "filter": preproc.get("filter"),
                                    "target_column": target["column"] if isinstance(target, dict) else target,
                                    "target_table": target.get("table") if isinstance(target, dict) else None,
                                    "distance": distance,
                                    "activity_threshold": threshold,
                                }
                            )
    return results


def _activity_path(item, activity_params="default"):
    """Build activity output path."""
    name = f"{item['dataset']}__{item['columns']}__{item['preprocessing_02_core']}__{item['filter']}"
    return f"data/processed/copairs/runs/activity/{name}__{activity_params}/results/activity_map_results.csv"


def _consistency_path(item):
    """Build consistency output path."""
    name = (
        f"{item['dataset']}__{item['columns']}__{item['preprocessing_02_core_cons']}"
        f"__{item['filter']}__{item['target_column']}__{item['distance']}"
    )
    return f"data/processed/copairs/runs/consistency/{name}/results/consistency_map_results.csv"


def _run_in_env(env, notebook_module, function_name, *args):
    """Run a notebook function in a specific pixi environment."""
    cmd = [
        "pixi",
        "run",
        "-e",
        env,
        "python",
        str(PROJ_ROOT / "run_task.py"),
        notebook_module,
        function_name,
        *[str(a) for a in args],
    ]
    run_env = {**os.environ, "OMP_NUM_THREADS": "8"}
    if env == "rapids":
        with _gpu_lock:
            gpu_id = next(_gpu_cycle)
        run_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(cmd, check=True, cwd=str(PROJ_ROOT), env=run_env)


# ---------------------------------------------------------------------------
# Section 2: Processing tasks
# ---------------------------------------------------------------------------


@task()
def process_chembl() -> File:
    from nb33_ss_annotation_processing import process_chembl

    output = process_chembl()
    return File(str(output))


@task()
def process_repurposing_hub() -> File:
    from nb33_ss_annotation_processing import process_repurposing_hub

    output = process_repurposing_hub()
    return File(str(output))


@task()
def process_motive(db: File) -> File:
    from nb33_ss_annotation_processing import process_motive

    output = process_motive(db_path=str(db.path))
    return File(str(output))


@task()
def process_chemical_probes() -> File:
    _run_in_env("jump-smiles", "nb34_ss_probe_curation", "process_chemical_probes")
    return File("data/interim/chemical_probes_processed.csv")


@task()
def process_kinase_probes() -> File:
    _run_in_env("cheminformatics", "nb34_ss_probe_curation", "process_kinase_probes")
    return File("data/interim/kinase_probes.csv")


@task()
def process_toxcast() -> File:
    from nb35_ss_toxcast import process_toxcast

    output = process_toxcast()
    return File(str(output))


@task()
def process_toxicity_pk() -> File:
    _run_in_env("cheminformatics", "nb36_ss_toxicity_annotations", "process_toxicity_pk")
    return File("data/interim/toxicity_pk_processed.csv")


@task()
def process_mitotox() -> File:
    _run_in_env("cheminformatics", "nb36_ss_toxicity_annotations", "process_mitotox")
    return File("data/interim/mitotox_processed.csv")


@task()
def compute_compound_properties() -> File:
    _run_in_env("cheminformatics", "nb37_ss_compound_featurization", "compute_properties")
    return File("data/interim/compound_featurization/properties.parquet")


@task()
def compute_morgan_fingerprints() -> File:
    _run_in_env("cheminformatics", "nb37_ss_compound_featurization", "compute_morgan")
    return File("data/interim/compound_featurization/morgan_fp.npz")


@task()
def compute_chemberta_embeddings() -> File:
    _run_in_env("chemberta", "nb37_ss_compound_featurization", "compute_chemberta")
    return File("data/interim/compound_featurization/chemberta_77m_mlm.npz")


@task()
def fetch_image_dimensions() -> File:
    output_path = PROJ_ROOT / "data" / "interim" / "image_dimensions.csv"
    with open(output_path, "w") as f:
        subprocess.run(
            ["./scripts/fetch_image_dimensions.sh"],
            check=True,
            cwd=str(PROJ_ROOT),
            stdout=f,
        )
    return File(str(output_path))


# ---------------------------------------------------------------------------
# Section 3: Database augmentation
# ---------------------------------------------------------------------------


@task()
def augment_metadata_db(
    chembl: File,
    repurposing: File,
    probes: File,
    kinase: File,
    motive: File,
    toxcast: File,
    toxicity_pk: File,
    mitotox: File,
    properties: File,
    image_dims: File,
) -> File:
    """Copy base DuckDB and run SQL augmentation script."""
    base_db = PROJ_ROOT / "data" / "external" / "jump_metadata.duckdb"
    output = PROJ_ROOT / "data" / "interim" / "jump_metadata_augmented.duckdb"
    sql_script = PROJ_ROOT / "scripts" / "augment_metadata_db.sql"

    shutil.copy2(base_db, output)
    subprocess.run(
        ["duckdb", str(output), f".read {sql_script}"],
        check=True,
        cwd=str(PROJ_ROOT),
    )
    return File(str(output))


# ---------------------------------------------------------------------------
# Section 4: Profile tasks (anndata, batch correction, filtering)
# ---------------------------------------------------------------------------


@task()
def profiles_to_anndata(dataset: str) -> list[File]:
    _run_in_env(
        "rapids",
        "nb39_ss_profile_conversion",
        "convert_profiles",
        dataset,
    )
    return [
        File(f"data/interim/anndata/{dataset}.h5ad"),
        File(f"data/interim/anndata/{dataset}_perturbation.h5ad"),
    ]


@task()
def batch_correct(dataset: str, variant: str, db: File, config_path: str = "") -> File:
    args = [dataset, variant, str(db.path)]
    if config_path:
        args.append(config_path)
    _run_in_env(
        "rapids",
        "nb38_ss_batch_correction",
        "run_batch_correct",
        *args,
    )
    return File(f"data/raw/profiles/{dataset}_{variant}.parquet")


@task()
def filter_source7_exclusive(dataset: str, db: File) -> File:
    from nb42_ss_profile_filtering import filter_source7_exclusive as _filter

    output = _filter(
        input_parquet=str(PROJ_ROOT / "data" / "raw" / "profiles" / f"{dataset}.parquet"),
        db_path=str(db.path),
    )
    return File(str(output))


@task()
def generate_derived_profiles(db: File) -> list[File]:
    """Generate batch-corrected and source_7-filtered profile variants."""
    source_config = str(PROJ_ROOT / "configs" / "batch_correction_source.yaml")
    results = []
    for dataset in ["compound_no_source7", "compound_DL_CPCNN_no_source7"]:
        results.append(batch_correct(dataset, "rsc", db))
        results.append(batch_correct(dataset, "rsc_source", db, config_path=source_config))
    for dataset in ["compound", "compound_DL_CPCNN"]:
        results.append(filter_source7_exclusive(dataset, db))
    return results


@task()
def create_union_profiles(
    activity1: File,
    activity2: File,
    parquet1: File,
    parquet2: File,
    suffix: str = "active_union",
) -> list[File]:
    from nb42_ss_profile_filtering import create_union_profiles as _create

    outputs = _create(
        activity1=str(activity1.path),
        activity2=str(activity2.path),
        parquet1=str(parquet1.path),
        parquet2=str(parquet2.path),
        suffix=suffix,
    )
    return [File(str(p)) for p in outputs]


# ---------------------------------------------------------------------------
# Section 5: Copairs tasks (Hydra compose + CopairsRunner API)
# ---------------------------------------------------------------------------


_hydra_lock = threading.Lock()


def _compose_copairs_config(config_name, overrides):
    """Compose a Hydra config for CopairsRunner."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str((PROJ_ROOT / "configs" / "copairs" / "metrics").resolve())
    with _hydra_lock:
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _set_copairs_output_dir(cfg, output_dir):
    """Manually set output.directory since ${hydra:runtime.output_dir} isn't available."""
    OmegaConf.set_struct(cfg, False)
    cfg.output.directory = str(output_dir)
    OmegaConf.set_struct(cfg, True)


@task()
def copairs_activity(
    dataset: str,
    columns: str,
    preprocessing: str,
    filter_name: str,
    activity_params: str = "default",
    db: File = None,
) -> File:
    from copairs_runner import CopairsRunner

    cfg = _compose_copairs_config(
        "activity",
        [
            f"dataset={dataset}",
            f"columns={columns}",
            f"preprocessing_02_core={preprocessing}",
            f"filter={filter_name}",
            f"activity_params={activity_params}",
        ],
    )
    output_dir = (
        PROJ_ROOT
        / "data"
        / "processed"
        / "copairs"
        / "runs"
        / "activity"
        / f"{dataset}__{columns}__{preprocessing}__{filter_name}__{activity_params}"
        / "results"
    )
    _set_copairs_output_dir(cfg, output_dir)
    os.environ["COPAIRS_OUTPUT"] = str(PROJ_ROOT)

    runner = CopairsRunner(cfg)
    results = runner.run()
    runner.save_results(results, "activity")

    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / "config.yaml")

    return File(str(output_dir / "activity_map_results.csv"))


@task()
def copairs_consistency(
    dataset: str,
    columns: str,
    preprocessing_cons: str,
    filter_name: str,
    target_column: str,
    distance: str,
    db: File = None,
    activity_result: File = None,
) -> File:
    from copairs_runner import CopairsRunner

    cfg = _compose_copairs_config(
        "consistency",
        [
            f"dataset={dataset}",
            f"columns={columns}",
            f"preprocessing_02_core={preprocessing_cons}",
            f"filter={filter_name}",
            f"target_column={target_column}",
            f"distance={distance}",
        ],
    )
    output_dir = (
        PROJ_ROOT
        / "data"
        / "processed"
        / "copairs"
        / "runs"
        / "consistency"
        / f"{dataset}__{columns}__{preprocessing_cons}__{filter_name}__{target_column}__{distance}"
        / "results"
    )
    _set_copairs_output_dir(cfg, output_dir)
    os.environ["COPAIRS_OUTPUT"] = str(PROJ_ROOT)

    runner = CopairsRunner(cfg)
    results = runner.run()
    runner.save_results(results, "consistency")

    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / "config.yaml")

    return File(str(output_dir / "consistency_map_results.csv"))


@task()
def copairs_consistency_sweep(
    dataset: str,
    columns: str,
    preprocessing_cons: str,
    filter_name: str,
    target_column: str,
    distance: str,
    activity_threshold: float,
    db: File = None,
    activity_result: File = None,
) -> File:
    from copairs_runner import CopairsRunner

    threshold_str = f"{activity_threshold:.2f}"
    cfg = _compose_copairs_config(
        "consistency",
        [
            f"dataset={dataset}",
            f"columns={columns}",
            f"preprocessing_02_core={preprocessing_cons}",
            f"filter={filter_name}",
            f"target_column={target_column}",
            f"distance={distance}",
            f"preprocessing_02_core.activity_threshold={threshold_str}",
        ],
    )
    output_dir = (
        PROJ_ROOT
        / "data"
        / "processed"
        / "copairs"
        / "runs"
        / "consistency"
        / f"{dataset}__{columns}__{preprocessing_cons}__{filter_name}__{target_column}__{distance}__{threshold_str}"
        / "results"
    )
    _set_copairs_output_dir(cfg, output_dir)
    os.environ["COPAIRS_OUTPUT"] = str(PROJ_ROOT)

    runner = CopairsRunner(cfg)
    results = runner.run()
    runner.save_results(results, "consistency")

    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / "config.yaml")

    return File(str(output_dir / "consistency_map_results.csv"))


@task()
def build_copairs_results_db(activity_files: list[File], consistency_files: list[File]) -> File:
    from nb41_ss_copairs_db import build_results_db

    output = build_results_db()
    return File(str(output))


@task()
def all_activity(db: File) -> list[File]:
    """Run all activity analyses."""
    config = _load_copairs_matrix()
    results = []

    for item in config["activity"]:
        results.append(
            copairs_activity(
                dataset=item["dataset"],
                columns=item["columns"],
                preprocessing=item["preprocessing_02_core"],
                filter_name=item["filter"],
                db=db,
            )
        )

    for item in _expand_activity_matrix(config.get("activity_withinsource", [])):
        results.append(
            copairs_activity(
                dataset=item["dataset"],
                columns=item["columns"],
                preprocessing=item["preprocessing_02_core"],
                filter_name=item["filter"],
                activity_params="withinsource",
                db=db,
            )
        )

    for item in _expand_activity_matrix(config.get("activity_crosssource", [])):
        results.append(
            copairs_activity(
                dataset=item["dataset"],
                columns=item["columns"],
                preprocessing=item["preprocessing_02_core"],
                filter_name=item["filter"],
                activity_params="crosssource",
                db=db,
            )
        )

    return results


@task()
def all_union_profiles(activity_results: list[File]) -> list[File]:
    """Create union profiles from activity results."""

    def _find_activity(dataset, columns, preprocessing, filter_name, params="default"):
        pattern = f"{dataset}__{columns}__{preprocessing}__{filter_name}__{params}"
        for f in activity_results:
            if pattern in str(f.path):
                return f
        return None

    outputs = []

    act_cp = _find_activity("compound_no_source7", "feat_all", "activity_no_target2", "all_sources")
    act_dl = _find_activity("compound_DL_CPCNN_no_source7", "feat_all", "activity_no_target2", "all_sources")
    if act_cp and act_dl:
        outputs.append(
            create_union_profiles(
                activity1=act_cp,
                activity2=act_dl,
                parquet1=File("data/raw/profiles/compound_no_source7.parquet"),
                parquet2=File("data/raw/profiles/compound_DL_CPCNN_no_source7.parquet"),
            )
        )

    act_cp_s7 = _find_activity("compound_with_source7", "feat_all", "activity_no_target2", "all_sources")
    act_dl_s7 = _find_activity("compound_DL_CPCNN_with_source7", "feat_all", "activity_no_target2", "all_sources")
    if act_cp_s7 and act_dl_s7:
        outputs.append(
            create_union_profiles(
                activity1=act_cp_s7,
                activity2=act_dl_s7,
                parquet1=File("data/raw/profiles/compound_with_source7.parquet"),
                parquet2=File("data/raw/profiles/compound_DL_CPCNN_with_source7.parquet"),
            )
        )

    return outputs


@task()
def all_consistency(db: File, activity_results: list[File], union_files: list[File] | None = None) -> list[File]:
    """Run all consistency analyses.

    union_files is accepted for redun dependency tracking - prefiltered
    consistency runs consume the union profiles it references.
    """
    config = _load_copairs_matrix()
    consistency_items = _expand_consistency_matrix(config["consistency"])
    results = []

    for item in consistency_items:
        is_prefiltered = item["dataset"].endswith("_active_union")
        act_result = None
        if not is_prefiltered and item.get("preprocessing_02_core_act"):
            act_path = (
                f"{item['dataset']}__{item['columns']}__{item['preprocessing_02_core_act']}__{item['filter']}__default"
            )
            for f in activity_results:
                if act_path in str(f.path):
                    act_result = f
                    break

        results.append(
            copairs_consistency(
                dataset=item["dataset"],
                columns=item["columns"],
                preprocessing_cons=item["preprocessing_02_core_cons"],
                filter_name=item["filter"],
                target_column=item["target_column"],
                distance=item["distance"],
                db=db,
                activity_result=act_result,
            )
        )

    prefiltered_items = _expand_prefiltered_matrix(config.get("consistency_prefiltered", []))
    for item in prefiltered_items:
        results.append(
            copairs_consistency(
                dataset=item["dataset"],
                columns=item["columns"],
                preprocessing_cons=item["preprocessing_02_core_cons"],
                filter_name=item["filter"],
                target_column=item["target_column"],
                distance=item["distance"],
                db=db,
            )
        )

    return results


@task()
def all_consistency_sweep(db: File, activity_results: list[File]) -> list[File]:
    """Run all consistency sweep analyses."""
    config = _load_copairs_matrix()
    sweep_items = _expand_sweep_matrix(config.get("consistency_sweep", []))
    results = []

    for item in sweep_items:
        act_result = None
        if item.get("preprocessing_02_core_act"):
            act_path = (
                f"{item['dataset']}__{item['columns']}__{item['preprocessing_02_core_act']}__{item['filter']}__default"
            )
            for f in activity_results:
                if act_path in str(f.path):
                    act_result = f
                    break

        results.append(
            copairs_consistency_sweep(
                dataset=item["dataset"],
                columns=item["columns"],
                preprocessing_cons=item["preprocessing_02_core_cons"],
                filter_name=item["filter"],
                target_column=item["target_column"],
                distance=item["distance"],
                activity_threshold=item["activity_threshold"],
                db=db,
                activity_result=act_result,
            )
        )

    return results


@task()
def all_copairs(db: File, derived_profiles: list[File] = None) -> File:
    """Run all copairs analyses and build results database."""
    activity_results = all_activity(db)
    union_files = all_union_profiles(activity_results)
    consistency_results = all_consistency(db, activity_results, union_files)
    sweep_results = all_consistency_sweep(db, activity_results)

    return build_copairs_results_db(
        activity_files=activity_results,
        consistency_files=consistency_results + sweep_results,
    )


# ---------------------------------------------------------------------------
# Section 6: UMAP tasks
# ---------------------------------------------------------------------------


@task()
def umap_compute(dataset: str, metric: str, filter_type: str, db: File = None, h5ad: File = None) -> list[File]:
    _run_in_env(
        "rapids",
        "nb40_ss_umap_pipeline",
        "compute_umap",
        dataset,
        metric,
        filter_type,
    )
    suffix = f"{metric}_{filter_type}_umap"
    return [
        File(f"data/interim/anndata/{dataset}_{suffix}.h5ad"),
        File(f"data/interim/anndata/{dataset}_perturbation_{suffix}.h5ad"),
    ]


@task()
def umap_plot(dataset: str, metric: str, filter_type: str, umap_h5ad: File = None) -> list[File]:
    from nb13_ss_umap_visualization import plot_umap_combined

    outdir = str(PROJ_ROOT / "data" / "processed" / "umap" / dataset)
    outputs = plot_umap_combined(dataset, metric, filter_type, output_dir=outdir)
    return [File(str(p)) for p in outputs]


@task()
def all_umap(db: File, copairs_db: File) -> list[File]:
    anndata_cache = {}
    results = []
    for dataset, metric, filter_type in UMAP_COMBOS:
        base_dataset = dataset.replace("_active_union", "")
        if base_dataset not in anndata_cache:
            anndata_cache[base_dataset] = profiles_to_anndata(base_dataset)
        h5ad = anndata_cache[base_dataset]
        compute_result = umap_compute(dataset, metric, filter_type, db=db, h5ad=h5ad)
        results.append(compute_result)
        results.append(umap_plot(dataset, metric, filter_type, umap_h5ad=compute_result))
    return results


# ---------------------------------------------------------------------------
# Section 7: Exploration tasks
# ---------------------------------------------------------------------------


@task()
def explore_cosine_comparison(dataset: str = DEFAULT_DATASET) -> File:
    from nb00_ss_config import DEFAULT_DPI
    from nb01_ss_cosine_comparison import plot_cosine_comparison, query_cosine_comparison

    df = query_cosine_comparison(dataset)
    fig = plot_cosine_comparison(df, dataset)
    outdir = PROJ_ROOT / "data" / "processed" / "exploration" / "0.01" / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "cosine_vs_abs_cosine_scatter.png"
    fig.savefig(outpath, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    return File(str(outpath))


@task()
def explore_threshold_sweep(dataset: str = DEFAULT_DATASET) -> list[File]:
    from nb00_ss_config import DEFAULT_DPI
    from nb05_ss_threshold_sweep import plot_threshold_summary, plot_threshold_tradeoff, query_threshold_sweep

    df = query_threshold_sweep(dataset)
    outdir = PROJ_ROOT / "data" / "processed" / "exploration" / "0.05" / dataset
    outdir.mkdir(parents=True, exist_ok=True)

    fig1 = plot_threshold_summary(df)
    path1 = outdir / "threshold_sweep_summary.png"
    fig1.savefig(path1, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")

    fig2 = plot_threshold_tradeoff(df)
    path2 = outdir / "threshold_sweep_tradeoff.png"
    fig2.savefig(path2, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    return [File(str(path1)), File(str(path2))]


@task()
def explore_stratification() -> File:
    _run_in_env("deepchem", "nb06_ss_stratification", "run_stratification_demo")
    outdir = PROJ_ROOT / "data" / "processed" / "exploration" / "0.02"
    return File(str(outdir / ".complete"))


@task()
def explore_umap_seed_scan(dataset: str = DEFAULT_DATASET) -> File:
    _run_in_env("rapids", "nb08_ss_umap_seed_scan", "run_seed_scan", dataset)
    outdir = PROJ_ROOT / "data" / "processed" / "exploration" / "0.04" / dataset
    return File(str(outdir / "seed_comparison.png"))


@task()
def explore_circos() -> list[File]:
    from nb07_nc_circos import run_circos

    outputs = run_circos()
    return [File(str(p)) for p in outputs]


@task()
def explore_activity_consistency(dataset: str = DEFAULT_DATASET) -> File:
    from nb00_ss_config import DEFAULT_DPI
    from nb22_ss_activity_consistency_relationship import get_activity_consistency_data, plot_activity_vs_consistency

    # 0.05 is the lowest sweep threshold - gives the broadest landscape view.
    # The interactive notebook defaults to 0.30 for a stricter filtered view.
    df = get_activity_consistency_data(dataset, activity_threshold=0.05)
    outdir = PROJ_ROOT / "data" / "processed" / "exploration" / "0.06" / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plot_activity_vs_consistency(df)
    outpath = outdir / "activity_vs_consistency.png"
    fig.savefig(outpath, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    return File(str(outpath))


@task()
def explore_power_analysis(dataset: str = DEFAULT_DATASET) -> File:
    from nb00_ss_config import DEFAULT_DPI
    from nb23_ss_power_analysis import get_compound_power_data, plot_power_analysis

    df = get_compound_power_data(dataset)
    outdir = PROJ_ROOT / "data" / "processed" / "exploration" / "0.07" / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plot_power_analysis(df)
    outpath = outdir / "power_analysis.png"
    fig.savefig(outpath, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    return File(str(outpath))


# ---------------------------------------------------------------------------
# Section 8: Analysis tasks
# ---------------------------------------------------------------------------


@task()
def target_consistency(
    dataset: str,
    group_type: str,
    umap_filter: str,
    distance: str,
) -> list[File]:
    from nb16_ss_target_consistency import run_target_consistency

    outdir = PROJ_ROOT / "data" / "processed" / "target-consistency" / dataset / group_type / distance
    outputs = run_target_consistency(
        dataset=dataset,
        group_type=group_type,
        umap_filter=umap_filter,
        distance=distance,
        output_dir=str(outdir),
    )
    return [File(str(p)) for p in outputs]


@task()
def all_target_consistency(db: File, copairs_db: File, umap_files: File = None) -> list[File]:
    config = _load_copairs_matrix()
    consistency_items = _expand_consistency_matrix(config["consistency"])
    valid_tuples = {
        (item["dataset"], item["target_column"], item["distance"])
        for item in consistency_items
        if item["dataset"] in TARGET_CONSISTENCY_DATASETS
    }

    active_umap_datasets = {d for d, m, f in UMAP_COMBOS if f == "active" and m == "cosine"}
    results = []
    for dataset, group_type, distance in valid_tuples:
        results.append(target_consistency(dataset, group_type, "all", distance))
        if dataset in active_umap_datasets:
            results.append(target_consistency(dataset, group_type, "active", distance))
    return results


@task()
def profile_comparison(copairs_db: File) -> File:
    from nb10_jfh_profile_comparison import run_profile_comparison

    outdir = PROJ_ROOT / "data" / "processed" / "profile-comparison"
    run_profile_comparison(output_dir=str(outdir))
    return File(str(outdir / ".complete"))


@task()
def data_quality(
    dataset: str = DEFAULT_DATASET,
    preprocessing: str = DEFAULT_PREPROCESSING,
    filter_name: str = DEFAULT_FILTER,
    activity_params: str = DEFAULT_ACTIVITY_PARAMS,
    copairs_db: File = None,
) -> File:
    from nb15_ss_cell_count_quality import run_data_quality

    outdir = PROJ_ROOT / "data" / "processed" / "data-quality" / dataset / preprocessing
    run_data_quality(
        dataset=dataset,
        preprocessing=preprocessing,
        filter_name=filter_name,
        activity_params=activity_params,
        output_dir=str(outdir),
    )
    return File(str(outdir / ".complete"))


@task()
def cross_source_reproducibility(dataset: str = DEFAULT_DATASET, copairs_db: File = None) -> list[File]:
    from nb14_ss_cross_source_reproducibility import run_cross_source

    results = []
    for preprocessing, filter_name, subdir in CROSS_SOURCE_COMBOS:
        outdir = PROJ_ROOT / "data" / "processed" / "cross-source-reproducibility" / dataset / subdir
        run_cross_source(
            dataset=dataset,
            preprocessing=preprocessing,
            filter_name=filter_name,
            output_dir=str(outdir),
        )
        results.append(File(str(outdir / "within_vs_cross_scatter.png")))
    return results


@task()
def chemical_space(morgan: File = None) -> File:
    _run_in_env("rapids", "nb17_ss_chemical_space", "run_chemical_space")
    return File("data/processed/chemical-space/.complete")


@task()
def compound_traits(
    dataset: str = DEFAULT_DATASET,
    preprocessing: str = DEFAULT_PREPROCESSING,
    copairs_db: File = None,
) -> File:
    from nb19_ss_compound_traits import run_compound_traits

    outdir = PROJ_ROOT / "data" / "processed" / "compound-traits" / dataset / preprocessing
    run_compound_traits(
        dataset=dataset,
        preprocessing=preprocessing,
        output_dir=str(outdir),
    )
    return File(str(outdir / ".complete"))


@task()
def structure_morphology(
    dataset: str = DEFAULT_DATASET,
    preprocessing: str = DEFAULT_PREPROCESSING,
    structure_rep: str = "morgan",
    fingerprints: File = None,
) -> File:
    _run_in_env(
        "rapids",
        "nb20_ss_structure_morphology",
        "run_structure_morphology",
        dataset,
        preprocessing,
        structure_rep,
    )
    outdir = PROJ_ROOT / "data" / "processed" / "structure-morphology" / dataset / preprocessing / structure_rep
    return File(str(outdir / ".complete"))


@task()
def all_structure_morphology(morgan: File = None, chemberta: File = None, h5ad: list[File] = None) -> list[File]:
    return [
        structure_morphology(structure_rep="morgan", fingerprints=morgan),
        structure_morphology(structure_rep="chemberta", fingerprints=chemberta),
    ]


@task()
def phenotype_prediction(
    dataset: str = DEFAULT_DATASET,
    preprocessing: str = DEFAULT_PREPROCESSING,
    structure_rep: str = "morgan",
    copairs_db: File = None,
    morgan: File = None,
) -> File:
    _run_in_env(
        "deepchem",
        "nb21_ss_phenotype_prediction",
        "run_phenotype_prediction",
        dataset,
        preprocessing,
        structure_rep,
    )
    outdir = PROJ_ROOT / "data" / "processed" / "phenotype-prediction" / dataset / preprocessing / structure_rep
    return File(str(outdir / ".complete"))


@task()
def fingerprint_metrics(morgan: File = None, chemberta: File = None) -> File:
    from nb18_ss_fingerprint_metrics import run_fingerprint_metrics

    outdir = PROJ_ROOT / "data" / "processed" / "fingerprint-metrics"
    run_fingerprint_metrics(output_dir=str(outdir))
    return File(str(outdir / ".complete"))


# ---------------------------------------------------------------------------
# Section 9: Batch & source7 analysis tasks
# ---------------------------------------------------------------------------


@task()
def source7_4way_comparison(copairs_db: File) -> File:
    from nb11_ss_source7_4way_comparison import compute_head_to_head, get_activity_summary

    outdir = PROJ_ROOT / "data" / "processed" / "source7-comparison"
    summary = get_activity_summary()
    h2h = compute_head_to_head()
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "activity_summary.csv", index=False)
    h2h.to_csv(outdir / "head_to_head.csv", index=False)
    return File(str(outdir / "head_to_head.csv"))


@task()
def batch_effect_quantification() -> File:
    _run_in_env("rapids", "nb09_ss_source7_batch", "run_batch_quantification")
    outdir = PROJ_ROOT / "data" / "processed" / "source7-comparison"
    return File(str(outdir / "batch_effect_summary.csv"))


@task()
def harmony_comparison(copairs_db: File) -> File:
    from nb12_ss_harmony_comparison import run_harmony_comparison

    outdir = PROJ_ROOT / "data" / "processed" / "source7-comparison"
    run_harmony_comparison(output_dir=str(outdir))
    return File(str(outdir / "summary.json"))


@task()
def batch_and_source7(copairs_db: File) -> list[File]:
    return [
        source7_4way_comparison(copairs_db),
        batch_effect_quantification(),
        harmony_comparison(copairs_db),
    ]


# ---------------------------------------------------------------------------
# Section 10: Srijit analysis tasks
# ---------------------------------------------------------------------------


@task()
def pains_prediction(dataset: str = DEFAULT_DATASET) -> File:
    from nb24_seal_pains_prediction import run_pains_prediction

    outdir = PROJ_ROOT / "data" / "processed" / "pains-prediction" / dataset
    run_pains_prediction(dataset=dataset, output_dir=str(outdir))
    return File(str(outdir / "summary.json"))


@task()
def activity_cliffs(dataset: str = DEFAULT_DATASET) -> File:
    _run_in_env("rapids", "nb25_seal_activity_cliffs", "run_activity_cliffs", dataset)
    outdir = PROJ_ROOT / "data" / "processed" / "activity-cliffs" / dataset
    return File(str(outdir / "summary.json"))


@task()
def phenoseeker_cliffs(dataset: str = DEFAULT_DATASET) -> File:
    from nb26_seal_phenoseeker_cliffs import run_phenoseeker

    outdir = PROJ_ROOT / "data" / "processed" / "activity-cliffs-phenoseeker" / dataset
    run_phenoseeker(dataset=dataset, output_dir=str(outdir))
    return File(str(outdir / "summary.json"))


@task()
def sar_vignette(
    dataset: str = DEFAULT_DATASET,
    preprocessing: str = DEFAULT_PREPROCESSING,
) -> File:
    _run_in_env(
        "cheminformatics",
        "nb27_seal_sar_vignette",
        "run_sar_vignette",
        dataset,
        preprocessing,
    )
    outdir = PROJ_ROOT / "data" / "processed" / "sar-vignette" / dataset
    return File(str(outdir / "summary.json"))


@task()
def toxicity_pk_prediction(dataset: str = DEFAULT_DATASET) -> File:
    _run_in_env(
        "cheminformatics",
        "nb28_seal_toxicity_pk",
        "run_toxicity_prediction",
        dataset,
    )
    outdir = PROJ_ROOT / "data" / "processed" / "toxicity-pk-prediction" / dataset
    return File(str(outdir / "summary.json"))


@task()
def commercial_compounds() -> File:
    _run_in_env(
        "cheminformatics",
        "nb29_seal_commercial_compounds",
        "run_commercial_analysis",
    )
    outdir = PROJ_ROOT / "data" / "processed" / "commercial-compounds"
    return File(str(outdir / "summary.json"))


@task()
def mitotox_morphology(dataset: str = DEFAULT_DATASET) -> File:
    _run_in_env(
        "cheminformatics",
        "nb30_seal_mitotox_morphology",
        "run_mitotox_analysis",
        dataset,
    )
    outdir = PROJ_ROOT / "data" / "processed" / "mitotox-analysis" / dataset
    return File(str(outdir / "summary.json"))


@task()
def mmp9_inhibitors() -> File:
    from nb31_seal_mmp9_inhibitors import run_mmp9_analysis

    outdir = PROJ_ROOT / "data" / "processed" / "mmp9-inhibitors" / DEFAULT_DATASET
    run_mmp9_analysis(output_dir=str(outdir))
    return File(str(outdir / "summary.json"))


# ---------------------------------------------------------------------------
# Section 11: Orchestration entry points
# ---------------------------------------------------------------------------


@task()
def processing() -> list[File]:
    """Run all external data processing (independent tasks)."""
    return [
        process_chembl(),
        process_repurposing_hub(),
        process_chemical_probes(),
        process_kinase_probes(),
        process_toxcast(),
        process_toxicity_pk(),
        process_mitotox(),
        compute_compound_properties(),
        compute_morgan_fingerprints(),
        compute_chemberta_embeddings(),
        fetch_image_dimensions(),
    ]


@task()
def core() -> list[File]:
    """Database + copairs. Replaces `just run`."""
    db = augment_metadata_db(
        chembl=process_chembl(),
        repurposing=process_repurposing_hub(),
        probes=process_chemical_probes(),
        kinase=process_kinase_probes(),
        motive=process_motive(db=File("data/external/jump_metadata.duckdb")),
        toxcast=process_toxcast(),
        toxicity_pk=process_toxicity_pk(),
        mitotox=process_mitotox(),
        properties=compute_compound_properties(),
        image_dims=fetch_image_dimensions(),
    )
    morgan = compute_morgan_fingerprints()
    chemberta = compute_chemberta_embeddings()
    derived_profiles = generate_derived_profiles(db)
    copairs_db = all_copairs(db, derived_profiles=derived_profiles)
    return [db, copairs_db, morgan, chemberta]


@task()
def main() -> list[File]:
    """Full pipeline. Replaces `just run-all`."""
    core_outputs = core()
    db = core_outputs[0]
    copairs_db = core_outputs[1]
    morgan = core_outputs[2]
    chemberta = core_outputs[3]

    umap_results = all_umap(db, copairs_db)

    return [
        db,
        copairs_db,
        profile_comparison(copairs_db),
        umap_results,
        all_target_consistency(db, copairs_db, umap_files=umap_results),
        data_quality(copairs_db=copairs_db),
        cross_source_reproducibility(copairs_db=copairs_db),
        chemical_space(morgan=morgan),
        compound_traits(copairs_db=copairs_db),
        all_structure_morphology(morgan=morgan, chemberta=chemberta, h5ad=umap_results),
        phenotype_prediction(copairs_db=copairs_db, morgan=morgan),
        fingerprint_metrics(morgan=morgan, chemberta=chemberta),
    ]


@task()
def explore() -> list[File]:
    """Phase 0 explorations. Replaces `just explore`."""
    return [
        explore_cosine_comparison(),
        explore_threshold_sweep(),
        explore_stratification(),
        explore_umap_seed_scan(),
        explore_circos(),
        explore_activity_consistency(),
        explore_power_analysis(),
    ]


@task()
def srijit_all() -> list[File]:
    """Srijit's analyses. Replaces `just srijit`."""
    return [
        pains_prediction(),
        activity_cliffs(),
        phenoseeker_cliffs(),
        sar_vignette(),
        toxicity_pk_prediction(),
        commercial_compounds(),
        mitotox_morphology(),
        mmp9_inhibitors(),
    ]


@task()
def batch_source7() -> list[File]:
    """Batch correction & source 7 analysis. Replaces `just batch-and-source7`."""
    copairs_db = File("data/processed/copairs_results.duckdb")
    return batch_and_source7(copairs_db)
