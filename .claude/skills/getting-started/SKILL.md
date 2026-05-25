---
name: getting-started
description: First-run setup for jpx. Installs prerequisites, downloads data, and launches a notebook. Use when someone says "help me get started", "set up this repo", "how do I run this", or opens the repo for the first time.
---

# Getting Started with jpx

## Prerequisites

- [pixi](https://pixi.sh) - multi-environment package manager (handles Python, CUDA, conda, and PyPI deps)
- [uv](https://docs.astral.sh/uv/) - Python package runner (provides `uvx` for launching marimo)
- [rclone](https://rclone.org) - for downloading data from public S3 (no credentials needed)

`just` and `marimo` are installed automatically by `pixi install`.
On NixOS, `pixi` is provided by the flake: `nix develop` or `direnv allow`.

## Setup steps

1. Install dependencies and activate the environment:

```bash
pixi install
pixi shell
```

`pixi shell` puts `just`, `marimo`, and other project tools on PATH.

2. Choose a data path:

**Fast path** - download pre-computed results (~12 GB):
```bash
just get-results
```

**Full path** - download inputs and run the pipeline (~32 GB inputs, ~13 min on 4x H100):
```bash
just get-inputs
just redun main
just redun batch_source7
just redun explore
just redun srijit_all
```

3. Launch a notebook to verify:

```bash
uvx marimo@latest edit --sandbox notebooks/nb16_ss_target_consistency.py
```

`uvx marimo@latest --sandbox` runs notebooks with their PEP 723 inline dependencies in an isolated env, independent of the pixi env.

4. Install third-party agent skills (marimo notebook authoring, paper implementation, etc.):

```bash
npx skills add marimo-team/skills --agent claude-code -y
npx skills add marimo-team/marimo-pair --agent claude-code -y
```

## What to explore first

- `nb16_ss_target_consistency` - target consistency volcano plots and UMAP overlays
- `nb14_ss_cross_source_reproducibility` - how well results replicate across data centers
- `nb19_ss_compound_traits` - which chemical properties predict morphological activity
- `nb22_ss_activity_consistency_relationship` - how single-compound activity relates to target-level consistency

Once set up, use the `compose-notebook` skill to compose new analyses from the catalog.

## GPU notebooks

Notebooks that need NVIDIA GPUs (nb08, nb09, nb17, nb20, nb25, nb38-nb40) use the rapids pixi environment:

```bash
pixi run -e rapids marimo edit notebooks/nb08_ss_umap_seed_scan.py
```

Do not use `--sandbox` with GPU notebooks - it shadows conda packages.

## Troubleshooting

- `ModuleNotFoundError` on launch: you forgot `--sandbox` (pure-Python) or used `--sandbox` with a GPU notebook
- Pipeline hangs at copairs: check `CUDA_VISIBLE_DEVICES` if GPU memory is contended
- `pixi install` fails on macOS: GPU features (rapids, chemberta, deepchem) are linux-only; the default env works on both platforms
