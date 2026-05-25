# CLAUDE.md - jpx

See `AGENTS.md` for the shared project guidance.
This file exists so Claude Code discovers the same contract without duplicating it.
All project guidance edits go in `AGENTS.md`, not here; keep this file thin.

## Lineage

This repo was created as an orphan commit from `broadinstitute/jump_production` branch `marimo-redun-migration` (commit `ad21c1d`, 2026-05-25).
The full development history lives in jump_production.
Notebooks and scripts in jump_production may receive updates that should be ported here - check `broadinstitute/jump_production` for upstream changes to `notebooks/`, `workflow.py`, or `configs/`.

## Adding dependencies

- Add Python packages to `[project] dependencies` in `pyproject.toml`, not `[tool.pixi.dependencies]` (which is for conda packages).
- `copairs-runner` git dep uses monorepo subdirectory - must pin to a commit hash, not a branch name.

## Development tips

- When adding/modifying annotations, use `just redun augment_metadata_db`. Redun caches copairs outputs automatically.
- If database doesn't rebuild: delete it first: `rm data/interim/jump_metadata_augmented.duckdb && just redun augment_metadata_db`
- Every pixi env used by `_run_in_env()` must have `marimo` in its conda deps.
- Source exclusions: Source 7 (concentration difference), Source 15 (merged channels), Source 4/13 (compound-specific visualizations).
- Reproducibility: clean rebuilds produce bit-identical mAP values and base profiles. Only UMAP coordinates, permutation p-values, and ML model outputs differ (expected stochastic variation).
