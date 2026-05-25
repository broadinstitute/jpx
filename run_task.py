"""Invoke a marimo notebook @app.function from any pixi environment.

Usage:
    pixi run -e rapids python run_task.py nb40_ss_umap_pipeline compute_umap compound_no_source7 cosine all
"""

import sys
from pathlib import Path

notebooks_dir = str(Path(__file__).parent / "notebooks")
if notebooks_dir not in sys.path:
    sys.path.insert(0, notebooks_dir)

module = __import__(sys.argv[1])
func = getattr(module, sys.argv[2])
func(*sys.argv[3:])
