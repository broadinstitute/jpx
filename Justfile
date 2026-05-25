# jpx - JUMP Production eXplore
# Usage: just <command> [args]

set dotenv-load := true

# =============================================================================
# QUICK START:
#   just get-inputs           # Download input data from public S3
#   just redun main           # Run full pipeline
#   just redun explore        # Exploration notebooks
#   just redun srijit_all     # Srijit's analysis notebooks
#
# SKIP THE PIPELINE (use pre-computed results):
#   just get-results          # Download all results from public S3
#
# ADMIN (refresh from original sources):
#   just get-from-sources     # Download from original URLs (Pooch, hash-verified)
#
# All data is on a public S3 bucket (no credentials needed).
# =============================================================================

# ==================== PROJECT CONFIGURATION ====================

# Public S3 bucket (anonymous access, no credentials needed)
S3_BUCKET := "imaging-platform"
S3_PROJECT_PATH := "projects/cpg0042-chandrasekaran-jump/workspace/publication_data/2025_Chandrasekaran/jump_production_datastore"
S3_PREFIX := "s3://" + S3_BUCKET + "/" + S3_PROJECT_PATH

# Rclone configuration for public S3 (no credentials needed)
RCLONE_FLAGS := env("RCLONE_FLAGS", "--transfers 16 --checkers 16")
RCLONE_SYNC := "rclone sync " + RCLONE_FLAGS + " -v --stats-one-line --s3-provider=AWS --s3-region=us-east-1 --exclude '.DS_Store' --exclude '__pycache__/**' --exclude '*.pyc'"

# S5CMD configuration for listing
S5CMD_FLAGS := env("S5CMD_FLAGS", "--numworkers 16 --no-sign-request")
S5CMD := "s5cmd " + S5CMD_FLAGS

# Pipeline orchestration
REDUN := "pixi run redun run workflow.py"

# Directory names (following Cookiecutter Data Science structure)
DATA_DIR := "data"
EXTERNAL_DIR := DATA_DIR + "/external"
RAW_DIR := DATA_DIR + "/raw"
INTERIM_DIR := DATA_DIR + "/interim"
PROCESSED_DIR := DATA_DIR + "/processed"

# Default recipe (shows help)
default:
    @just --list

# ==================== MAIN WORKFLOW ====================

# Run a redun task (e.g., just redun core, just redun main, just redun explore)
redun *args:
    @{{REDUN}} {{args}}

# ==================== DATA SYNC ====================

# Download from original sources (ChEMBL, CellPainting Gallery, Zenodo, etc.)
# Hash-verified via Pooch. Equivalent to get-inputs but with full URL provenance.
get-from-sources:
    @echo "Downloading from original sources (Pooch, hash-verified)..."
    @pixi run python -c "import sys; sys.path.insert(0, 'notebooks'); from nb43_ss_download_data import download_all; download_all()"

# Download input data from public S3 (no credentials needed)
get-inputs:
    @echo "Downloading input data from S3..."
    @mkdir -p {{EXTERNAL_DIR}} {{RAW_DIR}}/profiles
    @echo "Syncing external/..."
    {{RCLONE_SYNC}} ":s3:{{S3_BUCKET}}/{{S3_PROJECT_PATH}}/external/" {{EXTERNAL_DIR}}/
    @echo "Syncing profiles/..."
    {{RCLONE_SYNC}} ":s3:{{S3_BUCKET}}/{{S3_PROJECT_PATH}}/profiles/" {{RAW_DIR}}/profiles/
    @echo "Done!"

# Download pre-computed results from public S3 (skip the pipeline)
get-results:
    @echo "Downloading pre-computed results from S3..."
    @mkdir -p {{INTERIM_DIR}} {{PROCESSED_DIR}}
    @echo "Syncing interim/..."
    {{RCLONE_SYNC}} ":s3:{{S3_BUCKET}}/{{S3_PROJECT_PATH}}/interim/" {{INTERIM_DIR}}/
    @echo "Syncing processed/..."
    {{RCLONE_SYNC}} ":s3:{{S3_BUCKET}}/{{S3_PROJECT_PATH}}/processed/" {{PROCESSED_DIR}}/
    @echo "Done!"

# Download specific results subdirectory
get-results-for run_path:
    @echo "Getting results for: {{run_path}}"
    @mkdir -p {{PROCESSED_DIR}}/{{run_path}}
    {{RCLONE_SYNC}} ":s3:{{S3_BUCKET}}/{{S3_PROJECT_PATH}}/processed/{{run_path}}/" {{PROCESSED_DIR}}/{{run_path}}/

# ==================== NOTEBOOK DEPS ====================

# Pin PEP 723 inline deps to resolved versions (marimo and python-dotenv stay unpinned)
lock-notebooks *args:
    python3 scripts/lock_notebooks.py {{args}} notebooks/nb*.py

# Strip version pins from PEP 723 inline deps
unlock-notebooks *args:
    python3 scripts/lock_notebooks.py --unlock {{args}} notebooks/nb*.py

# ==================== LINTING ====================

# Lint and format check (use `just lint --fix` to auto-fix)
lint *args:
    pixi run ruff check {{args}} notebooks/ scripts/ workflow.py run_task.py
    pixi run ruff format --check notebooks/ scripts/ workflow.py run_task.py

# Auto-format all Python files
fmt:
    pixi run ruff format notebooks/ scripts/ workflow.py run_task.py
    pixi run ruff check --fix notebooks/ scripts/ workflow.py run_task.py

# ==================== UTILITIES ====================

# Delete all pipeline outputs (interim + processed). Inputs are untouched.
[confirm("This will delete all files in data/interim/ and data/processed/. Continue?")]
clean:
    @if command -v trash >/dev/null 2>&1; then \
        trash {{INTERIM_DIR}}/* {{PROCESSED_DIR}}/*; \
    else \
        rm -rf {{INTERIM_DIR}}/* {{PROCESSED_DIR}}/*; \
    fi
    @echo "Done! (data/external/ and data/raw/ are untouched)"

# See pipeline status (redun execution log)
status:
    @echo "Pipeline status:"
    @pixi run redun log

# Show current configuration
config:
    @echo "Current Configuration:"
    @echo "  S3_BUCKET: {{S3_BUCKET}}"
    @echo "  S3_PROJECT_PATH: {{S3_PROJECT_PATH}}"
    @echo "  S3_PREFIX: {{S3_PREFIX}}"
    @echo "  Data Directories:"
    @echo "    - External: {{EXTERNAL_DIR}}"
    @echo "    - Raw: {{RAW_DIR}}"
    @echo "    - Interim: {{INTERIM_DIR}}"
    @echo "    - Processed: {{PROCESSED_DIR}}"

# List what's in S3
list-s3:
    @echo "Listing S3 contents:"
    @echo "\nExternals:"
    @{{S5CMD}} ls {{S3_PREFIX}}/external/ | head -5
    @echo "\nInterim:"
    @{{S5CMD}} ls {{S3_PREFIX}}/interim/ | head -5
    @echo "\nProcessed:"
    @{{S5CMD}} ls {{S3_PREFIX}}/processed/ | head -5
    @echo "\n(showing first 5 items per category)"
