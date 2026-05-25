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
# All data is on a public S3 bucket (no credentials needed).
# =============================================================================

# ==================== PROJECT CONFIGURATION ====================

# Public S3 bucket (anonymous access, no credentials needed)
S3_BUCKET := "imaging-platform"
S3_PROJECT_PATH := "projects/cpg0042-chandrasekaran-jump/workspace/publication_data/2025_Chandrasekaran/jump_production_datastore"
S3_PREFIX := "s3://" + S3_BUCKET + "/" + S3_PROJECT_PATH

# Rclone configuration for public S3 (no credentials needed)
RCLONE_FLAGS := env_var_or_default("RCLONE_FLAGS", "--transfers 16 --checkers 16")
RCLONE_SYNC := "rclone sync " + RCLONE_FLAGS + " -v --stats-one-line --s3-provider=AWS --s3-region=us-east-1 --exclude '.DS_Store' --exclude '__pycache__/**' --exclude '*.pyc'"

# S5CMD configuration for listing
S5CMD_FLAGS := env_var_or_default("S5CMD_FLAGS", "--numworkers 16 --no-sign-request")
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

# ==================== UTILITIES ====================

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
