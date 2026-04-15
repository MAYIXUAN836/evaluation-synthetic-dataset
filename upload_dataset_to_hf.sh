#!/usr/bin/env bash
# Upload local Evaluation dataset folder to Hugging Face dataset repository.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${ROOT_DIR}/Dataset"
DATASET_HF_REPO="${DATASET_HF_REPO:-YixuanMa/evaluation-synthetic-dataset-data}"
HF_TOKEN="${HF_TOKEN:-}"
UPLOAD_NUM_WORKERS="${UPLOAD_NUM_WORKERS:-8}"
UPLOAD_REPORT_EVERY="${UPLOAD_REPORT_EVERY:-30}"

if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "❌ Dataset directory not found: ${DATASET_DIR}"
    exit 1
fi

if [[ -z "${HF_TOKEN}" ]]; then
    echo "ℹ️ HF_TOKEN not provided. Will use local huggingface-cli login session if available."
fi

echo "📤 Uploading ${DATASET_DIR} to Hugging Face dataset repo: ${DATASET_HF_REPO}"
echo "   Dataset size is large; upload may take hours."
echo "   Workers: ${UPLOAD_NUM_WORKERS}, report every: ${UPLOAD_REPORT_EVERY}s"

ROOT_DIR_ENV="${ROOT_DIR}" \
DATASET_HF_REPO_ENV="${DATASET_HF_REPO}" \
HF_TOKEN_ENV="${HF_TOKEN}" \
UPLOAD_NUM_WORKERS_ENV="${UPLOAD_NUM_WORKERS}" \
UPLOAD_REPORT_EVERY_ENV="${UPLOAD_REPORT_EVERY}" \
/home/projectx/miniconda/bin/python - <<'PYTHON_EOF'
import os
from pathlib import Path
from huggingface_hub import HfApi

root_dir = Path(os.environ["ROOT_DIR_ENV"])
repo_id = os.environ["DATASET_HF_REPO_ENV"]
token = os.environ.get("HF_TOKEN_ENV") or None
num_workers = int(os.environ.get("UPLOAD_NUM_WORKERS_ENV", "8"))
report_every = int(os.environ.get("UPLOAD_REPORT_EVERY_ENV", "30"))

api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

api.upload_large_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=str(root_dir),
    allow_patterns=["Dataset/**"],
    num_workers=num_workers,
    print_report=True,
    print_report_every=report_every,
)

print(f"[hf] upload completed: {repo_id}")
PYTHON_EOF

echo "✅ Upload finished: ${DATASET_HF_REPO}"
