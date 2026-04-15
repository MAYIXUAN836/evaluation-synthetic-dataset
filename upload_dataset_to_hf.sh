#!/usr/bin/env bash
# Upload local Evaluation dataset folder to Hugging Face dataset repository.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${ROOT_DIR}/Dataset"
DATASET_HF_REPO="${DATASET_HF_REPO:-YixuanMa/evaluation-synthetic-dataset-data}"
HF_TOKEN="${HF_TOKEN:-}"
UPLOAD_NUM_WORKERS="${UPLOAD_NUM_WORKERS:-8}"
UPLOAD_REPORT_EVERY="${UPLOAD_REPORT_EVERY:-30}"
UPLOAD_PHASED="${UPLOAD_PHASED:-1}"

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
echo "   Phased upload: ${UPLOAD_PHASED}"

ROOT_DIR_ENV="${ROOT_DIR}" \
DATASET_HF_REPO_ENV="${DATASET_HF_REPO}" \
HF_TOKEN_ENV="${HF_TOKEN}" \
UPLOAD_NUM_WORKERS_ENV="${UPLOAD_NUM_WORKERS}" \
UPLOAD_REPORT_EVERY_ENV="${UPLOAD_REPORT_EVERY}" \
UPLOAD_PHASED_ENV="${UPLOAD_PHASED}" \
/home/projectx/miniconda/bin/python - <<'PYTHON_EOF'
import os
from pathlib import Path
from huggingface_hub import HfApi

root_dir = Path(os.environ["ROOT_DIR_ENV"])
repo_id = os.environ["DATASET_HF_REPO_ENV"]
token = os.environ.get("HF_TOKEN_ENV") or None
num_workers = int(os.environ.get("UPLOAD_NUM_WORKERS_ENV", "8"))
report_every = int(os.environ.get("UPLOAD_REPORT_EVERY_ENV", "30"))
phased = os.environ.get("UPLOAD_PHASED_ENV", "1") == "1"

api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

if phased:
    dataset_root = root_dir / "Dataset"
    phase_dirs = sorted(
        [p for p in dataset_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    for phase_dir in phase_dirs:
        pattern = f"Dataset/{phase_dir.name}/**"
        print(f"[hf] phase start: {pattern}")
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(root_dir),
            allow_patterns=[pattern],
            num_workers=num_workers,
            print_report=True,
            print_report_every=report_every,
        )
        print(f"[hf] phase done: {pattern}")
else:
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
