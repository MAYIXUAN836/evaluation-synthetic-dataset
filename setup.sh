#!/usr/bin/env bash
# setup.sh for Evaluation_synthetic_dataset
# 快速初始化环境和依赖，可选下载权重

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-synrs3d}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
INSTALL_CUDA_TORCH="${INSTALL_CUDA_TORCH:-1}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-1}"
DOWNLOAD_DATASETS_FROM_HF="${DOWNLOAD_DATASETS_FROM_HF:-0}"
DOWNLOAD_SYNTHEWORLD="${DOWNLOAD_SYNTHEWORLD:-1}"
HF_REPO="${HF_REPO:-YixuanMa/rs3dada-checkpoints}"
DATASET_HF_REPO="${DATASET_HF_REPO:-YixuanMa/evaluation-synthetic-dataset-data}"

echo "📦 Setting up Evaluation_synthetic_dataset environment..."
echo "  Environment: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}"
echo "  Download Weights: ${DOWNLOAD_WEIGHTS}"
echo "  Download Datasets: ${DOWNLOAD_DATASETS}"
echo "  Download Datasets From HF: ${DOWNLOAD_DATASETS_FROM_HF}"
echo "  Dataset HF Repo: ${DATASET_HF_REPO}"

# Load conda
source /home/projectx/miniconda/etc/profile.d/conda.sh || {
    echo "⚠️ conda not found. Please ensure conda is installed."
    exit 1
}

# Create or activate conda environment
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "🔧 Creating conda environment: ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
else
    echo "✓ Environment ${ENV_NAME} already exists"
fi

conda activate "${ENV_NAME}"
echo "✓ Activated environment: ${ENV_NAME}"

# Install PyTorch with CUDA support (if requested)
if [[ "${INSTALL_CUDA_TORCH}" == "1" ]]; then
    echo "📥 Installing PyTorch with CUDA 11.8..."
    conda install -y pytorch::pytorch=2.2.1 pytorch::torchvision=0.17.1 pytorch::torchaudio=2.2.1 pytorch::pytorch-cuda=11.8 -c pytorch -c nvidia
else
    echo "⏭️ Skipping CUDA PyTorch installation (INSTALL_CUDA_TORCH=0)"
fi

# Install conda dependencies
echo "📥 Installing system dependencies via conda..."
conda install -y gdal

# Install Python dependencies
echo "📥 Installing Python packages..."
pip install -U pip
pip install albumentations tqdm ever-beta==0.2.3 huggingface_hub rasterio requests beautifulsoup4

# Optional: Download model weights from Hugging Face
if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
    echo "📥 Downloading model weights from ${HF_REPO}..."
    mkdir -p "${ROOT_DIR}/SynRS3D/pretrain"
    
    # Check if huggingface_hub is installed
    if ! python -c "import huggingface_hub" 2>/dev/null; then
        echo "⚠️ huggingface_hub not available, skipping weight download"
    else
        python -c "from huggingface_hub import hf_hub_download; hf_hub_download('${HF_REPO}', 'RS3DAda_vitl_DPT_height.pth', local_dir='${ROOT_DIR}/SynRS3D/pretrain')" && \
        python -c "from huggingface_hub import hf_hub_download; hf_hub_download('${HF_REPO}', 'RS3DAda_vitl_DPT_segmentation.pth', local_dir='${ROOT_DIR}/SynRS3D/pretrain')" || {
            echo "⚠️ Failed to download weights. Make sure you have HF_TOKEN set if repo is private."
        }
    fi
else
    echo "⏭️ Skipping weight download (DOWNLOAD_WEIGHTS=0)"
fi

# Optional: Download datasets from Hugging Face dataset repo.
if [[ "${DOWNLOAD_DATASETS_FROM_HF}" == "1" ]]; then
    echo "📥 Downloading Dataset/ from Hugging Face dataset repo: ${DATASET_HF_REPO}"
    ROOT_DIR_ENV="${ROOT_DIR}" \
    DATASET_HF_REPO_ENV="${DATASET_HF_REPO}" \
    /home/projectx/miniconda/bin/python - <<'PYTHON_EOF'
import os
from huggingface_hub import snapshot_download

root_dir = os.environ["ROOT_DIR_ENV"]
repo_id = os.environ["DATASET_HF_REPO_ENV"]

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=root_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["Dataset/**"],
)

print(f"[hf] dataset snapshot synced from {repo_id} to {root_dir}")
PYTHON_EOF
elif [[ "${DOWNLOAD_DATASETS}" == "1" ]]; then
    echo "📥 Downloading datasets from public sources..."
    bash "${ROOT_DIR}/Experiment1/down_all.sh"

    if [[ "${DOWNLOAD_SYNTHEWORLD}" == "1" ]]; then
        python "${ROOT_DIR}/Experiment1/download_syntheworld.py" \
            --synthetic-root "${ROOT_DIR}/Dataset/synthetic_dataset" || {
                echo "⚠️ SyntheWorld payload download skipped or failed"
            }
    fi
else
    echo "⏭️ Skipping dataset download"
    echo "   Use DOWNLOAD_DATASETS_FROM_HF=1 for HF snapshot or DOWNLOAD_DATASETS=1 for public sources."
fi

# Basic post-setup sanity checks
for required_path in "${ROOT_DIR}/SynRS3D/pretrain/RS3DAda_vitl_DPT_height.pth" "${ROOT_DIR}/SynRS3D/pretrain/RS3DAda_vitl_DPT_segmentation.pth"; do
    if [[ -f "${required_path}" ]]; then
        echo "✓ Found model: ${required_path##*/}"
    else
        echo "⚠️ Missing model: ${required_path##*/}"
    fi
done

if [[ -d "${ROOT_DIR}/Dataset" ]]; then
    echo "✓ Dataset root present: ${ROOT_DIR}/Dataset"
fi

echo ""
echo "✅ Setup complete for ${ENV_NAME}!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate ${ENV_NAME}"
echo "  2. Run experiments (e.g.): cd Experiment1 && bash evaluation.sh"
echo ""
echo "Optional: Download weights after setup"
echo "  DOWNLOAD_WEIGHTS=1 bash setup.sh"
echo "Optional: Download Dataset/ from Hugging Face dataset repo"
echo "  DOWNLOAD_DATASETS_FROM_HF=1 DATASET_HF_REPO=YixuanMa/evaluation-synthetic-dataset-data bash setup.sh"
echo "Optional: Download public datasets"
echo "  DOWNLOAD_DATASETS=1 bash setup.sh"
echo ""
