#!/usr/bin/env bash
# setup.sh for Evaluation_synthetic_dataset
# 快速初始化环境和依赖，可选下载权重

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-synrs3d}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
INSTALL_CUDA_TORCH="${INSTALL_CUDA_TORCH:-1}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-0}"
HF_REPO="${HF_REPO:-YOUR_NAME/rs3dada-checkpoints}"

echo "📦 Setting up Evaluation_synthetic_dataset environment..."
echo "  Environment: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}"
echo "  Download Weights: ${DOWNLOAD_WEIGHTS}"

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
pip install albumentations tqdm ever-beta==0.2.3 huggingface_hub rasterio

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

echo ""
echo "✅ Setup complete for ${ENV_NAME}!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate ${ENV_NAME}"
echo "  2. Run experiments (e.g.): cd Experiment1 && bash evaluation.sh"
echo ""
echo "Optional: Download weights after setup"
echo "  DOWNLOAD_WEIGHTS=1 bash setup.sh"
echo ""
