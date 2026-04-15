#!/bin/bash
# ============================================================
# Dataset Downloader - 所有数据集统一下载脚本
# 用法: bash download_all.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$(cd "${SCRIPT_DIR}/../Dataset" && pwd)"
HTML_DIR="${OUTPUT_DIR}"  # 存放gdrive_*.html的目录
VENV_DIR="${SCRIPT_DIR}/.downloader_venv"

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi

# 使用本地虚拟环境，避免系统Python受限（PEP 668）
"${VENV_DIR}/bin/python" -m pip install -q --upgrade pip

PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
ZENODO_GET="${VENV_DIR}/bin/zenodo_get"
AWS="${VENV_DIR}/bin/aws"

mkdir -p "${OUTPUT_DIR}"

echo "=============================="
echo " Dataset Downloader"
echo "=============================="

# --------------------------------------------------------------
# 1. Google Drive 数据集（用Python脚本批量下载）
# --------------------------------------------------------------
echo ""
echo "[1/3] Google Drive datasets..."
"${PIP}" install requests beautifulsoup4 -q

"${PYTHON}" "${SCRIPT_DIR}/download_datasets.py" \
    --html_dir "${HTML_DIR}" \
    --output_dir "${OUTPUT_DIR}"

# --------------------------------------------------------------
# 2. SyntCities - Zenodo (record: 6967325)
# --------------------------------------------------------------
echo ""
echo "[2/3] SyntCities from Zenodo..."
"${PIP}" install zenodo_get -q

mkdir -p "${OUTPUT_DIR}/SyntCities"
cd "${OUTPUT_DIR}/SyntCities"

# 只下载Paris.zip（14.5GB）
"${ZENODO_GET}" 6967325 -g "Paris.zip"

echo "  [✓] SyntCities 下载完成"

# --------------------------------------------------------------
# 3. RarePlanes - AWS Open Data（只下synthetic RGB部分）
# --------------------------------------------------------------
echo ""
echo "[3/3] RarePlanes from AWS..."
mkdir -p "${OUTPUT_DIR}/synthetic_dataset/RarePlanes/train_images"
mkdir -p "${OUTPUT_DIR}/synthetic_dataset/RarePlanes/test_images"

"${PIP}" install awscli -q

"${AWS}" s3 cp s3://rareplanes-public/synthetic/train/images/ \
    "${OUTPUT_DIR}/synthetic_dataset/RarePlanes/train_images/" \
    --recursive \
    --no-sign-request \
    --only-show-errors \
    --region us-west-2

"${AWS}" s3 cp s3://rareplanes-public/synthetic/test/images/ \
    "${OUTPUT_DIR}/synthetic_dataset/RarePlanes/test_images/" \
    --recursive \
    --no-sign-request \
    --only-show-errors \
    --region us-west-2

echo "  [✓] RarePlanes 下载完成"

echo ""
echo "=============================="
echo " 全部完成！"
echo " 保存位置: ${OUTPUT_DIR}"
echo "=============================="