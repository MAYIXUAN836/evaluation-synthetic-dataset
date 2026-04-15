# Deployment Guide

本文档说明如何把 Evaluation_synthetic_dataset 发布到 GitHub 和 Hugging Face。

## 前置条件

- GitHub 账号和已设置 SSH key 或 personal access token
- Hugging Face 账号，已装 `huggingface_hub` 包
- 本地已初始化并配置 Git

## Step 1: GitHub 仓库初始化

```bash
cd /home/projectx/Evaluation_synthetic_dataset

# 初始化 Git 仓库
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 添加所有文件（.gitignore 已排除大文件）
git add .

# 首次提交
git commit -m "Initial commit: Evaluation synthetic dataset with RS3DAda baseline"

# 改为主分支（GitHub 默认）
git branch -M main

# 添加远程仓库
# 替换 YOUR_NAME 为你的 GitHub 用户名
git remote add origin git@github.com:YOUR_NAME/evaluation-synthetic-dataset.git

# 推送到 GitHub
git push -u origin main
```

## Step 2: Hugging Face 权重仓库创建和上传

### 创建 Hugging Face 模型仓库

```bash
pip install -U huggingface_hub

# 登录
huggingface-cli login

# 创建模型仓库
huggingface-cli repo create rs3dada-checkpoints --type model
```

### 上传权重文件

```bash
# 从仓库根目录执行
cd /home/projectx/Evaluation_synthetic_dataset

# 上传两个权重文件（替换 YOUR_NAME）
huggingface-cli upload YOUR_NAME/rs3dada-checkpoints \
  SynRS3D/pretrain/RS3DAda_vitl_DPT_height.pth \
  RS3DAda_vitl_DPT_height.pth

huggingface-cli upload YOUR_NAME/rs3dada-checkpoints \
  SynRS3D/pretrain/RS3DAda_vitl_DPT_segmentation.pth \
  RS3DAda_vitl_DPT_segmentation.pth
```

或使用 Python：

```bash
python << 'EOF'
from huggingface_hub import HfApi

api = HfApi()
repo_id = "YOUR_NAME/rs3dada-checkpoints"

# 上传权重
api.upload_file(
    path_or_fileobj="/home/projectx/Evaluation_synthetic_dataset/SynRS3D/pretrain/RS3DAda_vitl_DPT_height.pth",
    path_in_repo="RS3DAda_vitl_DPT_height.pth",
    repo_id=repo_id,
    repo_type="model"
)

api.upload_file(
    path_or_fileobj="/home/projectx/Evaluation_synthetic_dataset/SynRS3D/pretrain/RS3DAda_vitl_DPT_segmentation.pth",
    path_in_repo="RS3DAda_vitl_DPT_segmentation.pth",
    repo_id=repo_id,
    repo_type="model"
)

print("✅ Weights uploaded successfully!")
EOF
```

## Step 3: 验证与后续配置

### 验证 GitHub 仓库

```bash
cd /home/projectx/Evaluation_synthetic_dataset
git log --oneline
git remote -v
```

### 验证 Hugging Face 上传

访问 `https://huggingface.co/YOUR_NAME/rs3dada-checkpoints` 检查文件

### 更新 setup.sh 中的 HF_REPO

编辑 `setup.sh`：

```bash
# 第 9 行附近，改为
HF_REPO="${HF_REPO:-YOUR_NAME/rs3dada-checkpoints}"
```

## Step 4: 用户快速开始（验证流程）

```bash
# 克隆代码仓库
git clone git@github.com:YOUR_NAME/evaluation-synthetic-dataset.git
cd evaluation-synthetic-dataset

# 一键装环境
bash setup.sh

# 可选：下载权重
DOWNLOAD_WEIGHTS=1 bash setup.sh

# 运行实验
conda activate synrs3d
cd Experiment1
bash evaluation.sh
```

## 常见问题

**Q: 如何更新权重文件？**  
A: 在 Hugging Face 仓库页面直接删除旧文件，再上传新版本。

**Q: 权重下载时提示认证错误？**  
A: 仓库设为公开，或确保已执行 `huggingface-cli login`

**Q: Git push 失败（拒绝连接）？**  
A: 检查 SSH key 是否正确配置或改用 HTTPS：  
```bash
git remote set-url origin https://github.com/YOUR_NAME/evaluation-synthetic-dataset.git
```

---

更新者：[Your Name]  
最后更新：2026-04-15
