# Quick Start

## 三条命令快速开始

### 1. 克隆仓库
```bash
git clone git@github.com:YOUR_NAME/evaluation-synthetic-dataset.git
cd evaluation-synthetic-dataset
```

### 2. 一键装环境
```bash
bash setup.sh
```

### 3. 下载权重（可选）
```bash
DOWNLOAD_WEIGHTS=1 bash setup.sh
```

## 运行 Experiment1

```bash
# 激活环境
conda activate synrs3d

# 进入实验目录
cd Experiment1

# 查看实验说明  
cat README.md

# 运行评估（例）
python E1_batch_texture_eval.py \
  --no-download-syntheworld \
  --synthetic-root ../Dataset/synthetic_dataset \
  --real-dirs ../Dataset/real_dataset/DFC19/opt \
  --output-dir style_eval_batch_v1 \
  --batch-size 16 \
  --num-workers 0
```

## 包含的数据集

- **SynRS3D**: 69,667 张合成遥感图像，分辨率 0.05~1m
- **Real datasets**: DFC18, DFC19, GeoNRW, OpenGeoposChallenge 等

## RS3DAda 模型

预训练权重位置：`SynRS3D/pretrain/`

| 模型 | 文件 | 大小 | 用途 |
|------|------|------|------|
| RS3DAda (高度估计) | RS3DAda_vitl_DPT_height.pth | ~1.4GB | 估计地表高度 |
| RS3DAda (土地利用分类) | RS3DAda_vitl_DPT_segmentation.pth | ~1.4GB | 土地利用分类 |

## 环境变量配置

### setup.sh 支持的环境变量：

```bash
# 自定义环境名
ENV_NAME=my_synrs3d bash setup.sh

# 跳过 CUDA PyTorch 安装（装 CPU 版本）
INSTALL_CUDA_TORCH=0 bash setup.sh

# 立刻下载权重
DOWNLOAD_WEIGHTS=1 bash setup.sh

# 自定义 HF 仓库（默认 YOUR_NAME/rs3dada-checkpoints）
HF_REPO=username/custom-checkpoints bash setup.sh
```

## 常见问题

**Q: 装环境时出现 `/dev/shm` 不足错误？**  
A: 本机原因，scripts 里已固定用 `num_workers=0`

**Q: 权重文件未能自动下载？**  
A: 手动从 HF 下载，或确保 `huggingface_hub` 已装

**Q: 想用自己的数据集训练？**  
A: 参考 `SynRS3D/README.md` 中数据组织方式即可

---

详细说明请参考 [DEPLOY.md](DEPLOY.md)
