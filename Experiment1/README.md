# Experiment1 使用说明

本 README 记录了下一次复现本实验时，终端需要执行的完整命令（激活环境、进入目录、运行评估）。

## 1. 激活环境

```bash
conda activate comfyui
```

如果你的 shell 没有自动加载 conda，可先执行：

```bash
source /home/projectx/miniconda/etc/profile.d/conda.sh
conda activate comfyui
```

## 2. 进入项目目录

```bash
cd /home/projectx/Evaluation_synthetic_dataset
```

## 3. 运行 Experiment1（推荐稳定参数）

```bash
PYTHONUNBUFFERED=1 /home/projectx/miniconda/envs/comfyui/bin/python \
  Experiment1/E1_batch_texture_eval.py \
  --no-download-syntheworld \
  --synthetic-root Dataset/synthetic_dataset \
  --real-dirs Dataset/real_dataset/DFC19/opt \
  --output-dir Experiment1/style_eval_batch_final_v2 \
  --max-synth-images 120 \
  --max-real-images 320 \
  --batch-size 16 \
  --num-workers 0 \
  --kid-subset-size 50 \
  > Experiment1/style_eval_batch_final_v2_run.log 2>&1
```

## 4. 查看运行日志

```bash
tail -n 120 Experiment1/style_eval_batch_final_v2_run.log
```

## 5. 结果输出位置

运行完成后，主要结果在：

- `Experiment1/style_eval_batch_final_v2/summary_table.md`
- `Experiment1/style_eval_batch_final_v2/summary_table.csv`
- `Experiment1/style_eval_batch_final_v2/skipped_datasets.csv`
- `Experiment1/style_eval_batch_final_v2/tsne_all_datasets_grid.png`

## 6. 常见注意事项

- 本机上建议固定使用 `--num-workers 0`，可避免 `/dev/shm` 共享内存不足导致的 DataLoader 报错。
- 如果你已提前下载好 SyntheWorld，保留 `--no-download-syntheworld` 可避免重复下载。
- 若想使用两个真实域一起评估，可把 `--real-dirs` 改为：

```bash
--real-dirs Dataset/real_dataset/DFC19/opt Dataset/real_dataset/GeoNRW/opt
```
