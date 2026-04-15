#!/usr/bin/env python3
import argparse
import csv
import math
import random
import subprocess
import tarfile
import zipfile
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3
import tifffile

import texture_evaluation as te


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


ARCHIVE_EXTS = (".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz")
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

TYPE_MAP = {
    "GTA-V-SID": "Game Engine",
    "GTAH": "Game Engine",
    "AICD": "Blender",
    "Synthinel-1": "Blender",
    "SyntCities": "Blender",
    "RarePlanes": "Blender",
    "SMARS": "Blender",
    "SMARS_SMARS_Release": "Blender",
    "SyntheWorld": "Blender+SD",
    "SynRS3D": "Blender+SD",
    "Ours (SynthUrbanSat)": "Diffusion",
}


class MixedImageDataset(Dataset):
    """Dataset that supports normal files and zip members.

    Source formats:
    - ('file', Path)
    - ('zip', Path_to_zip, inner_path)
    """

    def __init__(self, sources, transform=None):
        self.sources = list(sources)
        self.transform = transform

    def __len__(self):
        return len(self.sources)

    def _pil_from_array(self, arr):
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)

        arr = arr.astype(np.float32)
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]
        return Image.fromarray(arr)

    def __getitem__(self, idx):
        src = self.sources[idx]
        kind = src[0]

        if kind == "file":
            path = src[1]
            try:
                img = Image.open(path).convert("RGB")
            except UnidentifiedImageError:
                arr = tifffile.imread(str(path))
                img = self._pil_from_array(arr)
        elif kind == "zip":
            zpath, inner = src[1], src[2]
            with zipfile.ZipFile(zpath, "r") as zf:
                with zf.open(inner) as f:
                    try:
                        img = Image.open(f).convert("RGB")
                    except UnidentifiedImageError:
                        f.seek(0)
                        arr = tifffile.imread(f)
                        img = self._pil_from_array(arr)
        else:
            raise ValueError(f"Unknown source type: {kind}")

        if self.transform is not None:
            img = self.transform(img)
        return img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Batch quality evaluation for synthetic datasets: auto-extract archives, "
            "compute FID via texture_evaluation.py functions, and export summary tables."
        )
    )
    p.add_argument(
        "--synthetic-root",
        type=str,
        default="Dataset/synthetic_dataset",
        help="Root directory containing synthetic datasets.",
    )
    p.add_argument(
        "--real-dirs",
        nargs="+",
        default=["Dataset/real_dataset/DFC19/opt", "Dataset/real_dataset/GeoNRW/opt"],
        help="One or more real-image directories used as reference.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="style_eval_batch",
        help="Directory to save per-dataset outputs and summary tables.",
    )
    p.add_argument(
        "--extract-archives",
        action="store_true",
        help="If set, extract archives under synthetic-root before evaluation.",
    )
    p.add_argument(
        "--include-zip-members",
        action="store_true",
        default=True,
        help="Read image files directly from .zip members if available.",
    )
    p.add_argument(
        "--download-syntheworld",
        action="store_true",
        default=True,
        help="Try downloading SyntheWorld payload files when only Git LFS pointers are present.",
    )
    p.add_argument(
        "--no-download-syntheworld",
        dest="download_syntheworld",
        action="store_false",
        help="Skip SyntheWorld auto-download step.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--max-real-images",
        type=int,
        default=1500,
        help="Cap the number of real images for speed (0 means unlimited).",
    )
    p.add_argument(
        "--max-synth-images",
        type=int,
        default=1500,
        help="Cap images per synthetic dataset for speed (0 means unlimited).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional explicit synthetic dataset names to evaluate (top-level folder names).",
    )
    p.add_argument(
        "--tsne-per-dataset-samples",
        type=int,
        default=300,
        help="Max real/synth samples used in each per-dataset t-SNE subplot.",
    )
    p.add_argument(
        "--kid-subsets",
        type=int,
        default=30,
        help="Number of random subsets for KID estimation.",
    )
    p.add_argument(
        "--kid-subset-size",
        type=int,
        default=200,
        help="Subset size for each KID estimate.",
    )
    p.add_argument(
        "--ours-placeholder",
        type=str,
        default="Ours (SynthUrbanSat)",
        help="Name of placeholder row/panel for your method.",
    )
    p.add_argument(
        "--upper-bound-real-dirs",
        nargs="*",
        default=[],
        help=(
            "Optional extra real-image directories to evaluate as upper-bound baselines "
            "against the reference real set from --real-dirs."
        ),
    )
    p.add_argument(
        "--upper-bound-prefix",
        type=str,
        default="RealUB",
        help="Prefix used in summary names for upper-bound real baselines.",
    )
    return p.parse_args()


def extract_archives(root: Path) -> None:
    archives = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ARCHIVE_EXTS
    )
    if not archives:
        print("[extract] no archive files found")
        return

    print(f"[extract] found {len(archives)} archives")
    for arc in archives:
        if arc.name.lower().endswith((".z01", ".z02", ".z03")):
            continue

        marker = arc.with_suffix(arc.suffix + ".extracted")
        if marker.exists():
            continue

        out_dir = arc.parent
        try:
            if arc.suffix.lower() == ".zip":
                # unzip handles split zip sets (e.g. .z01/.z02 + .zip) better than zipfile.
                subprocess.run(
                    ["unzip", "-o", str(arc), "-d", str(out_dir)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif arc.suffix.lower() in (".tar", ".gz", ".tgz", ".bz2", ".xz"):
                with tarfile.open(arc, "r:*") as tf:
                    tf.extractall(path=out_dir)
            else:
                continue
            marker.write_text("ok\n", encoding="utf-8")
            print(f"[extract] done: {arc}")
        except Exception as e:
            print(f"[extract] skip/fail: {arc} ({e})")


def _resolve_from_repo(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def sample_paths(paths, k, rng):
    if k <= 0 or len(paths) <= k:
        return paths
    idx = rng.sample(range(len(paths)), k)
    return [paths[i] for i in idx]


def discover_synth_roots(root: Path, only_names):
    cands = [p for p in root.iterdir() if p.is_dir()]
    cands = sorted(cands, key=lambda x: x.name.lower())
    if only_names:
        allowed = set(only_names)
        cands = [p for p in cands if p.name in allowed]
    return cands


def _is_image_name(name: str) -> bool:
    n = name.lower()
    if not n.endswith(IMG_EXTENSIONS):
        return False
    if any(token in n for token in ("_gt", "mask", "ndsm", "dsm", "building_gt")):
        return False
    return True


def collect_sources(ds_root: Path, include_zip_members: bool = True):
    sources = []

    # normal files
    file_paths = te.collect_image_paths(ds_root)
    sources.extend(("file", p) for p in file_paths)

    # zip members
    if include_zip_members:
        for zpath in ds_root.rglob("*.zip"):
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    for m in zf.infolist():
                        if m.is_dir():
                            continue
                        if _is_image_name(m.filename):
                            sources.append(("zip", zpath, m.filename))
            except Exception:
                continue

    # deduplicate on a stable key
    uniq = {}
    for s in sources:
        if s[0] == "file":
            k = ("file", str(Path(s[1]).resolve()))
        else:
            k = ("zip", str(Path(s[1]).resolve()), s[2])
        uniq[k] = s

    return list(uniq.values())


def infer_skip_reason(ds_root: Path) -> str:
    zip_files = sorted(ds_root.rglob("*.zip"))
    if zip_files:
        lfs_like = 0
        for z in zip_files:
            try:
                if z.stat().st_size < 1024:
                    txt = z.read_text(encoding="utf-8", errors="ignore")
                    if txt.startswith("version https://git-lfs.github.com/spec"):
                        lfs_like += 1
            except Exception:
                continue
        if lfs_like == len(zip_files):
            return "Archives are Git LFS pointers; payload files are not downloaded yet"
        return "Archives exist but no readable image members were found"
    return "No readable images in folders or zip members"


def _is_lfs_pointer_file(p: Path) -> bool:
    try:
        if p.stat().st_size > 1024:
            return False
        txt = p.read_text(encoding="utf-8", errors="ignore")
        return txt.startswith("version https://git-lfs.github.com/spec")
    except Exception:
        return False


def ensure_syntheworld_payload(ds_root: Path) -> bool:
    """Download real SyntheWorld payload from HF when local files are LFS pointers."""
    target_files = ["512-1.zip", "512-2.zip", "512-3.zip", "1024_split.zip", "1024_split.z01", "1024_split.z02"]
    pointers = [f for f in target_files if (ds_root / f).exists() and _is_lfs_pointer_file(ds_root / f)]
    if not pointers:
        return False

    print(f"[syntheworld] detected LFS pointers for: {pointers}")

    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        print("[syntheworld] huggingface_hub not available, skip auto-download")
        return False

    cache_dir = ds_root / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for fn in target_files:
        try:
            tmp = Path(
                hf_hub_download(
                    repo_id="JTRNEO/SyntheWorld",
                    repo_type="dataset",
                    filename=fn,
                    cache_dir=str(cache_dir),
                )
            )
            # Ensure final file sits exactly under dataset dir.
            final = ds_root / fn
            if tmp.resolve() != final.resolve() and tmp.exists():
                shutil.copy2(tmp, final)
            print(f"[syntheworld] downloaded: {fn}")
        except Exception as e:
            print(f"[syntheworld] failed: {fn} ({e})")

    refreshed = [f for f in target_files if (ds_root / f).exists() and not _is_lfs_pointer_file(ds_root / f)]
    ok = len(refreshed) > 0
    print(f"[syntheworld] payload_ready={ok}")
    return ok


def compute_fid_stable(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(sigma1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma1 + off) @ (sigma2 + off), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(max(fid, 0.0))


def sample_sources(sources, k, rng):
    if k <= 0 or len(sources) <= k:
        return sources
    idx = rng.sample(range(len(sources)), k)
    return [sources[i] for i in idx]


def compute_kid(feats_real, feats_fake, n_subsets=30, subset_size=200, rng_seed=42):
    """Kernel Inception Distance with polynomial kernel.

    Uses unbiased MMD^2 estimator and returns mean across subsets.
    """
    rng = np.random.default_rng(rng_seed)
    x = np.asarray(feats_real, dtype=np.float64)
    y = np.asarray(feats_fake, dtype=np.float64)
    n = min(len(x), len(y), subset_size)
    if n < 2:
        return float("nan")

    d = x.shape[1]

    def _mmd2_unbiased(xb, yb):
        k_xx = ((xb @ xb.T) / d + 1.0) ** 3
        k_yy = ((yb @ yb.T) / d + 1.0) ** 3
        k_xy = ((xb @ yb.T) / d + 1.0) ** 3

        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)

        term_xx = k_xx.sum() / (n * (n - 1))
        term_yy = k_yy.sum() / (n * (n - 1))
        term_xy = k_xy.mean()
        return term_xx + term_yy - 2.0 * term_xy

    vals = []
    for _ in range(n_subsets):
        ix = rng.choice(len(x), size=n, replace=False)
        iy = rng.choice(len(y), size=n, replace=False)
        vals.append(_mmd2_unbiased(x[ix], y[iy]))
    return float(np.mean(vals))


def compute_inception_score(sources, batch_size, num_workers, device, splits=10, model=None, tfm=None):
    if model is None or tfm is None:
        weights = Inception_V3_Weights.IMAGENET1K_V1
        # Some torchvision versions require aux_logits=True when pretrained weights are used.
        model = inception_v3(weights=weights, transform_input=False).to(device)
        model.eval()
        tfm = weights.transforms()

    ds = MixedImageDataset(sources, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    probs = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out
            p = F.softmax(logits, dim=1)
            probs.append(p.cpu().numpy())

    probs = np.concatenate(probs, axis=0)
    n = probs.shape[0]
    if n < 2:
        return float("nan")

    splits = max(1, min(splits, n))
    split_size = n // splits
    if split_size == 0:
        return float("nan")

    scores = []
    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(py + 1e-12))
        scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))
    return float(np.mean(scores))


def compute_clipiqa(sources, batch_size, num_workers, device, metric=None, tfm=None):
    try:
        import piq
    except Exception:
        return float("nan")

    if metric is None:
        metric = piq.CLIPIQA(data_range=1.0).to(device)
    if tfm is None:
        tfm = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    ds = MixedImageDataset(sources, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    vals = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            s = metric(x)
            vals.append(s.detach().cpu().numpy())

    if not vals:
        return float("nan")
    return float(np.mean(np.concatenate(vals, axis=0)))


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "-"
    return f"{v:.4f}"


def write_summary_tables(rows, out_dir: Path):
    csv_path = out_dir / "summary_table.csv"
    md_path = out_dir / "summary_table.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Type", "FID", "KID", "IS", "CLIP-IQA"])
        for r in rows:
            writer.writerow([
                r["method"],
                r["type"],
                _fmt(r.get("fid")),
                _fmt(r.get("kid")),
                _fmt(r.get("is")),
                _fmt(r.get("clip_iqa")),
            ])

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Method | Type | FID↓ | KID↓ | IS↑ | CLIP-IQA↑ |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        f.write("| Real (Reference) | Real | - | - | - | - |\n")
        for r in rows:
            f.write(
                f"| {r['method']} | {r['type']} | {_fmt(r.get('fid'))} | {_fmt(r.get('kid'))} | {_fmt(r.get('is'))} | {_fmt(r.get('clip_iqa'))} |\n"
            )

    print(f"[summary] wrote: {csv_path}")
    print(f"[summary] wrote: {md_path}")


def write_skip_report(skips, out_dir: Path):
    p = out_dir / "skipped_datasets.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Reason"])
        for name, reason in skips:
            w.writerow([name, reason])
    print(f"[summary] wrote: {p}")


def make_tsne_grid(rows, real_feats, out_path: Path, max_samples=300, seed=42, placeholder_name="Ours (SynthUrbanSat)"):
    if not rows:
        return

    rng = np.random.default_rng(seed)

    # Include placeholder panel
    display_rows = list(rows)
    has_placeholder = any(r["method"] == placeholder_name for r in display_rows)
    if not has_placeholder:
        display_rows.append(
            {
                "method": placeholder_name,
                "type": TYPE_MAP.get(placeholder_name, "Diffusion"),
                "placeholder": True,
            }
        )

    n = len(display_rows)
    cols = min(3, n)
    rows_n = math.ceil(n / cols)

    plt.rcParams.update(
        {
            "axes.facecolor": "#f8f8f7",
            "figure.facecolor": "#f2f1ed",
            "font.family": "DejaVu Serif",
        }
    )

    fig, axes = plt.subplots(rows_n, cols, figsize=(6.2 * cols, 5.2 * rows_n), dpi=220)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    # Shared real embedding across subplots: one global t-SNE so real manifold stays identical.
    valid_rows = [r for r in display_rows if not r.get("placeholder") and not r.get("skipped")]
    nr = min(max_samples, len(real_feats))
    real_idx = rng.choice(len(real_feats), size=nr, replace=False)
    real_sel = real_feats[real_idx]

    synth_blocks = []
    synth_ranges = {}
    cursor = nr
    for r in valid_rows:
        sf = r["feats"]
        ns = min(max_samples, len(sf))
        idx = rng.choice(len(sf), size=ns, replace=False)
        ss = sf[idx]
        synth_blocks.append(ss)
        synth_ranges[r["method"]] = (cursor, cursor + ns)
        cursor += ns

    if synth_blocks:
        X_all = np.concatenate([real_sel] + synth_blocks, axis=0)
        perpl = min(45, max(10, (len(X_all) - 1) // 6))
        emb_all = TSNE(
            n_components=2,
            perplexity=perpl,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(X_all)
        emb_real = emb_all[:nr]
    else:
        emb_real = np.zeros((nr, 2), dtype=np.float32)
        emb_all = emb_real

    x_min, y_min = emb_all.min(axis=0)
    x_max, y_max = emb_all.max(axis=0)
    x_pad = (x_max - x_min) * 0.08 + 1e-3
    y_pad = (y_max - y_min) * 0.08 + 1e-3

    for i, r in enumerate(display_rows):
        ax = axes[i]
        method = r["method"]

        if r.get("placeholder") or r.get("skipped"):
            ax.set_facecolor("#ecebe8")
            ax.text(
                0.5,
                0.55,
                method,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="#1f2937",
            )
            ax.text(
                0.5,
                0.42,
                "Reserved Slot" if r.get("placeholder") else "Unavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#4b5563",
            )
            if r.get("skipped"):
                ax.text(
                    0.5,
                    0.30,
                    str(r.get("skip_reason", "Skipped"))[:88],
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#6b7280",
                    wrap=True,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        rs, re = synth_ranges[method]
        es = emb_all[rs:re]

        ax.scatter(emb_real[:, 0], emb_real[:, 1], s=12, c="#1e40af", alpha=0.55, edgecolors="none", label="Real GT")
        ax.scatter(es[:, 0], es[:, 1], s=14, c="#f59e0b", alpha=0.78, edgecolors="none", label="Synthetic")

        ax.set_title(f"{method}", fontsize=13, fontweight="bold", pad=10)
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.7, color="#9ca3af")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xticks([])
        ax.set_yticks([])

        fid_txt = _fmt(r.get("fid"))
        kid_txt = _fmt(r.get("kid"))
        is_txt = _fmt(r.get("is"))
        clip_txt = _fmt(r.get("clip_iqa"))
        ax.text(
            0.02,
            0.03,
            f"FID {fid_txt} | KID {kid_txt} | IS {is_txt} | CLIP-IQA {clip_txt}",
            transform=ax.transAxes,
            fontsize=8.6,
            color="#111827",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffffcc", "edgecolor": "#d1d5db"},
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    legend_handles = [
        mpatches.Patch(color="#1e40af", label="Real GT"),
        mpatches.Patch(color="#f59e0b", label="Synthetic"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99), fontsize=11)
    fig.suptitle("t-SNE Comparison: Real GT vs Synthetic Datasets", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[summary] wrote: {out_path}")


def main():
    args = parse_args()

    synthetic_root = _resolve_from_repo(args.synthetic_root)
    output_dir = _resolve_from_repo(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    if args.extract_archives:
        extract_archives(synthetic_root)

    if args.download_syntheworld:
        syntheworld_root = synthetic_root / "SyntheWorld"
        if syntheworld_root.exists():
            ensure_syntheworld_payload(syntheworld_root)

    # 1) Build real reference once
    real_paths = []
    for d in args.real_dirs:
        real_paths.extend(te.collect_image_paths(_resolve_from_repo(d)))

    if not real_paths:
        raise ValueError(f"No real images found in --real-dirs: {args.real_dirs}")

    real_paths = sample_paths(real_paths, args.max_real_images, rng)
    print(f"[real] using {len(real_paths)} images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device: {device}")

    extractor = te.InceptionV3FeatureExtractor(device)
    transform = extractor.preprocess

    real_sources = [("file", p) for p in real_paths]
    real_ds = MixedImageDataset(real_sources, transform=transform)
    real_loader = DataLoader(
        real_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    real_feats = te.get_activations(real_loader, extractor, device)
    mu_real, sigma_real = te.compute_stats(real_feats)

    # Initialize expensive models once and reuse for all datasets.
    is_weights = Inception_V3_Weights.IMAGENET1K_V1
    is_model = inception_v3(weights=is_weights, transform_input=False).to(device)
    is_model.eval()
    is_tfm = is_weights.transforms()

    clip_metric = None
    clip_tfm = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    try:
        import piq

        clip_metric = piq.CLIPIQA(data_range=1.0).to(device)
    except Exception:
        clip_metric = None

    # 2) Evaluate each synthetic dataset
    rows = []
    skips = []
    synth_roots = discover_synth_roots(synthetic_root, args.datasets)
    if not synth_roots:
        raise ValueError(f"No synthetic dataset directories found under: {synthetic_root}")

    for ds_root in synth_roots:
        sources = collect_sources(ds_root, include_zip_members=args.include_zip_members)
        if not sources:
            reason = infer_skip_reason(ds_root)
            print(f"[skip] {ds_root.name}: {reason}")
            skips.append((ds_root.name, reason))
            rows.append(
                {
                    "method": ds_root.name,
                    "type": TYPE_MAP.get(ds_root.name, "Unknown"),
                    "fid": float("nan"),
                    "kid": float("nan"),
                    "is": float("nan"),
                    "clip_iqa": float("nan"),
                    "skipped": True,
                    "skip_reason": reason,
                }
            )
            continue

        sources = sample_sources(sources, args.max_synth_images, rng)
        print(f"[eval] {ds_root.name}: {len(sources)} images")

        ds = MixedImageDataset(sources, transform=transform)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        feats = te.get_activations(dl, extractor, device)
        mu_s, sigma_s = te.compute_stats(feats)
        fid = compute_fid_stable(mu_real, sigma_real, mu_s, sigma_s)
        kid = compute_kid(
            real_feats,
            feats,
            n_subsets=args.kid_subsets,
            subset_size=args.kid_subset_size,
            rng_seed=args.seed,
        )
        ins = compute_inception_score(
            sources,
            batch_size=max(8, args.batch_size // 2),
            num_workers=args.num_workers,
            device=device,
            model=is_model,
            tfm=is_tfm,
        )
        clip_iqa = compute_clipiqa(
            sources,
            batch_size=max(8, args.batch_size // 2),
            num_workers=args.num_workers,
            device=device,
            metric=clip_metric,
            tfm=clip_tfm,
        )

        row = {
            "method": ds_root.name,
            "type": TYPE_MAP.get(ds_root.name, "Unknown"),
            "fid": float(fid),
            "kid": float(kid),
            "is": float(ins),
            "clip_iqa": float(clip_iqa),
            "feats": feats,
        }
        rows.append(row)

        ds_out = output_dir / ds_root.name
        ds_out.mkdir(parents=True, exist_ok=True)
        (ds_out / "metrics.txt").write_text(
            "\n".join(
                [
                    f"FID (real vs synthetic): {fid:.6f}",
                    f"KID (real vs synthetic): {kid:.6f}",
                    f"IS (synthetic): {ins:.6f}",
                    f"CLIP-IQA (synthetic): {clip_iqa:.6f}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    # 3) Optional: evaluate additional real sets as upper-bound references.
    for ub_dir in args.upper_bound_real_dirs:
        ub_root = _resolve_from_repo(ub_dir)
        ub_paths = te.collect_image_paths(ub_root)
        if not ub_paths:
            reason = f"No readable images in upper-bound real dir: {ub_dir}"
            print(f"[skip] upper-bound {ub_dir}: {reason}")
            skips.append((f"{args.upper_bound_prefix}_{ub_root.name}", reason))
            continue

        ub_sources = [("file", p) for p in sample_paths(ub_paths, args.max_synth_images, rng)]
        print(f"[upper-bound] {ub_root.name}: {len(ub_sources)} images")

        ub_ds = MixedImageDataset(ub_sources, transform=transform)
        ub_dl = DataLoader(
            ub_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        ub_feats = te.get_activations(ub_dl, extractor, device)
        ub_mu, ub_sigma = te.compute_stats(ub_feats)
        ub_fid = compute_fid_stable(mu_real, sigma_real, ub_mu, ub_sigma)
        ub_kid = compute_kid(
            real_feats,
            ub_feats,
            n_subsets=args.kid_subsets,
            subset_size=args.kid_subset_size,
            rng_seed=args.seed,
        )
        ub_is = compute_inception_score(
            ub_sources,
            batch_size=max(8, args.batch_size // 2),
            num_workers=args.num_workers,
            device=device,
            model=is_model,
            tfm=is_tfm,
        )
        ub_clip = compute_clipiqa(
            ub_sources,
            batch_size=max(8, args.batch_size // 2),
            num_workers=args.num_workers,
            device=device,
            metric=clip_metric,
            tfm=clip_tfm,
        )

        ub_method = f"{args.upper_bound_prefix}_{ub_root.name}"
        rows.append(
            {
                "method": ub_method,
                "type": "Real Upper Bound",
                "fid": float(ub_fid),
                "kid": float(ub_kid),
                "is": float(ub_is),
                "clip_iqa": float(ub_clip),
                "feats": ub_feats,
            }
        )

        ub_out = output_dir / ub_method
        ub_out.mkdir(parents=True, exist_ok=True)
        (ub_out / "metrics.txt").write_text(
            "\n".join(
                [
                    f"FID (reference real vs upper-bound real): {ub_fid:.6f}",
                    f"KID (reference real vs upper-bound real): {ub_kid:.6f}",
                    f"IS (upper-bound real): {ub_is:.6f}",
                    f"CLIP-IQA (upper-bound real): {ub_clip:.6f}",
                    f"source_dir: {ub_root}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    # Add placeholder row for yours.
    rows.append(
        {
            "method": args.ours_placeholder,
            "type": TYPE_MAP.get(args.ours_placeholder, "Diffusion"),
            "fid": float("nan"),
            "kid": float("nan"),
            "is": float("nan"),
            "clip_iqa": float("nan"),
            "placeholder": True,
        }
    )

    rows = sorted(
        rows,
        key=lambda x: (
            x.get("placeholder", False),
            x.get("skipped", False),
            x.get("fid", np.inf) if not (isinstance(x.get("fid"), float) and np.isnan(x.get("fid"))) else np.inf,
        ),
    )
    write_summary_tables(rows, output_dir)
    write_skip_report(skips, output_dir)
    tsne_path = output_dir / "tsne_all_datasets_grid.png"
    make_tsne_grid(
        rows,
        real_feats,
        tsne_path,
        max_samples=args.tsne_per_dataset_samples,
        seed=args.seed,
        placeholder_name=args.ours_placeholder,
    )

    print("[done] batch evaluation completed")


if __name__ == "__main__":
    main()
