import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3
import tifffile


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class ImagePathDataset(Dataset):
    """Simple dataset that loads images from a list of paths."""

    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            arr = tifffile.imread(str(path))
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
            img = Image.fromarray(arr)

        if self.transform is not None:
            img = self.transform(img)
        return img


def collect_image_paths(root):
    """Recursively collect image file paths under a single root dir."""
    root = Path(root)
    if not root.exists():
        return []
    paths = []
    for ext in IMG_EXTENSIONS:
        for p in root.rglob(f"*{ext}"):
            name = p.name.lower()
            if any(token in name for token in ("_gt", "mask", "ndsm", "dsm", "building_gt")):
                continue
            paths.append(p)
    paths = sorted(set(paths))
    return paths


class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 wrapper that returns 2048-d features (pool3 activations)."""

    def __init__(self, device):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.preprocess = weights.transforms()

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)


class InceptionV3Classifier(nn.Module):
    """InceptionV3 wrapper that returns 1000-class softmax (for IS)."""

    def __init__(self, device):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, transform_input=False)
        model.eval()
        self.model = model.to(device)
        self.preprocess = weights.transforms()

    @torch.no_grad()
    def forward(self, x):
        logits = self.model(x)
        return F.softmax(logits, dim=-1)


def get_activations(dataloader, feature_extractor, device):
    """Run images through Inception and collect features as a NumPy array."""
    feats = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            preds = feature_extractor(batch)
            feats.append(preds.cpu().numpy())
    if not feats:
        raise ValueError("No images found for feature extraction.")
    return np.concatenate(feats, axis=0)


def compute_stats(activations: np.ndarray):
    """Compute mean and covariance of activations along the batch dimension."""
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root for symmetric positive semi-definite matrices."""
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    sqrt_vals = np.sqrt(vals)
    return (vecs * sqrt_vals) @ vecs.T


def calculate_fid(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Fréchet Inception Distance between two Gaussian distributions."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    cov_prod = sigma1 @ sigma2
    covmean = _sqrtm_psd(cov_prod)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _sqrtm_psd((sigma1 + offset) @ (sigma2 + offset))
    covmean = np.real(covmean)
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def calculate_kid(feats_real: np.ndarray, feats_synth: np.ndarray,
                  subset_size: int = 1000, num_subsets: int = 100,
                  degree: int = 3, gamma: float = None, coef: float = 1.0) -> tuple:
    """Kernel Inception Distance (KID) - unbiased estimator.

    Returns (kid_mean, kid_std).
    Uses polynomial kernel: k(x,y) = (x·y/d + coef)^degree
    """
    n_real = feats_real.shape[0]
    n_synth = feats_synth.shape[0]
    d = feats_real.shape[1]

    if gamma is None:
        gamma = 1.0 / d

    subset_size = min(subset_size, n_real, n_synth)

    rng = np.random.default_rng(42)
    kid_scores = []

    for _ in range(num_subsets):
        idx_r = rng.choice(n_real, size=subset_size, replace=False)
        idx_s = rng.choice(n_synth, size=subset_size, replace=False)
        r = feats_real[idx_r].astype(np.float64)
        s = feats_synth[idx_s].astype(np.float64)

        # Polynomial kernel matrices
        k_rr = (r @ r.T / d + coef) ** degree
        k_ss = (s @ s.T / d + coef) ** degree
        k_rs = (r @ s.T / d + coef) ** degree

        # Unbiased MMD^2 estimator
        n = subset_size
        mmd2 = (
            (k_rr.sum() - np.trace(k_rr)) / (n * (n - 1))
            + (k_ss.sum() - np.trace(k_ss)) / (n * (n - 1))
            - 2.0 * k_rs.mean()
        )
        kid_scores.append(mmd2)

    kid_scores = np.array(kid_scores)
    return float(kid_scores.mean()), float(kid_scores.std())


def calculate_is(dataloader, classifier, device, splits: int = 10) -> tuple:
    """Inception Score (IS).

    IS = exp( E_x[ KL( p(y|x) || p(y) ) ] )
    Returns (is_mean, is_std) across splits.
    """
    probs_list = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            p_yx = classifier(batch)
            probs_list.append(p_yx.cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)  # (N, 1000)
    n = probs.shape[0]
    split_size = n // splits

    scores = []
    for i in range(splits):
        part = probs[i * split_size: (i + 1) * split_size]
        p_y = part.mean(axis=0, keepdims=True)          # marginal p(y)
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        kl = kl.sum(axis=1).mean()
        scores.append(np.exp(kl))

    scores = np.array(scores)
    return float(scores.mean()), float(scores.std())


def calculate_clip_iqa(paths: list, device, batch_size: int = 32) -> float:
    """CLIP-IQA: perceptual image quality score using CLIP.

    Uses contrastive prompts "Good photo" vs "Bad photo".
    Returns mean score in [0, 1] (higher = better quality).
    """
    try:
        import clip
    except ImportError:
        print("  [!] clip not installed. Run: pip install git+https://github.com/openai/CLIP.git")
        return float("nan")

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Contrastive text prompts
    prompts = ["Good photo.", "Bad photo."]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    scores = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i: i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            imgs.append(preprocess(img))

        if not imgs:
            continue

        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            img_feats = model.encode_image(batch)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        # Cosine similarity with text prompts
        logits = (img_feats @ text_feats.T)          # (B, 2)
        probs = logits.softmax(dim=-1)               # (B, 2)
        quality_scores = probs[:, 0].cpu().numpy()   # prob of "Good photo"
        scores.extend(quality_scores.tolist())

    return float(np.mean(scores)) if scores else float("nan")


def run_tsne(
    real_feats, real_labels, real_label_names,
    synth_feats, synth_labels, synth_label_names,
    output_path, max_samples=2000, random_state=42,
):
    rng = np.random.default_rng(random_state)

    def _subsample(feats, labels):
        n = feats.shape[0]
        if n <= max_samples:
            return feats, labels
        idx = rng.choice(n, size=max_samples, replace=False)
        return feats[idx], labels[idx]

    real_sel, real_lbl = _subsample(real_feats, real_labels)
    synth_sel, synth_lbl = _subsample(synth_feats, synth_labels)
    X = np.concatenate([real_sel, synth_sel], axis=0)

    n_real_ds = max(1, len(real_label_names))
    n_synth_ds = max(1, len(synth_label_names))

    def _color_map(n, cmap):
        if n == 1:
            return {0: cmap(0.7)}
        return {i: cmap(0.2 + 0.6 * (i / (n - 1))) for i in range(n)}

    real_color_map = _color_map(n_real_ds, plt.cm.Blues)
    synth_color_map = _color_map(n_synth_ds, plt.cm.Oranges)

    colors = []
    for lbl in real_lbl:
        colors.append(real_color_map[int(lbl)])
    for lbl in synth_lbl:
        colors.append(synth_color_map[int(lbl)])

    n_samples = X.shape[0]
    perplexity = min(30, max(5, (n_samples - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=random_state)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(9, 7))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=6, alpha=0.7, c=colors)

    handles = []
    for i, name in enumerate(real_label_names):
        handles.append(mpatches.Patch(color=real_color_map[i], label=f"Real - {name}"))
    for i, name in enumerate(synth_label_names):
        handles.append(mpatches.Patch(color=synth_color_map[i], label=f"Synth - {name}"))
    if handles:
        plt.legend(handles=handles, fontsize=8, loc="best", ncol=2)

    plt.title("t-SNE of Inception Features (Real: cool, Synth: warm)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def visualize_random_pairs(real_paths, synth_paths, output_path,
                           num_pairs=6, seed=42):
    if not real_paths or not synth_paths:
        return
    rng = random.Random(seed)
    n = min(num_pairs, len(real_paths), len(synth_paths))
    real_sample = rng.sample(real_paths, n)
    synth_sample = rng.sample(synth_paths, n)

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        try:
            r_img = Image.open(real_sample[i]).convert("RGB")
            s_img = Image.open(synth_sample[i]).convert("RGB")
        except Exception:
            continue
        axes[i, 0].imshow(r_img)
        axes[i, 0].set_title(f"Real: {real_sample[i].name}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(s_img)
        axes[i, 1].set_title(f"Synthetic: {synth_sample[i].name}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate style similarity between real and synthetic satellite images."
    )
    parser.add_argument("--real-dirs", nargs="+", required=True)
    parser.add_argument("--synthetic-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="style_eval_results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tsne-samples", type=int, default=2000)
    parser.add_argument("--num-vis-pairs", type=int, default=6)
    parser.add_argument("--max-real-images", type=int, default=0)
    parser.add_argument("--max-synth-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kid-subsets", type=int, default=100,
                        help="Number of subsets for KID estimation.")
    parser.add_argument("--kid-subset-size", type=int, default=1000,
                        help="Subset size for each KID estimate.")
    parser.add_argument("--is-splits", type=int, default=10,
                        help="Number of splits for IS computation.")
    parser.add_argument("--skip-clip-iqa", action="store_true",
                        help="Skip CLIP-IQA computation (requires openai/clip).")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ── Collect paths ──────────────────────────────────────────────────────────
    real_paths, real_dataset_ids, real_dataset_names = [], [], []
    for d in args.real_dirs:
        paths_d = collect_image_paths(d)
        if not paths_d:
            continue
        label_idx = len(real_dataset_names)
        p = Path(d)
        label_name = "/".join(p.parts[-2:]) if len(p.parts) >= 2 else p.name
        real_dataset_names.append(label_name)
        real_paths.extend(paths_d)
        real_dataset_ids.extend([label_idx] * len(paths_d))

    synth_paths, synth_dataset_ids, synth_dataset_names = [], [], []
    for d in args.synthetic_dirs:
        paths_d = collect_image_paths(d)
        if not paths_d:
            continue
        label_idx = len(synth_dataset_names)
        p = Path(d)
        label_name = "/".join(p.parts[-2:]) if len(p.parts) >= 2 else p.name
        synth_dataset_names.append(label_name)
        synth_paths.extend(paths_d)
        synth_dataset_ids.extend([label_idx] * len(paths_d))

    if not real_paths:
        raise ValueError(f"No real images found under: {args.real_dirs}")
    if not synth_paths:
        raise ValueError(f"No synthetic images found under: {args.synthetic_dirs}")

    if args.max_real_images > 0 and len(real_paths) > args.max_real_images:
        idx = rng.sample(range(len(real_paths)), args.max_real_images)
        real_paths = [real_paths[i] for i in idx]
        real_dataset_ids = [real_dataset_ids[i] for i in idx]
    if args.max_synth_images > 0 and len(synth_paths) > args.max_synth_images:
        idx = rng.sample(range(len(synth_paths)), args.max_synth_images)
        synth_paths = [synth_paths[i] for i in idx]
        synth_dataset_ids = [synth_dataset_ids[i] for i in idx]

    print(f"Real images  : {len(real_paths)}")
    print(f"Synth images : {len(synth_paths)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Feature extractor (FID + KID) ─────────────────────────────────────────
    feature_extractor = InceptionV3FeatureExtractor(device)

    real_dataset   = ImagePathDataset(real_paths,  feature_extractor.preprocess)
    synth_dataset  = ImagePathDataset(synth_paths, feature_extractor.preprocess)

    real_loader  = DataLoader(real_dataset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())
    synth_loader = DataLoader(synth_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())

    print("\n[1/4] Extracting Inception features...")
    real_feats  = get_activations(real_loader,  feature_extractor, device)
    synth_feats = get_activations(synth_loader, feature_extractor, device)

    # ── FID ───────────────────────────────────────────────────────────────────
    print("[2/4] Computing FID...")
    mu_r, sigma_r = compute_stats(real_feats)
    mu_s, sigma_s = compute_stats(synth_feats)
    fid = calculate_fid(mu_r, sigma_r, mu_s, sigma_s)
    print(f"  FID  : {fid:.4f}")

    # ── KID ───────────────────────────────────────────────────────────────────
    print("[2/4] Computing KID...")
    kid_mean, kid_std = calculate_kid(
        real_feats, synth_feats,
        subset_size=args.kid_subset_size,
        num_subsets=args.kid_subsets,
    )
    print(f"  KID  : {kid_mean:.6f} ± {kid_std:.6f}")

    # ── IS ────────────────────────────────────────────────────────────────────
    print("[3/4] Computing IS (on synthetic images only)...")
    classifier = InceptionV3Classifier(device)
    is_loader = DataLoader(
        ImagePathDataset(synth_paths, classifier.preprocess),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    is_mean, is_std = calculate_is(is_loader, classifier, device, splits=args.is_splits)
    print(f"  IS   : {is_mean:.4f} ± {is_std:.4f}")

    # ── CLIP-IQA ──────────────────────────────────────────────────────────────
    clip_iqa = float("nan")
    if not args.skip_clip_iqa:
        print("[4/4] Computing CLIP-IQA (on synthetic images)...")
        clip_iqa = calculate_clip_iqa(synth_paths, device, batch_size=args.batch_size)
        print(f"  CLIP-IQA : {clip_iqa:.4f}")
    else:
        print("[4/4] CLIP-IQA skipped.")

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_txt = output_dir / "metrics.txt"
    with metrics_txt.open("w") as f:
        f.write(f"FID          : {fid:.6f}\n")
        f.write(f"KID (mean)   : {kid_mean:.8f}\n")
        f.write(f"KID (std)    : {kid_std:.8f}\n")
        f.write(f"IS (mean)    : {is_mean:.4f}\n")
        f.write(f"IS (std)     : {is_std:.4f}\n")
        f.write(f"CLIP-IQA     : {clip_iqa:.4f}\n")
    print(f"\nMetrics saved to {metrics_txt}")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    tsne_path = output_dir / "tsne_real_vs_synthetic.png"
    print(f"Running t-SNE → {tsne_path}")
    run_tsne(
        real_feats, np.array(real_dataset_ids), real_dataset_names,
        synth_feats, np.array(synth_dataset_ids), synth_dataset_names,
        tsne_path, max_samples=args.max_tsne_samples, random_state=args.seed,
    )

    # ── Side-by-side visualization ────────────────────────────────────────────
    vis_path = output_dir / "side_by_side_examples.png"
    print(f"Side-by-side visualization → {vis_path}")
    visualize_random_pairs(real_paths, synth_paths, vis_path,
                           num_pairs=args.num_vis_pairs, seed=args.seed)

    print("\nDone. Results saved under:", output_dir)


if __name__ == "__main__":
    main()