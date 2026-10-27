"""
Stage 3: Inspect Raw Image Directory, Generate Smoke-Test Calibration Dataset
(if no external dataset is present yet), Build & Validate Manifest, and Test
PyTorch DataLoader + GPU Mixed-Precision Batch Transfer.
Location: D:\\Deepfake Detection\\scripts\\stage3_prepare_and_verify_data.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import timm
from PIL import Image

from src.config import DIRS, ModelConfig
from src.data.dataset import create_image_dataloaders
from src.data.manifest import save_manifest
from src.data.scanner import SUPPORTED_IMAGE_EXTENSIONS, build_and_validate_image_manifest


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f" {title}")
    print(f"{'=' * 72}")


def generate_synthetic_smoke_dataset(smoke_root: Path, seed: int = 42) -> None:
    """
    Creates a small, deterministic on-disk dataset in `data/raw/image_detection_smoke`
    ONLY for pipeline verification when `data/raw/image_detection` has no user files yet.
    Includes:
      - In-distribution sources (src_000 .. src_044):
          * real/real_photo (natural smooth gradient + camera-like sensor noise)
          * fake/Stable_Diffusion_v1_4 (periodic high-frequency grid artifact)
          * fake/GLIDE (checkerboard upsampling artifact)
      - Out-of-distribution sources (ood_src_000 .. ood_src_014):
          * real/real_photo_ood (unseen real source group for balanced OOD evaluation)
          * fake/Midjourney (unseen generator radial frequency rings)
    """
    rng = np.random.default_rng(seed)
    height, width = 320, 320
    y_grid, x_grid = np.mgrid[0:height, 0:width]

    # 1. Generate 45 in-distribution source groups
    for idx in range(45):
        src_id = f"src_{idx:03d}"
        base_phase = rng.uniform(0, 2 * np.pi)
        base_color = rng.uniform(60, 190, size=(1, 1, 3))

        # Smooth natural-like base scene associated with this source_id
        smooth_bg = (
            base_color
            + 35.0 * np.sin(x_grid[:, :, None] / 40.0 + base_phase)
            + 35.0 * np.cos(y_grid[:, :, None] / 40.0 - base_phase)
        )

        # A. Real image for src_id (smooth scene + mild Gaussian sensor noise)
        real_arr = np.clip(smooth_bg + rng.normal(0, 3.5, size=(height, width, 3)), 0, 255).astype(np.uint8)
        real_path = smoke_root / "SmokeBench" / "real" / "real_photo" / f"{src_id}_real.png"
        real_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(real_arr).save(real_path)

        # B. Fake variant for the same src_id (either Stable_Diffusion_v1_4 or GLIDE)
        gen_name = "Stable_Diffusion_v1_4" if idx % 2 == 0 else "GLIDE"
        if gen_name == "Stable_Diffusion_v1_4":
            # Periodic 8x8 latent-diffusion-like high-frequency grid modulation
            artifact = 14.0 * (np.sin(x_grid * np.pi / 4.0) * np.sin(y_grid * np.pi / 4.0))[:, :, None]
        else:
            # 4x4 checkerboard transposed-conv-like artifact
            checker = (((x_grid // 2) + (y_grid // 2)) % 2 * 2 - 1)[:, :, None] * 12.0
            artifact = checker

        fake_arr = np.clip(smooth_bg + artifact + rng.normal(0, 2.0, size=(height, width, 3)), 0, 255).astype(np.uint8)
        fake_path = smoke_root / "SmokeBench" / "fake" / gen_name / f"{src_id}_fake.png"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fake_arr).save(fake_path)

    # 2. Generate 15 OOD source groups (both real and unseen Midjourney fake)
    for idx in range(15):
        ood_id = f"ood_src_{idx:03d}"
        base_phase = rng.uniform(0, 2 * np.pi)
        base_color = rng.uniform(70, 185, size=(1, 1, 3))
        smooth_bg = (
            base_color
            + 30.0 * np.sin(x_grid[:, :, None] / 35.0 + base_phase)
            + 30.0 * np.cos(y_grid[:, :, None] / 35.0 - base_phase)
        )

        # OOD Real
        real_ood_arr = np.clip(smooth_bg + rng.normal(0, 3.5, size=(height, width, 3)), 0, 255).astype(np.uint8)
        real_ood_path = smoke_root / "SmokeBench" / "real" / "real_photo_ood" / f"{ood_id}_real.png"
        real_ood_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(real_ood_arr).save(real_ood_path)

        # OOD Fake (Midjourney simulated radial high-frequency rings + grid)
        radius = np.sqrt((x_grid - width / 2.0) ** 2 + (y_grid - height / 2.0) ** 2)
        radial_artifact = 13.0 * np.sin(radius * np.pi / 3.5)[:, :, None]
        fake_ood_arr = np.clip(smooth_bg + radial_artifact, 0, 255).astype(np.uint8)
        fake_ood_path = smoke_root / "SmokeBench" / "fake" / "Midjourney" / f"{ood_id}_fake.png"
        fake_ood_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fake_ood_arr).save(fake_ood_path)


def main() -> None:
    print_section("1. CHECKING RAW IMAGE DATASET DIRECTORY")
    raw_image_dir = DIRS["data_raw_image"]
    existing_images = [
        p for p in raw_image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    if len(existing_images) > 0:
        print(f"[FOUND] Detected {len(existing_images)} user image files in '{raw_image_dir}'.")
        active_data_root = raw_image_dir
        manifest_name = "image_manifest.csv"
        ood_generators = ["Midjourney", "BigGAN", "Flux_1_Dev"]
    else:
        smoke_root = PROJECT_ROOT / "data" / "raw" / "image_detection_smoke"
        print(f"[INFO] '{raw_image_dir}' currently contains 0 images.")
        print(f"[INFO] Generating on-disk smoke-test calibration dataset at:\n       '{smoke_root}'")
        print("       (NOTE: This synthetic dataset validates pipeline mechanics only, not benchmark accuracy.)")
        generate_synthetic_smoke_dataset(smoke_root, seed=ModelConfig.seed)
        active_data_root = smoke_root
        manifest_name = "image_smoke_manifest.csv"
        ood_generators = ["Midjourney", "real_photo_ood"]

    print_section("2. SCANNING DISK, COMPUTING SHA-256 & dHASH, AND VALIDATING SPLITS")
    manifest_df, audit_report, corrupted = build_and_validate_image_manifest(
        root_dir=active_data_root,
        ood_generators=ood_generators,
        val_ratio=0.15,
        test_id_ratio=0.15,
        seed=ModelConfig.seed,
    )

    manifest_csv_path = DIRS["manifests"] / manifest_name
    saved_csv, saved_audit = save_manifest(manifest_df, manifest_csv_path, audit_report)

    print(f"Manifest Saved To : {saved_csv}")
    print(f"Audit JSON Saved  : {saved_audit}")
    print(f"Total Valid Images: {audit_report['total_samples']} (Corrupted/Skipped: {len(corrupted)})")
    print(f"Unique Sources    : {audit_report['unique_sources']}")
    print(f"Unique SHA-256    : {audit_report['unique_sha256']}")
    print(f"dHash Collisions  : {audit_report['near_duplicate_dhashes_across_splits']}")
    print(f"Split x Class     : {audit_report['split_class_counts']}")
    print(f"Generator x Split : {audit_report['generator_split_counts']}")

    print_section("3. TESTING PYTORCH DATALOADERS & GPU MIXED-PRECISION FORWARD PASS")
    cfg = ModelConfig()
    loaders, class_weights, _ = create_image_dataloaders(
        manifest_csv=saved_csv,
        batch_size=cfg.default_batch_size,
        num_workers=0,
    )

    print(f"Computed Train Class Weights [Real=0, Fake=1]: {class_weights.tolist()}")
    for split_name, loader in loaders.items():
        print(f"  DataLoader['{split_name:<8}'] -> {len(loader.dataset):>4} samples | {len(loader):>2} batches (batch_size={cfg.default_batch_size})")

    # Fetch one batch from train loader and run through legacy_xception on GPU with bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_batch = next(iter(loaders["train"]))
    images = train_batch["image"].to(device, non_blocking=True)
    labels = train_batch["label"].to(device, non_blocking=True)

    print(f"\nSample Batch Tensor Shape : {tuple(images.shape)} | dtype: {images.dtype}")
    print(f"Batch Pixel Range (Norm)  : min={images.min().item():.3f}, max={images.max().item():.3f}")
    print(f"Batch Labels              : {labels.tolist()}")

    torch.cuda.reset_peak_memory_stats(device)
    model = timm.create_model(cfg.architecture, pretrained=True, num_classes=cfg.num_classes).to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
        logits = model(images)
        loss = criterion(logits, labels)

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss encountered during batch verification!")

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Forward Pass Loss (BF16)  : {loss.item():.4f} (Finite: True)")
    print(f"Peak VRAM (Batch={len(images)})     : {peak_mem_mb:.2f} MB")
    print("\n[SUCCESS] Stage 3 dataset ingestion, manifest audit, and DataLoader GPU test passed.")


if __name__ == "__main__":
    main()