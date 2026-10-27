"""
Stage 4: Train Baseline Xception Image Detector, Monitor Validation Metrics,
Save Best Checkpoint, Calibrate Threshold, and Evaluate on Test Splits.
Location: D:\\Deepfake Detection\\scripts\\stage4_train_image_detector.py
"""
import sys
from pathlib import Path

# Project root ko Python path mein inject karein
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import timm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score

from src.config import DIRS, ModelConfig
from src.data.dataset import create_image_dataloaders
from src.training.trainer import XceptionImageTrainer
from src.training.calibration import calibrate_threshold_youden, apply_calibrated_decision


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f" {title}")
    print(f"{'=' * 72}")


def main() -> None:
    print_section("1. SETTING UP DATA LOADERS & MANIFEST")
    manifests_dir = DIRS["manifests"]
    smoke_csv = manifests_dir / "image_smoke_manifest.csv"
    real_csv = manifests_dir / "image_manifest.csv"

    if real_csv.is_file():
        active_manifest = real_csv
        print(f"Using real image manifest : {active_manifest}")
    elif smoke_csv.is_file():
        active_manifest = smoke_csv
        print(f"Using smoke-test manifest : {active_manifest}")
    else:
        raise FileNotFoundError(
            "No manifest CSV found! Run Stage 3 (stage3_prepare_and_verify_data.py) first."
        )

    cfg = ModelConfig()
    loaders, class_weights, full_df = create_image_dataloaders(
        manifest_csv=active_manifest,
        batch_size=cfg.default_batch_size,
        num_workers=0,
    )

    print(f"Available Splits in Dataloaders: {list(loaders.keys())}")
    print(f"Class Weights [Real, Fake]     : {class_weights.tolist()}")

    print_section("2. INITIALIZING MODEL & DEVICE")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using Device                   : {device}")

    torch.cuda.manual_seed_all(cfg.seed)
    model = timm.create_model(cfg.architecture, pretrained=True, num_classes=cfg.num_classes)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized Model              : {cfg.architecture} ({total_params:,} parameters)")

    print_section("3. TRAINING & VALIDATION MONITORING")
    trainer = XceptionImageTrainer(
        model=model,
        device=device,
        class_weights=class_weights,
        model_cfg=cfg,
        lr=1e-4,
        weight_decay=1e-4,
    )

    epochs = 3
    checkpoint_dir = DIRS["checkpoints_image"]

    history = trainer.fit(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
    )

    print_section("4. THRESHOLD CALIBRATION ON VALIDATION SET")
    best_ckpt_path = checkpoint_dir / "best_model.pth"
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} (Val AUC: {checkpoint['val_auc']:.4f})")

    val_eval = trainer.evaluate(loaders["val"])
    calibrated_threshold = calibrate_threshold_youden(val_eval["labels"], val_eval["probabilities"])
    print(f"Optimal Calibrated Threshold (Youden's J) : {calibrated_threshold:.4f}")

    calib_meta = {"calibrated_threshold": calibrated_threshold, "val_roc_auc": float(val_eval["roc_auc"])}
    with open(checkpoint_dir / "calibration_meta.json", "w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2)

    print_section("5. EVALUATION ON TEST_ID AND TEST_OOD SPLITS")
    eval_splits = {}
    if "test_id" in loaders:
        eval_splits["test_id"] = loaders["test_id"]
    if "test_ood" in loaders:
        eval_splits["test_ood"] = loaders["test_ood"]

    for split_name, loader in eval_splits.items():
        print(f"\n--- Evaluating Split: {split_name.upper()} ---")
        metrics = trainer.evaluate(loader)
        preds_calibrated = apply_calibrated_decision(metrics["probabilities"], threshold=calibrated_threshold)
        
        acc = accuracy_score(metrics["labels"], preds_calibrated)
        f1 = f1_score(metrics["labels"], preds_calibrated, zero_division=0)
        auc = metrics["roc_auc"]
        cm = confusion_matrix(metrics["labels"], preds_calibrated)

        print(f"Accuracy  : {acc:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC-AUC   : {auc:.4f}")
        print(f"Confusion Matrix (Rows=True [Real, Fake], Cols=Pred):\n{cm}")

        eval_summary = {
            "split": split_name,
            "accuracy": float(acc),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "threshold_used": calibrated_threshold,
        }
        eval_out_path = DIRS["outputs_eval"] / f"eval_{split_name}_summary.json"
        DIRS["outputs_eval"].mkdir(parents=True, exist_ok=True)
        with open(eval_out_path, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)
        print(f"Saved evaluation report to {eval_out_path}")

    print(f"\n[SUCCESS] Stage 4 training, calibration, and evaluation pipeline completed successfully.")


if __name__ == "__main__":
    main()