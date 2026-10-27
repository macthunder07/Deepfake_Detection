"""
Scan real image dataset directory, run anti-leakage checks, and save image_manifest.csv.
Location: D:\\Deepfake Detection\\scripts\\build_real_manifest.py
"""
import sys
from pathlib import Path

# Project root ko Python path mein inject karein
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DIRS, ModelConfig
from src.data.scanner import build_and_validate_image_manifest
from src.data.manifest import save_manifest


def main() -> None:
    raw_image_dir = DIRS["data_raw_image"]
    print(f"Scanning raw image directory: {raw_image_dir}")

    # Define generators you wish to hold out strictly as out-of-distribution (OOD) test sets
    ood_generators = ["Midjourney", "BigGAN", "Flux_1_Dev"]

    try:
        manifest_df, audit_report, corrupted = build_and_validate_image_manifest(
            root_dir=raw_image_dir,
            ood_generators=ood_generators,
            val_ratio=0.15,
            test_id_ratio=0.15,
            seed=ModelConfig.seed,
        )

        manifest_csv_path = DIRS["manifests"] / "image_manifest.csv"
        saved_csv, saved_audit = save_manifest(manifest_df, manifest_csv_path, audit_report)

        print("\n[SUCCESS] Real dataset manifest built and validated successfully!")
        print(f"  Manifest CSV Saved : {saved_csv}")
        print(f"  Audit JSON Saved   : {saved_audit}")
        print(f"  Total Valid Images : {audit_report['total_samples']} (Corrupted/Skipped: {len(corrupted)})")
        print(f"  Unique Sources     : {audit_report['unique_sources']}")
        print(f"  Split x Class      : {audit_report['split_class_counts']}")
        print(f"  Generator x Split  : {audit_report['generator_split_counts']}")

    except Exception as exc:
        print(f"\n[ERROR] Manifest creation failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()