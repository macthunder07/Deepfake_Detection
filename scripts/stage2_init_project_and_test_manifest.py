"""
Stage 2: Scaffold Project Directories, Inspect Dataset Governance Registry,
and Verify the Leak-Free Manifest & Split Validation Engine.
Location: D:\\Deepfake Detection\\scripts\\stage2_init_project_and_test_manifest.py
"""

import hashlib
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import DATASET_REGISTRY, DIRS, ModelConfig
from src.data.manifest import assign_grouped_splits, validate_manifest


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f" {title}")
    print(f"{'=' * 72}")


def scaffold_directories() -> None:
    print_section("1. PROJECT DIRECTORY SCAFFOLDING")
    for key, folder in DIRS.items():
        existed = folder.exists()
        folder.mkdir(parents=True, exist_ok=True)
        status = "EXISTS " if existed else "CREATED"
        print(f"  [{status}] {key:<22} -> {folder}")


def display_dataset_governance() -> None:
    print_section("2. DATASET LICENSE, ACCESS & SPLIT GOVERNANCE REGISTRY")
    for name, info in DATASET_REGISTRY.items():
        print(f"\nDataset          : {info.name} (Task: {info.task.upper()})")
        print(f"  License        : {info.license_type}")
        print(f"  Access Rules   : {info.access_conditions}")
        print(f"  Real Sources   : {info.real_sources}")
        print(f"  Generators     : {', '.join(info.fake_generators)}")
        print(f"  Split Strategy : {info.recommended_role}")
        print(f"  Forensic Note  : {info.notes}")


def build_mock_manifest_df() -> pd.DataFrame:
    """
    Creates an in-memory mock manifest simulating 60 source videos/images
    with multiple frames/variants per source_id and an OOD generator ('Midjourney')
    to test group-preserving splits and leakage detection.
    """
    rows = []
    sample_counter = 1

    # Simulate 60 distinct base sources (e.g. source_000 to source_059)
    for src_idx in range(60):
        source_id = f"src_{src_idx:03d}"

        # 1. Real image/frame for this source_id
        real_path = f"mock/real/{source_id}_real.jpg"
        rows.append(
            {
                "sample_id": f"smp_{sample_counter:04d}",
                "file_path": real_path,
                "task": "image",
                "label": 0,
                "label_str": "real",
                "source_dataset": "GenImage",
                "generator": "real_photo",
                "source_id": source_id,
                "split": "",
                "sha256": hashlib.sha256(real_path.encode()).hexdigest(),
                "width": 299,
                "height": 299,
            }
        )
        sample_counter += 1

        # 2. In-distribution fake variant derived from the same source_id
        id_gen = "Stable_Diffusion_v1_4" if src_idx % 2 == 0 else "GLIDE"
        fake_id_path = f"mock/{id_gen}/{source_id}_fake.jpg"
        rows.append(
            {
                "sample_id": f"smp_{sample_counter:04d}",
                "file_path": fake_id_path,
                "task": "image",
                "label": 1,
                "label_str": "fake",
                "source_dataset": "GenImage",
                "generator": id_gen,
                "source_id": source_id,
                "split": "",
                "sha256": hashlib.sha256(fake_id_path.encode()).hexdigest(),
                "width": 299,
                "height": 299,
            }
        )
        sample_counter += 1

    # Simulate 15 standalone OOD samples from an unseen generator ('Midjourney')
    for ood_idx in range(15):
        source_id = f"ood_src_{ood_idx:03d}"
        ood_path = f"mock/Midjourney/{source_id}.jpg"
        rows.append(
            {
                "sample_id": f"smp_{sample_counter:04d}",
                "file_path": ood_path,
                "task": "image",
                "label": 1,
                "label_str": "fake",
                "source_dataset": "GenImage",
                "generator": "Midjourney",
                "source_id": source_id,
                "split": "",
                "sha256": hashlib.sha256(ood_path.encode()).hexdigest(),
                "width": 299,
                "height": 299,
            }
        )
        sample_counter += 1

    return pd.DataFrame(rows)


def run_manifest_validation_tests() -> None:
    print_section("3. MANIFEST ENGINE & ANTI-LEAKAGE SELF-TESTS")
    raw_df = build_mock_manifest_df()
    ood_holds = ["Midjourney"]

    # Test A: Assign grouped splits and verify valid manifest passes
    split_df = assign_grouped_splits(
        raw_df,
        ood_generators=ood_holds,
        val_ratio=0.15,
        test_id_ratio=0.15,
        seed=ModelConfig.seed,
    )

    report = validate_manifest(
        split_df,
        ood_generators=ood_holds,
        check_files_exist=False,
        strict_ood_source_isolation=True,
    )
    print("[PASS] Test A: Grouped split assignment & zero-leakage validation succeeded.")
    print(f"       Total samples: {report['total_samples']} | Unique sources: {report['unique_sources']}")
    print(f"       Split x Class breakdown: {report['split_class_counts']}")
    print(f"       Generator x Split breakdown: {report['generator_split_counts']}")

    # Test B: Deliberately inject source_id leakage (same source_id in train and val)
    leaked_source_df = split_df.copy()
    first_train_source = leaked_source_df.loc[leaked_source_df["split"] == "train", "source_id"].iloc[0]
    # Flip just one row of that source_id from 'train' to 'val'
    target_indices = leaked_source_df.index[leaked_source_df["source_id"] == first_train_source].tolist()
    leaked_source_df.loc[target_indices[0], "split"] = "val"

    try:
        validate_manifest(leaked_source_df, ood_generators=ood_holds, check_files_exist=False)
        print("[FAIL] Test B: Validator failed to catch source_id leakage!")
        sys.exit(1)
    except ValueError as exc:
        print(f"[PASS] Test B: Validator caught intentional source_id split leakage -> {str(exc).splitlines()[1].strip()}")

    # Test C: Deliberately inject duplicate SHA-256 hash across train and test_id
    leaked_hash_df = split_df.copy()
    train_sha = leaked_hash_df.loc[leaked_hash_df["split"] == "train", "sha256"].iloc[0]
    test_row_idx = leaked_hash_df.index[leaked_hash_df["split"] == "test_id"][0]
    leaked_hash_df.loc[test_row_idx, "sha256"] = train_sha

    try:
        validate_manifest(leaked_hash_df, ood_generators=ood_holds, check_files_exist=False)
        print("[FAIL] Test C: Validator failed to catch duplicate SHA-256 leakage!")
        sys.exit(1)
    except ValueError as exc:
        print(f"[PASS] Test C: Validator caught intentional SHA-256 duplicate leakage -> {str(exc).splitlines()[1].strip()}")

    # Test D: Deliberately leak an OOD generator ('Midjourney') into 'train'
    leaked_ood_df = split_df.copy()
    ood_row_idx = leaked_ood_df.index[leaked_ood_df["generator"] == "Midjourney"][0]
    leaked_ood_df.loc[ood_row_idx, "split"] = "train"

    try:
        validate_manifest(leaked_ood_df, ood_generators=ood_holds, check_files_exist=False)
        print("[FAIL] Test D: Validator failed to catch OOD generator leakage!")
        sys.exit(1)
    except ValueError as exc:
        print(f"[PASS] Test D: Validator caught intentional OOD generator leakage -> {str(exc).splitlines()[1].strip()}")

    print("\n[SUCCESS] Stage 2 scaffolding and manifest integrity tests completed with zero errors.")


def main() -> None:
    scaffold_directories()
    display_dataset_governance()
    run_manifest_validation_tests()


if __name__ == "__main__":
    main()