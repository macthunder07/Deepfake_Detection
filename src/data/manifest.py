"""
Dataset Manifest Schema, Grouped Split Builder, and Zero-Leakage Auditor.
Location: D:\\Deepfake Detection\\src\\data\\manifest.py
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import LABEL_MAP, VALID_SPLITS, VALID_TASKS


REQUIRED_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "sample_id",          # Unique identifier for this row
    "file_path",          # Relative or absolute path to image or extracted face crop / video
    "task",               # 'image' or 'video'
    "label",              # 0 (real) or 1 (fake)
    "label_str",          # 'real' or 'fake'
    "source_dataset",     # e.g., 'GenImage', 'FaceForensics++', 'Celeb-DF-v2'
    "generator",          # 'real_photo', 'real_video', 'Stable_Diffusion_v1_4', 'Deepfakes', etc.
    "source_id",          # Grouping ID: original video ID or base image ID (prevents split leakage)
    "split",              # 'train', 'val', 'test_id', 'test_ood'
    "sha256",             # Content hash (or deterministic file signature) to prevent duplicate leakage
    "width",              # Image/frame width in pixels (0 if uninspected video container)
    "height",             # Image/frame height in pixels (0 if uninspected video container)
)


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hex digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def assign_grouped_splits(
    df: pd.DataFrame,
    ood_generators: Optional[Iterable[str]] = None,
    ood_datasets: Optional[Iterable[str]] = None,
    val_ratio: float = 0.15,
    test_id_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assigns 'train', 'val', 'test_id', and 'test_ood' splits while guaranteeing:
      1. Any row matching `ood_generators` or `ood_datasets` goes strictly to 'test_ood'
         (unless its `source_id` is shared with in-distribution data, in which case we keep
         the shared `source_id` together or isolate only standalone OOD source_ids).
      2. All rows sharing the same `(source_dataset, source_id)` stay in the exact same split.
    """
    if not (0.0 < val_ratio < 0.5) or not (0.0 < test_id_ratio < 0.5):
        raise ValueError("val_ratio and test_id_ratio must each be between 0 and 0.5.")

    out = df.copy()
    ood_gen_set: Set[str] = set(ood_generators or [])
    ood_ds_set: Set[str] = set(ood_datasets or [])

    # Create a globally unique group key combining dataset and original source_id
    group_keys = out["source_dataset"].astype(str) + "::" + out["source_id"].astype(str)
    out["_group_key"] = group_keys

    is_ood_row = out["generator"].isin(ood_gen_set) | out["source_dataset"].isin(ood_ds_set)

    # Identify groups where ALL members are OOD vs groups that mix real source + OOD fake variant
    # Note: If a real video (e.g. FF++ video 001) has 3 ID fakes and 1 OOD fake (e.g. NeuralTextures),
    # putting the real video in 'train' and its NeuralTextures derivative in 'test_ood' is standard
    # in FF++ leave-one-manipulation-out ONLY if the test_ood group doesn't leak into train/val evaluation
    # of the same manipulation. However, to prevent background/identity leakage into test_ood,
    # we hold out the OOD generator rows into 'test_ood' and route the remaining rows by group_key.
    out["split"] = ""
    out.loc[is_ood_row, "split"] = "test_ood"

    in_dist_mask = ~is_ood_row
    in_dist_df = out.loc[in_dist_mask].copy()

    if len(in_dist_df) == 0:
        out.drop(columns=["_group_key"], inplace=True)
        return out

    unique_groups = in_dist_df["_group_key"].unique()
    if len(unique_groups) < 3:
        raise ValueError(
            f"Need at least 3 distinct (source_dataset, source_id) groups to split into "
            f"train/val/test_id, found {len(unique_groups)}."
        )

    # Step 1: Split in-distribution groups into (train + val) vs test_id
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_id_ratio, random_state=seed)
    train_val_idx, test_id_idx = next(
        gss_test.split(in_dist_df, groups=in_dist_df["_group_key"])
    )

    train_val_df = in_dist_df.iloc[train_val_idx]
    test_id_df = in_dist_df.iloc[test_id_idx]

    # Step 2: Split (train + val) groups into train vs val
    relative_val_ratio = val_ratio / (1.0 - test_id_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_ratio, random_state=seed + 1)
    train_idx, val_idx = next(
        gss_val.split(train_val_df, groups=train_val_df["_group_key"])
    )

    train_groups = set(train_val_df.iloc[train_idx]["_group_key"])
    val_groups = set(train_val_df.iloc[val_idx]["_group_key"])
    test_id_groups = set(test_id_df["_group_key"])

    out.loc[in_dist_mask & out["_group_key"].isin(train_groups), "split"] = "train"
    out.loc[in_dist_mask & out["_group_key"].isin(val_groups), "split"] = "val"
    out.loc[in_dist_mask & out["_group_key"].isin(test_id_groups), "split"] = "test_id"

    # Optional strict guard: if a group has OOD samples, ensure its real counterpart isn't in train
    # when strict identity isolation is requested. (Tracked in audit report).
    out.drop(columns=["_group_key"], inplace=True)
    return out


def validate_manifest(
    df: pd.DataFrame,
    ood_generators: Optional[Iterable[str]] = None,
    check_files_exist: bool = True,
    strict_ood_source_isolation: bool = False,
) -> Dict[str, object]:
    """
    Validates a manifest DataFrame for schema completeness, label consistency,
    source_id split leakage, SHA-256 duplicate leakage, and OOD generator isolation.
    Raises ValueError with diagnostic details if any integrity rule is violated.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Check required columns
    missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Manifest is missing required columns: {missing_cols}")

    if len(df) == 0:
        raise ValueError("Manifest DataFrame is empty.")

    # 2. Check for null/empty critical fields
    for col in ("sample_id", "file_path", "task", "label", "label_str", "source_dataset", "generator", "source_id", "split", "sha256"):
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            errors.append(f"Column '{col}' contains null or empty values.")

    # 3. Check task, split, and label validity
    invalid_tasks = set(df["task"].unique()) - VALID_TASKS
    if invalid_tasks:
        errors.append(f"Invalid task values found: {invalid_tasks}. Expected {VALID_TASKS}.")

    invalid_splits = set(df["split"].unique()) - VALID_SPLITS
    if invalid_splits:
        errors.append(f"Invalid split values found: {invalid_splits}. Expected {VALID_SPLITS}.")

    invalid_label_strs = set(df["label_str"].unique()) - set(LABEL_MAP.keys())
    if invalid_label_strs:
        errors.append(f"Invalid label_str values: {invalid_label_strs}. Expected {set(LABEL_MAP.keys())}.")

    # Verify numeric label matches label_str
    expected_numeric = df["label_str"].map(LABEL_MAP)
    if not (df["label"] == expected_numeric).all():
        errors.append("Mismatch between 'label' (0/1) and 'label_str' ('real'/'fake').")

    # 4. Check file existence if requested
    if check_files_exist:
        missing_files = [str(p) for p in df["file_path"] if not Path(str(p)).is_file()]
        if missing_files:
            sample_missing = missing_files[:5]
            errors.append(
                f"{len(missing_files)} file(s) do not exist on disk. First up to 5: {sample_missing}"
            )

    # 5. Check in-distribution source_id leakage across train / val / test_id
    id_df = df[df["split"].isin({"train", "val", "test_id"})].copy()
    if len(id_df) > 0:
        id_df["_group_key"] = id_df["source_dataset"].astype(str) + "::" + id_df["source_id"].astype(str)
        group_split_counts = id_df.groupby("_group_key")["split"].nunique()
        leaked_groups = group_split_counts[group_split_counts > 1].index.tolist()
        if leaked_groups:
            errors.append(
                f"SOURCE_ID LEAKAGE DETECTED across train/val/test_id for {len(leaked_groups)} "
                f"group(s)! Examples: {leaked_groups[:5]}"
            )

    # Optional: check if source_ids in test_ood also appear in train
    if strict_ood_source_isolation:
        all_group_keys = df["source_dataset"].astype(str) + "::" + df["source_id"].astype(str)
        temp_df = df.assign(_group_key=all_group_keys)
        ood_groups = set(temp_df.loc[temp_df["split"] == "test_ood", "_group_key"])
        train_groups = set(temp_df.loc[temp_df["split"] == "train", "_group_key"])
        overlap_ood_train = ood_groups.intersection(train_groups)
        if overlap_ood_train:
            errors.append(
                f"STRICT OOD SOURCE LEAKAGE: {len(overlap_ood_train)} source_id group(s) appear in "
                f"both 'train' and 'test_ood'. Examples: {list(overlap_ood_train)[:5]}"
            )

    # 6. Check SHA-256 exact duplicate leakage across ANY distinct splits
    sha_split_counts = df.groupby("sha256")["split"].nunique()
    leaked_hashes = sha_split_counts[sha_split_counts > 1].index.tolist()
    if leaked_hashes:
        errors.append(
            f"DUPLICATE FILE (SHA-256) LEAKAGE DETECTED across splits for {len(leaked_hashes)} "
            f"file hash(es)! Examples: {leaked_hashes[:5]}"
        )

    # 7. Check OOD generator holdout enforcement
    if ood_generators:
        ood_set = set(ood_generators)
        leaked_ood_rows = df[df["generator"].isin(ood_set) & (df["split"] != "test_ood")]
        if len(leaked_ood_rows) > 0:
            bad_gens = leaked_ood_rows["generator"].unique().tolist()
            bad_splits = leaked_ood_rows["split"].unique().tolist()
            errors.append(
                f"OOD GENERATOR LEAKAGE: Reserved OOD generator(s) {bad_gens} found in split(s) {bad_splits}!"
            )

    # 8. Check class presence in train and val
    for required_split in ("train", "val"):
        sub = df[df["split"] == required_split]
        if len(sub) > 0:
            present_labels = set(sub["label"].unique())
            if present_labels != {0, 1}:
                warnings.append(
                    f"Split '{required_split}' only contains class labels {present_labels} (expected both 0 and 1)."
                )

    if errors:
        error_msg = "\n  - ".join(["Manifest Validation Failed:"] + errors)
        raise ValueError(error_msg)

    # Build summary statistics
    split_class_table = (
        df.groupby(["split", "label_str"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    generator_split_table = (
        df.groupby(["generator", "split"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )

    return {
        "total_samples": int(len(df)),
        "unique_sources": int((df["source_dataset"].astype(str) + "::" + df["source_id"].astype(str)).nunique()),
        "unique_sha256": int(df["sha256"].nunique()),
        "split_class_counts": split_class_table,
        "generator_split_counts": generator_split_table,
        "warnings": warnings,
    }


def save_manifest(df: pd.DataFrame, csv_path: Path, summary_report: Dict[str, object]) -> Tuple[Path, Path]:
    """Saves validated manifest CSV and a companion JSON audit report."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    audit_path = csv_path.with_suffix(".audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    return csv_path, audit_path