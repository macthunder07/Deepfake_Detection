"""
Project-wide configuration, directory paths, baseline model settings,
and dataset license/access metadata registry.
Location: D:\\Deepfake Detection\\src\\config.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(r"D:\Deepfake Detection")

# Standard project directory layout
DIRS = {
    "data_raw_image": PROJECT_ROOT / "data" / "raw" / "image_detection",
    "data_raw_video": PROJECT_ROOT / "data" / "raw" / "video_detection",
    "data_processed_faces": PROJECT_ROOT / "data" / "processed" / "video_faces",
    "manifests": PROJECT_ROOT / "data" / "manifests",
    "checkpoints_image": PROJECT_ROOT / "checkpoints" / "image_detector",
    "checkpoints_video": PROJECT_ROOT / "checkpoints" / "video_detector",
    "outputs_eval": PROJECT_ROOT / "outputs" / "evaluation",
    "outputs_logs": PROJECT_ROOT / "outputs" / "logs",
    "configs": PROJECT_ROOT / "configs",
}


@dataclass(frozen=True)
class ModelConfig:
    """Baseline Xception model and preprocessing parameters."""
    architecture: str = "legacy_xception"
    num_classes: int = 2
    input_size: Tuple[int, int, int] = (3, 299, 299)
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    interpolation: str = "bicubic"
    default_batch_size: int = 16  # Conservative for 8 GB VRAM at 299x299
    use_amp: bool = True
    amp_dtype: str = "bfloat16"   # Supported natively on RTX 5060 (CC 12.0)
    seed: int = 42


LABEL_MAP: Dict[str, int] = {
    "real": 0,
    "fake": 1,
}

INV_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

VALID_TASKS = {"image", "video"}
VALID_SPLITS = {"train", "val", "test_id", "test_ood"}


@dataclass(frozen=True)
class DatasetGovernanceInfo:
    """Documents licensing, access conditions, classes, and metadata before download."""
    name: str
    task: str
    license_type: str
    access_conditions: str
    real_sources: str
    fake_generators: List[str]
    recommended_role: str
    notes: str


DATASET_REGISTRY: Dict[str, DatasetGovernanceInfo] = {
    "GenImage": DatasetGovernanceInfo(
        name="GenImage",
        task="image",
        license_type="Academic / Research Non-Commercial (ImageNet base + open diffusion subsets)",
        access_conditions="Public research links (Google Drive / Baidu / HuggingFace subsets); adhere to ImageNet terms.",
        real_sources="ImageNet-1K validation/training photographs",
        fake_generators=[
            "Stable_Diffusion_v1_4",
            "Stable_Diffusion_v1_5",
            "GLIDE",
            "VQDM",
            "Wukong",
            "BigGAN",
            "ADM",
            "Midjourney",
        ],
        recommended_role="Train/Val/Test_ID on SD v1.4/GLIDE/ADM; hold out Midjourney/BigGAN for Test_OOD.",
        notes="Keep prompt/class pairings aligned so real and fake sets share semantic distributions.",
    ),
    "CIFAKE_or_SDXL_Permitted": DatasetGovernanceInfo(
        name="CIFAKE / Open-License Diffusion Subset",
        task="image",
        license_type="CC-BY-4.0 / MIT / Apache-2.0 depending on HuggingFace subset",
        access_conditions="Direct open access without institutional request form.",
        real_sources="Real photographic benchmarks (COCO / OpenImages / CIFAR-real)",
        fake_generators=["SDXL", "Flux_1_Dev", "PixArt_Alpha", "Kandinsky"],
        recommended_role="Supplemental Train/Val or unseen modern generator Test_OOD benchmark.",
        notes="Beware of low-resolution 32x32 subsets (like raw CIFAR); prefer >=256px sources to avoid upscaling artifacts.",
    ),
    "FaceForensics++": DatasetGovernanceInfo(
        name="FaceForensics++ (FF++)",
        task="video",
        license_type="FaceForensics Terms of Use (Non-commercial academic research only)",
        access_conditions="Requires submitting institutional Google Form agreement to receive download script/mirrors.",
        real_sources="1,000 YouTube talking-head videos (c23 / c40 compression)",
        fake_generators=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"],
        recommended_role="Train/Val/Test_ID on 3-4 methods (e.g., Deepfakes, Face2Face, FaceSwap); reserve NeuralTextures or FaceShifter for Test_OOD.",
        notes="Each manipulated video derives from one of the 1,000 original source IDs. All manipulations of source_id '001' must share the exact same split.",
    ),
    "Celeb-DF-v2": DatasetGovernanceInfo(
        name="Celeb-DF (v2)",
        task="video",
        license_type="Non-commercial research license agreement",
        access_conditions="Requires user agreement form for official archive download.",
        real_sources="590 real celebrity YouTube interviews + 300 additional YouTube real videos",
        fake_generators=["Celeb_DF_Synthesis"],
        recommended_role="Cross-dataset Video Test_OOD (or secondary fine-tuning with strict subject/video grouping).",
        notes="Contains 5,639 high-quality deepfake videos with reduced color/boundary artifacts compared to early FF++.",
    ),
}