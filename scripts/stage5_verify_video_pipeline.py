"""
Stage 5: Verify Video Pipeline (Synthetic Video Generation, Face Extraction,
Xception Frame Inference, and Video-Level Score Aggregation).
Location: D:\\Deepfake Detection\\scripts\\stage5_verify_video_pipeline.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms

from src.config import DIRS, ModelConfig
from src.video.face_extractor import extract_faces_from_video
from src.video.aggregator import aggregate_video_scores


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f" {title}")
    print(f"{'=' * 72}")


def create_synthetic_test_video(video_path: Path, num_frames: int = 60, fps: int = 30) -> None:
    """Creates a synthetic test video with a moving face-like rectangle for pipeline verification."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (320, 320))

    rng = np.random.default_rng(42)
    for i in range(num_frames):
        frame = rng.integers(100, 200, size=(320, 320, 3), dtype=np.uint8)
        x = 100 + int(30 * np.sin(i / 10.0))
        y = 100 + int(30 * np.cos(i / 10.0))
        cv2.rectangle(frame, (x, y), (x + 80, y + 80), (180, 150, 130), -1)
        out.write(frame)

    out.release()


def main() -> None:
    print_section("1. GENERATING SYNTHETIC TEST VIDEO")
    raw_video_dir = DIRS["data_raw_video"]
    test_video_path = raw_video_dir / "sample_test_video.mp4"
    create_synthetic_test_video(test_video_path, num_frames=60, fps=30)
    print(f"Created synthetic test video at : {test_video_path}")

    print_section("2. EXTRACTING FACES & FRAMES")
    processed_faces_dir = DIRS["data_processed_faces"]
    extracted_records = extract_faces_from_video(
        video_path=test_video_path,
        output_face_dir=processed_faces_dir,
        sample_rate=10,
        target_size=(299, 299),
    )
    print(f"Extracted {len(extracted_records)} frame crops:")
    for rec in extracted_records:
        print(f"  - Frame {rec['frame_index']:03d} | Face Detected: {rec['face_detected']} | Path: {Path(rec['file_path']).name}")

    print_section("3. RUNNING FRAME-LEVEL INFERENCE WITH XCEPTION CHECKPOINT")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()

    best_ckpt_path = DIRS["checkpoints_image"] / "best_model.pth"
    model = timm.create_model(cfg.architecture, pretrained=True, num_classes=cfg.num_classes)
    
    if best_ckpt_path.is_file():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained image checkpoint from {best_ckpt_path}")
    else:
        print("[INFO] No trained checkpoint found; using base pretrained weights for pipeline test.")

    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )

    frame_probs = []
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16

    with torch.no_grad():
        for rec in extracted_records:
            pil_img = Image.open(rec["file_path"]).convert("RGB")
            tensor_img = transform(pil_img).unsqueeze(0).to(device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(tensor_img)
                prob_fake = torch.softmax(logits, dim=1)[0, 1].float().item()

            frame_probs.append(prob_fake)
            print(f"  Frame {rec['frame_index']:03d} Fake Probability: {prob_fake:.4f}")

    print_section("4. VIDEO-LEVEL SCORE AGGREGATION")
    for strategy in ("mean", "max", "top_k"):
        agg_result = aggregate_video_scores(np.array(frame_probs), strategy=strategy)
        print(f"  Strategy '{strategy:<6}' -> Video Score: {agg_result['video_score']:.4f} (Evaluated {agg_result['num_frames_evaluated']} frames)")

    print("\n[SUCCESS] Stage 5 video extraction and aggregation pipeline verified successfully.")


if __name__ == "__main__":
    main()