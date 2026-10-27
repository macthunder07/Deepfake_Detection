"""
Video Frame Sampling and Face Detection / Cropping Pipeline using OpenCV.
Location: D:\\Deepfake Detection\\src\\video\\face_extractor.py
"""

import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image


def extract_faces_from_video(
    video_path: Path,
    output_face_dir: Path,
    sample_rate: int = 15,  # Extract every 15th frame
    target_size: Tuple[int, int] = (299, 299),
) -> List[Dict[str, object]]:
    """
    Samples frames from a video, detects faces using OpenCV Haar Cascade (if available),
    or uses robust regional cropping (compatible with OpenCV 5.0+),
    crops and resizes face regions to target_size, and saves them to disk.
    Returns a list of extracted face metadata dictionaries.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video container: {video_path}")

    # Check if CascadeClassifier is available in this OpenCV build (OpenCV 5.0+ compatibility)
    face_cascade = None
    if hasattr(cv2, "CascadeClassifier"):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if Path(cascade_path).exists():
                face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            face_cascade = None

    video_name = video_path.stem
    video_face_folder = output_face_dir / video_name
    video_face_folder.mkdir(parents=True, exist_ok=True)

    extracted_records: List[Dict[str, object]] = []
    frame_idx = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            img_h, img_w, _ = frame.shape
            faces = []

            if face_cascade is not None and not face_cascade.empty():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )

            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                margin_w = int(w * 0.2)
                margin_h = int(h * 0.2)
                x1 = max(0, x - margin_w)
                y1 = max(0, y - margin_h)
                x2 = min(img_w, x + w + margin_w)
                y2 = min(img_h, y + h + margin_h)

                face_crop = frame[y1:y2, x1:x2]
                face_detected = True
                suffix = ""
            else:
                # Robust fallback for OpenCV 5.0+ or frames without detected faces:
                # Focus on the upper-center region (talking head portrait standard)
                crop_dim = int(min(img_h, img_w) * 0.75)
                start_y = max(0, int((img_h - crop_dim) * 0.35))
                start_x = max(0, (img_w - crop_dim) // 2)
                face_crop = frame[start_y:start_y+crop_dim, start_x:start_x+crop_dim]
                face_detected = False
                suffix = "_fallback"

            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(face_rgb).resize(target_size, Image.Resampling.BICUBIC)

            face_filename = f"{video_name}_f{frame_idx:05d}{suffix}.jpg"
            face_filepath = video_face_folder / face_filename
            pil_face.save(face_filepath, quality=95)

            extracted_records.append(
                {
                    "file_path": str(face_filepath.resolve()),
                    "video_path": str(video_path.resolve()),
                    "source_id": video_name,
                    "frame_index": frame_idx,
                    "face_detected": face_detected,
                }
            )
            saved_count += 1

        frame_idx += 1

    cap.release()
    return extracted_records