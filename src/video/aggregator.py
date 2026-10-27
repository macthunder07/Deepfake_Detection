"""
Video-Level Score Aggregation from Frame-Level Classification Probabilities.
Location: D:\\Deepfake Detection\\src\\video\\aggregator.py
"""

import numpy as np
from typing import Dict, Union


def aggregate_video_scores(
    frame_probabilities: np.ndarray,
    strategy: str = "mean",
    top_k_ratio: float = 0.2,
) -> Dict[str, Union[float, int]]:
    """
    Aggregates frame-level fake probabilities into a single video-level score.
    Strategies:
      - 'mean': Average probability across all sampled frames.
      - 'max': Maximum probability (worst-case frame detection).
      - 'top_k': Average of the top-k highest scoring frames (robust to occasional missed face frames).
    """
    if len(frame_probabilities) == 0:
        return {"video_score": 0.0, "num_frames_evaluated": 0}

    probs = np.asarray(frame_probabilities, dtype=np.float64)

    if strategy == "mean":
        score = float(np.mean(probs))
    elif strategy == "max":
        score = float(np.max(probs))
    elif strategy == "top_k":
        k = max(1, int(np.ceil(len(probs) * top_k_ratio)))
        top_probs = np.sort(probs)[-k:]
        score = float(np.mean(top_probs))
    else:
        raise ValueError(f"Unknown aggregation strategy: '{strategy}'. Use 'mean', 'max', or 'top_k'.")

    return {
        "video_score": score,
        "num_frames_evaluated": int(len(probs)),
    }