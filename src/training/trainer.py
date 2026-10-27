"""
Model Training, Validation Loop, and Checkpoint Management.
Location: D:\\Deepfake Detection\\src\\training\\trainer.py
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from src.config import ModelConfig


class XceptionImageTrainer:
    """Manages training, evaluation, and checkpoint persistence for Xception image detector."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_weights: torch.Tensor,
        model_cfg: ModelConfig = ModelConfig(),
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        self.model = model
        self.device = device
        self.cfg = model_cfg

        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.cfg.use_amp)
        self.amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == "bfloat16" else torch.float16

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.cfg.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = np.argmax(logits.float().detach().cpu().numpy(), axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_probs, all_preds, all_labels = [], [], []

        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.cfg.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
            preds = np.argmax(logits.float().cpu().numpy(), axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        eval_loss = total_loss / len(dataloader.dataset)
        eval_acc = accuracy_score(all_labels, all_preds)
        eval_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        try:
            eval_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            eval_auc = 0.5  # Fallback if batch contains single class

        return {
            "loss": eval_loss,
            "accuracy": eval_acc,
            "f1": eval_f1,
            "roc_auc": eval_auc,
            "probabilities": np.array(all_probs),
            "labels": np.array(all_labels),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_dir: Path,
    ) -> Dict[str, list]:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = checkpoint_dir / "best_model.pth"
        history: Dict[str, list] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_auc": []}
        best_val_auc = -1.0

        print(f"Beginning training for {epochs} epochs on device: {self.device}")
        
        for epoch in range(1, epochs + 1):
            t_start = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            elapsed = time.time() - t_start

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_auc"].append(val_metrics["roc_auc"])

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | Time: {elapsed:.1f}s"
            )

            # Save best checkpoint based on validation ROC-AUC
            if val_metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_auc": best_val_auc,
                        "architecture": self.cfg.architecture,
                        "input_size": self.cfg.input_size,
                    },
                    best_ckpt_path,
                )
                print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val AUC: {best_val_auc:.4f})")

        # Save history log
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return history