"""
Training Loop and Hardware Acceleration for LOB Deep Learning Predictor.
Implements dynamic CUDA allocation, CrossEntropyLoss, Adam optimizer,
gradient backpropagation, and metric tracking across epochs.
"""

import logging
import os
import sys
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LOBTrainer")


class LOBTrainer:
    """
    Manages neural network training on dedicated GPU/CUDA hardware or CPU fallback.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: str = "checkpoints",
    ):
        """
        :param model: PyTorch nn.Module
        :param learning_rate: Optimizer learning rate
        :param weight_decay: L2 regularization
        :param class_weights: Optional tensor of weights for CrossEntropyLoss
        :param checkpoint_dir: Directory to save model weights
        """
        # Dynamic CUDA device allocation as required by PRD
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log_hardware_info()

        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "epoch_time": [],
        }

    def _log_hardware_info(self) -> None:
        """Inspect and log active hardware acceleration status."""
        logger.info(f"Target compute device selected: {self.device}")
        if self.device.type == "cuda":
            device_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_idx)
            total_mem_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
            logger.info(f"CUDA Hardware: {gpu_name} (Total VRAM: {total_mem_gb:.2f} GB)")
            logger.info(f"PyTorch CUDA Version: {torch.version.cuda}")
        else:
            logger.warning("CUDA is not available. Falling back to CPU.")

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Execute one full training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = batch_y.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total_samples += batch_size

        epoch_loss = total_loss / max(total_samples, 1)
        epoch_acc = correct / max(total_samples, 1)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model performance on validation data without gradient tracking."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)

            batch_size = batch_y.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total_samples += batch_size

        val_loss = total_loss / max(total_samples, 1)
        val_acc = correct / max(total_samples, 1)
        return val_loss, val_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        save_best: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model over specified epochs, logging progress and tracking loss decrease.
        """
        best_val_loss = float("inf")
        best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_lob_model.pt")

        logger.info(f"Starting training loop for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            epoch_time = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_time"].append(epoch_time)

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    best_checkpoint_path,
                )
                logger.info(f"Saved new best model checkpoint to '{best_checkpoint_path}'")

        logger.info("Training complete.")
        return self.history
