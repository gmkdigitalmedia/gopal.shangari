"""
Comprehensive training pipeline for MNIST model with extensive logging and error handling.

Provides a robust training framework with checkpointing, early stopping,
learning rate scheduling, and comprehensive metrics tracking.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from ..models.cnn_model import MNISTCNNModel
from ..utils.exceptions import TrainingError, ValidationError
from ..utils.logger import get_logger, MLOpsLogger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    scheduler: str = "plateau"
    scheduler_params: Dict[str, Any] = None
    criterion: str = "cross_entropy"
    early_stopping_patience: int = 5
    save_checkpoints: bool = True
    checkpoint_dir: str = "artifacts/models/checkpoints"
    save_best_only: bool = True
    validation_frequency: int = 1
    device: str = "auto"
    mixed_precision: bool = False
    gradient_clip_val: Optional[float] = None

    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {}

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self, patience: int = 7, min_delta: float = 0.0, restore_best: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            val_score: Current validation score (higher is better)
            model: Model to potentially store weights from

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)

        return False

    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model weights."""
        if self.restore_best:
            self.best_weights = model.state_dict().copy()


class MetricsTracker:
    """Track and manage training metrics."""

    def __init__(self):
        self.metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rates": [],
            "epochs": [],
        }

    def update(self, **kwargs) -> None:
        """Update metrics."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_latest(self) -> Dict[str, float]:
        """Get latest metrics."""
        latest = {}
        for key, values in self.metrics.items():
            if values:
                latest[f"latest_{key}"] = values[-1]
        return latest

    def get_best(self) -> Dict[str, float]:
        """Get best metrics."""
        best = {}
        if self.metrics["train_accuracy"]:
            best["best_train_accuracy"] = max(self.metrics["train_accuracy"])
        if self.metrics["val_accuracy"]:
            best["best_val_accuracy"] = max(self.metrics["val_accuracy"])
        if self.metrics["train_loss"]:
            best["best_train_loss"] = min(self.metrics["train_loss"])
        if self.metrics["val_loss"]:
            best["best_val_loss"] = min(self.metrics["val_loss"])
        return best

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return self.metrics.copy()


class ModelTrainer:
    """
    Comprehensive model trainer with advanced features.

    Features:
    - Multiple optimizers and schedulers
    - Early stopping
    - Checkpointing
    - Mixed precision training
    - Gradient clipping
    - Comprehensive logging
    """

    def __init__(
        self,
        model: MNISTCNNModel,
        config: TrainingConfig,
        logger: Optional[MLOpsLogger] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            logger: Optional logger instance
        """
        self.model = model
        self.config = config
        self.logger = logger or MLOpsLogger(__name__)
        self.metrics_tracker = MetricsTracker()

        self._validate_config()
        self._setup_device()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_criterion()
        self._setup_early_stopping()
        self._setup_checkpointing()

        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def _validate_config(self) -> None:
        """Validate training configuration."""
        if self.config.epochs <= 0:
            raise ValidationError("Number of epochs must be positive")

        if self.config.learning_rate <= 0:
            raise ValidationError("Learning rate must be positive")

        if self.config.early_stopping_patience <= 0:
            raise ValidationError("Early stopping patience must be positive")

        if self.config.validation_frequency <= 0:
            raise ValidationError("Validation frequency must be positive")

    def _setup_device(self) -> None:
        """Setup training device."""
        try:
            self.device = torch.device(self.config.device)
            self.model = self.model.to(self.device)

            logger.info(f"Training device: {self.device}")

            if self.device.type == "cuda":
                logger.info(f"GPU: {torch.cuda.get_device_name()}")
                logger.info(
                    f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory // 1024**3} GB"
                )

        except Exception as e:
            raise TrainingError(f"Failed to setup device: {e}")

    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        try:
            optimizers = {
                "adam": optim.Adam,
                "adamw": optim.AdamW,
                "sgd": optim.SGD,
                "rmsprop": optim.RMSprop,
            }

            if self.config.optimizer not in optimizers:
                raise ValidationError(f"Unsupported optimizer: {self.config.optimizer}")

            optimizer_class = optimizers[self.config.optimizer]

            if self.config.optimizer in ["sgd"]:
                self.optimizer = optimizer_class(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    momentum=0.9,
                    weight_decay=1e-4,
                )
            else:
                self.optimizer = optimizer_class(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=1e-4,
                )

            logger.info(f"Optimizer: {self.config.optimizer}")

        except Exception as e:
            raise TrainingError(f"Failed to setup optimizer: {e}")

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        try:
            if self.config.scheduler == "plateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode="max",
                    factor=0.5,
                    patience=3,
                    **self.config.scheduler_params,
                )
            elif self.config.scheduler == "step":
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=5,
                    gamma=0.5,
                    **self.config.scheduler_params,
                )
            elif self.config.scheduler == "cosine":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.epochs,
                    **self.config.scheduler_params,
                )
            else:
                self.scheduler = None

            logger.info(f"Scheduler: {self.config.scheduler}")

        except Exception as e:
            raise TrainingError(f"Failed to setup scheduler: {e}")

    def _setup_criterion(self) -> None:
        """Setup loss function."""
        try:
            if self.config.criterion == "cross_entropy":
                self.criterion = nn.CrossEntropyLoss()
            elif self.config.criterion == "nll_loss":
                self.criterion = nn.NLLLoss()
            else:
                raise ValidationError(f"Unsupported criterion: {self.config.criterion}")

            logger.info(f"Loss function: {self.config.criterion}")

        except Exception as e:
            raise TrainingError(f"Failed to setup criterion: {e}")

    def _setup_early_stopping(self) -> None:
        """Setup early stopping."""
        if self.config.early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience, restore_best=True
            )
        else:
            self.early_stopping = None

    def _setup_checkpointing(self) -> None:
        """Setup checkpointing."""
        if self.config.save_checkpoints:
            self.checkpoint_dir = Path(self.config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                if self.config.mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)

                    self.scaler.scale(loss).backward()

                    if self.config.gradient_clip_val:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_val
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                    loss.backward()

                    if self.config.gradient_clip_val:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_val
                        )

                    self.optimizer.step()

                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                # Log progress
                if batch_idx % 100 == 0:
                    logger.info(
                        f"Batch {batch_idx}/{len(train_loader)}: "
                        f"Loss: {loss.item():.6f}, "
                        f"Accuracy: {100. * correct / total:.2f}%"
                    )

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                raise TrainingError(f"Training failed at batch {batch_idx}: {e}")

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                try:
                    data, target = data.to(self.device), target.to(self.device)

                    if self.config.mixed_precision:
                        with torch.cuda.amp.autocast():
                            output = self.model(data)
                            loss = self.criterion(output, target)
                    else:
                        output = self.model(data)
                        loss = self.criterion(output, target)

                    total_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)

                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    raise TrainingError(f"Validation failed: {e}")

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        try:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": asdict(self.config),
                "model_info": self.model.get_model_info(),
            }

            if self.scheduler:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)

            # Save best model
            if is_best:
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"New best model saved at epoch {epoch}")

            self.logger.log_checkpoint(str(checkpoint_path), metrics)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise TrainingError(f"Checkpoint saving failed: {e}")

    def train(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            Training results dictionary
        """
        try:
            self.logger.log_training_start(asdict(self.config))
            start_time = time.time()

            best_val_acc = 0.0

            for epoch in range(self.config.epochs):
                epoch_start = time.time()

                logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
                logger.info("-" * 50)

                # Training
                train_loss, train_acc = self.train_epoch(train_loader)

                # Validation
                val_loss, val_acc = 0.0, 0.0
                if val_loader and (epoch + 1) % self.config.validation_frequency == 0:
                    val_loss, val_acc = self.validate_epoch(val_loader)

                # Update metrics
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.metrics_tracker.update(
                    train_loss=train_loss,
                    train_accuracy=train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    learning_rates=current_lr,
                    epochs=epoch + 1,
                )

                # Log metrics
                epoch_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": current_lr,
                    "epoch_time": time.time() - epoch_start,
                }

                self.logger.log_metrics(epoch_metrics, step=epoch + 1)

                logger.info(
                    f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%"
                )
                if val_loader:
                    logger.info(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%")

                # Save checkpoint
                is_best = (
                    val_acc > best_val_acc if val_loader else train_acc > best_val_acc
                )
                if is_best:
                    best_val_acc = max(val_acc, train_acc)

                if self.config.save_checkpoints:
                    should_save = not self.config.save_best_only or is_best
                    if should_save:
                        self.save_checkpoint(epoch + 1, epoch_metrics, is_best)

                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_acc if val_loader else train_acc)
                    else:
                        self.scheduler.step()

                # Early stopping
                if self.early_stopping and val_loader:
                    if self.early_stopping(val_acc, self.model):
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break

                logger.info("")

            # Training completed
            total_time = time.time() - start_time
            final_metrics = {
                **self.metrics_tracker.get_latest(),
                **self.metrics_tracker.get_best(),
                "total_training_time": total_time,
            }

            self.logger.log_training_end(final_metrics, total_time)

            return {
                "final_metrics": final_metrics,
                "all_metrics": self.metrics_tracker.to_dict(),
                "training_time": total_time,
                "best_epoch": (
                    self.metrics_tracker.metrics["val_accuracy"].index(
                        max(self.metrics_tracker.metrics["val_accuracy"])
                    )
                    + 1
                    if self.metrics_tracker.metrics["val_accuracy"]
                    else len(self.metrics_tracker.metrics["train_accuracy"])
                ),
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise TrainingError(f"Training process failed: {e}")


def train_model(
    model: MNISTCNNModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    High-level training function.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Optional validation data loader
        config: Optional training configuration

    Returns:
        Training results
    """
    if config is None:
        config = {}

    training_config = TrainingConfig(**config)
    trainer = ModelTrainer(model, training_config)

    return trainer.train(train_loader, val_loader)
