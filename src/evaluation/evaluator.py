"""
Comprehensive model evaluation with detailed metrics and release decision support.

Provides extensive evaluation capabilities including confusion matrices,
per-class metrics, statistical tests, and automated release recommendations.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from ..models.cnn_model import MNISTCNNModel
from ..utils.exceptions import EvaluationError, ModelReleaseError
from ..utils.logger import get_logger, MLOpsLogger

logger = get_logger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    device: str = "auto"
    batch_size: int = 64
    num_classes: int = 10
    class_names: List[str] = None
    save_predictions: bool = True
    save_probabilities: bool = True
    calculate_roc_auc: bool = True
    detailed_report: bool = True

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.class_names is None:
            self.class_names = [str(i) for i in range(self.num_classes)]


@dataclass
class ReleaseThresholds:
    """Thresholds for model release decision."""

    min_accuracy: float = 0.95
    min_precision: float = 0.9
    min_recall: float = 0.9
    min_f1_score: float = 0.9
    max_loss: float = 0.2
    min_per_class_accuracy: float = 0.8
    max_inference_time_ms: float = 100.0
    min_confidence_threshold: float = 0.7


class ModelEvaluator:
    """
    Comprehensive model evaluator with release decision capabilities.

    Features:
    - Detailed metrics calculation
    - Per-class performance analysis
    - Confusion matrix generation
    - ROC-AUC calculation
    - Inference time measurement
    - Automated release recommendations
    """

    def __init__(
        self,
        model: MNISTCNNModel,
        config: EvaluationConfig,
        release_thresholds: Optional[ReleaseThresholds] = None,
        logger: Optional[MLOpsLogger] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            config: Evaluation configuration
            release_thresholds: Thresholds for release decision
            logger: Optional logger instance
        """
        self.model = model
        self.config = config
        self.release_thresholds = release_thresholds or ReleaseThresholds()
        self.logger = logger or MLOpsLogger(__name__)

        self._setup_device()
        self.model.eval()

    def _setup_device(self) -> None:
        """Setup evaluation device."""
        try:
            self.device = torch.device(self.config.device)
            self.model = self.model.to(self.device)
            logger.info(f"Evaluation device: {self.device}")
        except Exception as e:
            raise EvaluationError(f"Failed to setup device: {e}")

    def predict_batch(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on a batch of data.

        Args:
            data: Input batch tensor

        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            data = data.to(self.device)

            with torch.no_grad():
                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)

            return predictions.cpu(), probabilities.cpu()

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise EvaluationError(f"Batch prediction failed: {e}")

    def evaluate_dataset(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: Data loader for evaluation

        Returns:
            Comprehensive evaluation results
        """
        try:
            logger.info("Starting model evaluation...")
            start_time = time.time()

            all_predictions = []
            all_probabilities = []
            all_targets = []
            all_losses = []
            inference_times = []

            criterion = nn.CrossEntropyLoss()

            # Collect predictions
            for batch_idx, (data, targets) in enumerate(dataloader):
                batch_start = time.time()

                predictions, probabilities = self.predict_batch(data)

                # Calculate loss
                data, targets = data.to(self.device), targets.to(self.device)
                with torch.no_grad():
                    outputs = self.model(data)
                    loss = criterion(outputs, targets)

                all_predictions.extend(predictions.numpy())
                all_probabilities.extend(probabilities.numpy())
                all_targets.extend(targets.cpu().numpy())
                all_losses.append(loss.item())

                batch_time = (
                    (time.time() - batch_start) * 1000 / len(data)
                )  # ms per sample
                inference_times.append(batch_time)

                if batch_idx % 50 == 0:
                    logger.info(f"Processed batch {batch_idx}/{len(dataloader)}")

            # Convert to numpy arrays
            predictions = np.array(all_predictions)
            probabilities = np.array(all_probabilities)
            targets = np.array(all_targets)

            # Calculate comprehensive metrics
            evaluation_results = self._calculate_comprehensive_metrics(
                targets, predictions, probabilities, all_losses, inference_times
            )

            evaluation_time = time.time() - start_time
            evaluation_results["evaluation_time"] = evaluation_time

            logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
            self.logger.log_evaluation_results(evaluation_results)

            return evaluation_results

        except Exception as e:
            logger.error(f"Dataset evaluation failed: {e}")
            raise EvaluationError(f"Dataset evaluation failed: {e}")

    def _calculate_comprehensive_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        losses: List[float],
        inference_times: List[float],
    ) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        try:
            results = {}

            # Basic metrics
            results["accuracy"] = float(accuracy_score(targets, predictions))
            results["precision_macro"] = float(
                precision_score(targets, predictions, average="macro", zero_division=0)
            )
            results["recall_macro"] = float(
                recall_score(targets, predictions, average="macro", zero_division=0)
            )
            results["f1_macro"] = float(
                f1_score(targets, predictions, average="macro", zero_division=0)
            )
            results["precision_weighted"] = float(
                precision_score(
                    targets, predictions, average="weighted", zero_division=0
                )
            )
            results["recall_weighted"] = float(
                recall_score(targets, predictions, average="weighted", zero_division=0)
            )
            results["f1_weighted"] = float(
                f1_score(targets, predictions, average="weighted", zero_division=0)
            )

            # Loss metrics
            results["average_loss"] = float(np.mean(losses))
            results["loss_std"] = float(np.std(losses))

            # Inference time metrics
            results["avg_inference_time_ms"] = float(np.mean(inference_times))
            results["inference_time_std_ms"] = float(np.std(inference_times))
            results["max_inference_time_ms"] = float(np.max(inference_times))
            results["min_inference_time_ms"] = float(np.min(inference_times))

            # Confusion matrix
            cm = confusion_matrix(targets, predictions)
            results["confusion_matrix"] = cm.tolist()

            # Per-class metrics
            per_class_precision = precision_score(
                targets, predictions, average=None, zero_division=0
            )
            per_class_recall = recall_score(
                targets, predictions, average=None, zero_division=0
            )
            per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)

            results["per_class_metrics"] = {}
            for i in range(len(self.config.class_names)):
                class_name = self.config.class_names[i]
                class_accuracy = (
                    cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0.0
                )

                results["per_class_metrics"][class_name] = {
                    "precision": float(per_class_precision[i]),
                    "recall": float(per_class_recall[i]),
                    "f1_score": float(per_class_f1[i]),
                    "accuracy": float(class_accuracy),
                    "support": int(cm[i, :].sum()),
                }

            # ROC-AUC (for multiclass)
            if self.config.calculate_roc_auc and self.config.num_classes > 2:
                try:
                    targets_binarized = label_binarize(
                        targets, classes=range(self.config.num_classes)
                    )
                    roc_auc = roc_auc_score(
                        targets_binarized,
                        probabilities,
                        average="macro",
                        multi_class="ovr",
                    )
                    results["roc_auc_macro"] = float(roc_auc)
                except Exception as e:
                    logger.warning(f"Failed to calculate ROC-AUC: {e}")
                    results["roc_auc_macro"] = None

            # Confidence statistics
            max_probabilities = np.max(probabilities, axis=1)
            results["confidence_stats"] = {
                "mean_confidence": float(np.mean(max_probabilities)),
                "std_confidence": float(np.std(max_probabilities)),
                "min_confidence": float(np.min(max_probabilities)),
                "max_confidence": float(np.max(max_probabilities)),
            }

            # Prediction distribution
            unique, counts = np.unique(predictions, return_counts=True)
            results["prediction_distribution"] = {
                str(class_idx): int(count) for class_idx, count in zip(unique, counts)
            }

            # Target distribution
            unique, counts = np.unique(targets, return_counts=True)
            results["target_distribution"] = {
                str(class_idx): int(count) for class_idx, count in zip(unique, counts)
            }

            # Detailed classification report
            if self.config.detailed_report:
                report = classification_report(
                    targets,
                    predictions,
                    target_names=self.config.class_names,
                    output_dict=True,
                    zero_division=0,
                )
                results["classification_report"] = report

            return results

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            raise EvaluationError(f"Metrics calculation failed: {e}")

    def evaluate_release_readiness(
        self, evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate if model is ready for release based on thresholds.

        Args:
            evaluation_results: Results from model evaluation

        Returns:
            Release readiness assessment
        """
        try:
            logger.info("Evaluating model release readiness...")

            checks = {}
            passed_checks = []
            failed_checks = []

            # Accuracy check
            accuracy_pass = (
                evaluation_results["accuracy"] >= self.release_thresholds.min_accuracy
            )
            checks["accuracy"] = {
                "value": evaluation_results["accuracy"],
                "threshold": self.release_thresholds.min_accuracy,
                "passed": accuracy_pass,
                "description": "Overall accuracy meets minimum threshold",
            }

            # Precision check
            precision_pass = (
                evaluation_results["precision_macro"]
                >= self.release_thresholds.min_precision
            )
            checks["precision"] = {
                "value": evaluation_results["precision_macro"],
                "threshold": self.release_thresholds.min_precision,
                "passed": precision_pass,
                "description": "Macro-averaged precision meets minimum threshold",
            }

            # Recall check
            recall_pass = (
                evaluation_results["recall_macro"] >= self.release_thresholds.min_recall
            )
            checks["recall"] = {
                "value": evaluation_results["recall_macro"],
                "threshold": self.release_thresholds.min_recall,
                "passed": recall_pass,
                "description": "Macro-averaged recall meets minimum threshold",
            }

            # F1 score check
            f1_pass = (
                evaluation_results["f1_macro"] >= self.release_thresholds.min_f1_score
            )
            checks["f1_score"] = {
                "value": evaluation_results["f1_macro"],
                "threshold": self.release_thresholds.min_f1_score,
                "passed": f1_pass,
                "description": "Macro-averaged F1 score meets minimum threshold",
            }

            # Loss check
            loss_pass = (
                evaluation_results["average_loss"] <= self.release_thresholds.max_loss
            )
            checks["loss"] = {
                "value": evaluation_results["average_loss"],
                "threshold": self.release_thresholds.max_loss,
                "passed": loss_pass,
                "description": "Average loss is below maximum threshold",
            }

            # Inference time check
            inference_time_pass = (
                evaluation_results["avg_inference_time_ms"]
                <= self.release_thresholds.max_inference_time_ms
            )
            checks["inference_time"] = {
                "value": evaluation_results["avg_inference_time_ms"],
                "threshold": self.release_thresholds.max_inference_time_ms,
                "passed": inference_time_pass,
                "description": "Average inference time meets performance requirement",
            }

            # Per-class accuracy check
            per_class_accuracies = [
                metrics["accuracy"]
                for metrics in evaluation_results["per_class_metrics"].values()
            ]
            min_class_accuracy = min(per_class_accuracies)
            per_class_pass = (
                min_class_accuracy >= self.release_thresholds.min_per_class_accuracy
            )

            checks["per_class_accuracy"] = {
                "value": min_class_accuracy,
                "threshold": self.release_thresholds.min_per_class_accuracy,
                "passed": per_class_pass,
                "description": "All classes meet minimum accuracy threshold",
            }

            # Confidence check
            mean_confidence = evaluation_results["confidence_stats"]["mean_confidence"]
            confidence_pass = (
                mean_confidence >= self.release_thresholds.min_confidence_threshold
            )

            checks["confidence"] = {
                "value": mean_confidence,
                "threshold": self.release_thresholds.min_confidence_threshold,
                "passed": confidence_pass,
                "description": "Mean prediction confidence meets minimum threshold",
            }

            # Collect results
            for check_name, check_result in checks.items():
                if check_result["passed"]:
                    passed_checks.append(check_name)
                else:
                    failed_checks.append(check_name)

            # Overall decision
            all_checks_passed = len(failed_checks) == 0
            release_recommendation = "APPROVE" if all_checks_passed else "REJECT"

            release_assessment = {
                "release_recommendation": release_recommendation,
                "all_checks_passed": all_checks_passed,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "total_checks": len(checks),
                "passed_count": len(passed_checks),
                "failed_count": len(failed_checks),
                "checks_detail": checks,
                "thresholds_used": asdict(self.release_thresholds),
            }

            # Log results
            logger.info(f"Release recommendation: {release_recommendation}")
            logger.info(f"Checks passed: {len(passed_checks)}/{len(checks)}")
            if failed_checks:
                logger.warning(f"Failed checks: {failed_checks}")

            self.logger.log_evaluation_results(release_assessment)

            return release_assessment

        except Exception as e:
            logger.error(f"Release readiness evaluation failed: {e}")
            raise EvaluationError(f"Release readiness evaluation failed: {e}")

    def save_evaluation_results(
        self, results: Dict[str, Any], output_dir: str, filename: str = None
    ) -> None:
        """
        Save evaluation results to file.

        Args:
            results: Evaluation results dictionary
            output_dir: Output directory
            filename: Optional filename
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"evaluation_results_{timestamp}.json"

            filepath = output_path / filename

            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)

            with open(filepath, "w") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"Evaluation results saved to: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
            raise EvaluationError(f"Failed to save results: {e}")

    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj


def evaluate_model(
    model: MNISTCNNModel,
    dataloader: DataLoader,
    config: Optional[Dict[str, Any]] = None,
    release_thresholds: Optional[Dict[str, Any]] = None,
    output_dir: str = "artifacts/evaluation",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    High-level model evaluation function.

    Args:
        model: Model to evaluate
        dataloader: Data loader for evaluation
        config: Optional evaluation configuration
        release_thresholds: Optional release threshold configuration
        output_dir: Directory to save results

    Returns:
        Tuple of (evaluation_results, release_assessment)
    """
    try:
        if config is None:
            config = {}
        if release_thresholds is None:
            release_thresholds = {}

        eval_config = EvaluationConfig(**config)
        thresholds = ReleaseThresholds(**release_thresholds)

        evaluator = ModelEvaluator(model, eval_config, thresholds)

        # Evaluate model
        evaluation_results = evaluator.evaluate_dataset(dataloader)

        # Check release readiness
        release_assessment = evaluator.evaluate_release_readiness(evaluation_results)

        # Save results
        evaluator.save_evaluation_results(
            evaluation_results, output_dir, "evaluation_results.json"
        )
        evaluator.save_evaluation_results(
            release_assessment, output_dir, "release_assessment.json"
        )

        return evaluation_results, release_assessment

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise EvaluationError(f"Model evaluation failed: {e}")
