"""
Main MLOps Pipeline Orchestrator for MNIST Classification

This script orchestrates the complete MLOps pipeline including:
- Data loading and validation
- Model creation and training
- Model evaluation and release decision
- Comprehensive logging and artifact management

Usage:
    python main.py [--config CONFIG_FILE] [--mode MODE] [--model-path MODEL_PATH]

Modes:
    - train: Train a new model (default)
    - evaluate: Evaluate an existing model
    - pipeline: Run complete pipeline (train + evaluate)
"""

import sys
import time
import argparse
import torch
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import create_data_loaders
from src.models import create_model, MNISTCNNModel
from src.training import train_model
from src.evaluation import evaluate_model
from src.utils import setup_logging, get_logger, create_run_logger
from src.utils.exceptions import MLOpsError
from config.config import load_config, MLOpsConfig

logger = get_logger(__name__)


class MLOpsPipeline:
    """
    Complete MLOps Pipeline orchestrator.

    Manages the entire ML lifecycle from data loading to model release decisions.
    """

    def __init__(self, config: MLOpsConfig):
        """
        Initialize MLOps pipeline.

        Args:
            config: Complete MLOps configuration
        """
        self.config = config
        self.run_logger = create_run_logger(
            self.config.experiment_name, self.config.logging.log_file or "logs"
        )

        # Set random seed for reproducibility
        self._set_random_seed()

        # Create output directories
        self._setup_directories()

    def _set_random_seed(self) -> None:
        """Set random seeds for reproducibility."""
        try:
            import random
            import numpy as np

            torch.manual_seed(self.config.random_seed)
            torch.cuda.manual_seed_all(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)

            # Ensure deterministic behavior for CUDA
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            logger.info(f"Random seed set to: {self.config.random_seed}")

        except Exception as e:
            logger.warning(f"Failed to set random seed: {e}")

    def _setup_directories(self) -> None:
        """Create necessary output directories."""
        try:
            directories = [
                self.config.output_dir,
                f"{self.config.output_dir}/models",
                f"{self.config.output_dir}/evaluation",
                self.config.training.checkpoint_dir,
                "logs",
            ]

            for directory in directories:
                dir_path = Path(directory)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)

            logger.info("Output directories created successfully")

        except Exception as e:
            logger.warning(f"Some directories may already exist: {e}")
            # Continue execution - directories might already exist

    def load_data(self) -> tuple:
        """
        Load and validate data.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        try:
            logger.info("Loading data...")
            self.run_logger.log_data_info(
                {
                    "data_dir": self.config.data.data_dir,
                    "batch_size": self.config.data.batch_size,
                    "validation_split": self.config.data.validation_split,
                }
            )

            # Create data loaders
            data_config = {
                "data_dir": self.config.data.data_dir,
                "batch_size": self.config.data.batch_size,
                "validation_split": self.config.data.validation_split,
                "num_workers": self.config.data.num_workers,
                "pin_memory": self.config.data.pin_memory,
                "download": self.config.data.download,
                "transform_config": self.config.data.transform_config,
            }

            train_loader, val_loader, test_loader = create_data_loaders(data_config)

            # Validate data integrity
            from src.data.data_loader import MNISTDataLoader

            data_loader_instance = MNISTDataLoader(**data_config)

            if not data_loader_instance.validate_data_integrity(train_loader):
                raise MLOpsError("Training data integrity validation failed")

            if test_loader and not data_loader_instance.validate_data_integrity(
                test_loader
            ):
                raise MLOpsError("Test data integrity validation failed")

            logger.info("Data loading completed successfully")
            return train_loader, val_loader, test_loader

        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise MLOpsError(f"Data loading failed: {e}")

    def create_model(self) -> "MNISTCNNModel":
        """
        Create and validate model.

        Returns:
            Initialized model
        """
        try:
            logger.info("Creating model...")

            model_config = {
                "num_classes": self.config.model.num_classes,
                "input_channels": self.config.model.input_channels,
                "conv_channels": self.config.model.conv_channels,
                "fc_hidden_dims": self.config.model.fc_hidden_dims,
                "dropout_rate": self.config.model.dropout_rate,
                "use_batch_norm": self.config.model.use_batch_norm,
                "activation": self.config.model.activation,
            }

            model = create_model(model_config)

            # Log model information
            model_info = model.get_model_info()
            self.run_logger.log_model_info(model_info)

            logger.info(
                f"Model created with {model_info['total_parameters']:,} parameters"
            )
            return model

        except Exception as e:
            logger.error(f"Model creation failed: {e}")
            raise MLOpsError(f"Model creation failed: {e}")

    def train_model(self, model, train_loader, val_loader) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training results
        """
        try:
            logger.info("Starting model training...")

            training_config = {
                "epochs": self.config.training.epochs,
                "learning_rate": self.config.training.learning_rate,
                "optimizer": self.config.training.optimizer,
                "scheduler": self.config.training.scheduler,
                "scheduler_params": self.config.training.scheduler_params,
                "criterion": self.config.training.criterion,
                "early_stopping_patience": self.config.training.early_stopping_patience,
                "save_checkpoints": self.config.training.save_checkpoints,
                "checkpoint_dir": self.config.training.checkpoint_dir,
                "save_best_only": self.config.training.save_best_only,
                "validation_frequency": self.config.training.validation_frequency,
                "device": self.config.training.device,
                "mixed_precision": self.config.training.mixed_precision,
                "gradient_clip_val": self.config.training.gradient_clip_val,
            }

            training_results = train_model(
                model, train_loader, val_loader, training_config
            )

            logger.info("Model training completed successfully")
            return training_results

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise MLOpsError(f"Model training failed: {e}")

    def evaluate_model(self, model, test_loader) -> tuple:
        """
        Evaluate the model and make release decision.

        Args:
            model: Model to evaluate
            test_loader: Test data loader

        Returns:
            Tuple of (evaluation_results, release_assessment)
        """
        try:
            logger.info("Starting model evaluation...")

            evaluation_config = {
                "device": self.config.evaluation.device,
                "batch_size": self.config.evaluation.batch_size,
                "num_classes": self.config.evaluation.num_classes,
                "class_names": self.config.evaluation.class_names,
                "save_predictions": self.config.evaluation.save_predictions,
                "save_probabilities": self.config.evaluation.save_probabilities,
                "calculate_roc_auc": self.config.evaluation.calculate_roc_auc,
                "detailed_report": self.config.evaluation.detailed_report,
            }

            release_thresholds = {
                "min_accuracy": self.config.release_thresholds.min_accuracy,
                "min_precision": self.config.release_thresholds.min_precision,
                "min_recall": self.config.release_thresholds.min_recall,
                "min_f1_score": self.config.release_thresholds.min_f1_score,
                "max_loss": self.config.release_thresholds.max_loss,
                "min_per_class_accuracy": self.config.release_thresholds.min_per_class_accuracy,
                "max_inference_time_ms": self.config.release_thresholds.max_inference_time_ms,
                "min_confidence_threshold": self.config.release_thresholds.min_confidence_threshold,
            }

            output_dir = f"{self.config.output_dir}/evaluation"

            evaluation_results, release_assessment = evaluate_model(
                model, test_loader, evaluation_config, release_thresholds, output_dir
            )

            logger.info("Model evaluation completed successfully")
            return evaluation_results, release_assessment

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise MLOpsError(f"Model evaluation failed: {e}")

    def save_final_model(
        self,
        model,
        training_results: Dict[str, Any],
        evaluation_results: Dict[str, Any],
    ) -> str:
        """
        Save the final model with comprehensive metadata.

        Args:
            model: Trained model
            training_results: Training results
            evaluation_results: Evaluation results

        Returns:
            Path to saved model
        """
        try:
            model_dir = Path(self.config.output_dir) / "models"
            model_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_filename = f"{self.config.experiment_name}_{timestamp}.pt"
            model_path = model_dir / model_filename

            # Create comprehensive save data
            save_data = {
                "model_state_dict": model.state_dict(),
                "model_info": model.get_model_info(),
                "config": self.config.__dict__,
                "training_results": training_results,
                "evaluation_results": evaluation_results,
                "timestamp": timestamp,
                "experiment_name": self.config.experiment_name,
            }

            torch.save(save_data, model_path)

            # Also save model architecture separately
            model.save_model(model_path.with_suffix(".model.pt"))

            logger.info(f"Final model saved to: {model_path}")
            self.run_logger.log_checkpoint(str(model_path), evaluation_results)

            return str(model_path)

        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
            raise MLOpsError(f"Final model saving failed: {e}")

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete MLOps pipeline.

        Returns:
            Complete pipeline results
        """
        try:
            pipeline_start_time = time.time()

            logger.info("=" * 70)
            logger.info(f"Starting MLOps Pipeline: {self.config.experiment_name}")
            logger.info("=" * 70)

            # Step 1: Load Data
            train_loader, val_loader, test_loader = self.load_data()

            # Step 2: Create Model
            model = self.create_model()

            # Step 3: Train Model
            training_results = self.train_model(model, train_loader, val_loader)

            # Step 4: Evaluate Model
            evaluation_results, release_assessment = self.evaluate_model(
                model, test_loader
            )

            # Step 5: Save Final Model
            final_model_path = self.save_final_model(
                model, training_results, evaluation_results
            )

            # Step 6: Generate Pipeline Summary
            pipeline_duration = time.time() - pipeline_start_time

            pipeline_results = {
                "experiment_name": self.config.experiment_name,
                "pipeline_duration": pipeline_duration,
                "training_results": training_results,
                "evaluation_results": evaluation_results,
                "release_assessment": release_assessment,
                "final_model_path": final_model_path,
                "config_used": self.config.__dict__,
            }

            # Log final summary
            self._log_pipeline_summary(pipeline_results)

            logger.info("=" * 70)
            logger.info("MLOps Pipeline completed successfully!")
            logger.info(
                f"Release Recommendation: {release_assessment['release_recommendation']}"
            )
            logger.info("=" * 70)

            return pipeline_results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise MLOpsError(f"Pipeline execution failed: {e}")

    def _log_pipeline_summary(self, results: Dict[str, Any]) -> None:
        """Log comprehensive pipeline summary."""
        try:
            logger.info("\n" + "=" * 50)
            logger.info("PIPELINE SUMMARY")
            logger.info("=" * 50)

            # Training summary
            training_results = results["training_results"]
            final_metrics = training_results["final_metrics"]

            logger.info("Training Results:")
            logger.info(f"  Duration: {training_results['training_time']:.2f} seconds")
            logger.info(f"  Best Epoch: {training_results['best_epoch']}")
            logger.info(
                f"  Final Train Accuracy: {final_metrics.get('latest_train_accuracy', 0):.2f}%"
            )
            logger.info(
                f"  Final Val Accuracy: {final_metrics.get('latest_val_accuracy', 0):.2f}%"
            )

            # Evaluation summary
            evaluation_results = results["evaluation_results"]
            logger.info("\nEvaluation Results:")
            logger.info(f"  Test Accuracy: {evaluation_results['accuracy']:.4f}")
            logger.info(f"  Test Loss: {evaluation_results['average_loss']:.4f}")
            logger.info(f"  F1 Score: {evaluation_results['f1_macro']:.4f}")
            logger.info(
                f"  Inference Time: {evaluation_results['avg_inference_time_ms']:.2f} ms"
            )

            # Release decision
            release_assessment = results["release_assessment"]
            logger.info("\nRelease Assessment:")
            logger.info(
                f"  Recommendation: {release_assessment['release_recommendation']}"
            )
            logger.info(
                f"  Checks Passed: {release_assessment['passed_count']}/"
                f"{release_assessment['total_checks']}"
            )

            if release_assessment["failed_checks"]:
                logger.info(f"  Failed Checks: {release_assessment['failed_checks']}")

            logger.info(
                f"\nTotal Pipeline Duration: {results['pipeline_duration']:.2f} seconds"
            )
            logger.info(f"Final Model Saved: {results['final_model_path']}")
            logger.info("=" * 50)

        except Exception as e:
            logger.warning(f"Failed to log pipeline summary: {e}")


def main():
    """Main entry point for the MLOps pipeline."""
    parser = argparse.ArgumentParser(description="MNIST MLOps Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "pipeline"],
        default="pipeline",
        help="Pipeline execution mode",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to existing model (for evaluate mode)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Setup logging
        try:
            setup_logging(
                log_level=config.logging.log_level,
                log_file=config.logging.log_file,
                log_format=config.logging.log_format,
            )
        except Exception as e:
            print(f"Warning: Logging setup failed, using console only: {e}")
            setup_logging(log_level=config.logging.log_level)

        # Initialize pipeline
        pipeline = MLOpsPipeline(config)

        if args.mode == "pipeline":
            # Run complete pipeline
            results = pipeline.run_complete_pipeline()

            # Print final recommendation
            print(
                f"\nModel Release Recommendation: "
                f"{results['release_assessment']['release_recommendation']}"
            )
            print(f"Test Accuracy: {results['evaluation_results']['accuracy']:.4f}")
            print(f"Model saved to: {results['final_model_path']}")

        elif args.mode == "train":
            # Train only
            train_loader, val_loader, test_loader = pipeline.load_data()
            model = pipeline.create_model()
            training_results = pipeline.train_model(model, train_loader, val_loader)

            print("\nTraining completed!")
            print(
                f"Best validation accuracy: {training_results['final_metrics'].get('best_val_accuracy', 0):.4f}"
            )

        elif args.mode == "evaluate":
            # Evaluate existing model
            if not args.model_path:
                raise ValueError("Model path required for evaluation mode")

            from src.models.cnn_model import MNISTCNNModel

            model = MNISTCNNModel.load_model(args.model_path)

            _, _, test_loader = pipeline.load_data()
            evaluation_results, release_assessment = pipeline.evaluate_model(
                model, test_loader
            )

            print(
                f"\nModel Release Recommendation: {release_assessment['release_recommendation']}"
            )
            print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
