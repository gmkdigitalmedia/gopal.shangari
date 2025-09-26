"""
Centralized logging configuration for the MLOps pipeline.

Provides consistent logging setup across all modules with
proper formatting and multiple output destinations.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """
    Set up logging configuration for the entire application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Custom log format string
    """
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

    # Create formatters
    formatter = logging.Formatter(log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        try:
            # Create log directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Use append mode to avoid conflicts
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            root_logger.addHandler(file_handler)
        except Exception as e:
            # If file logging fails, continue with console logging only
            print(f"Warning: Could not setup file logging: {e}")
            print("Continuing with console logging only...")

    # Prevent duplicate logs
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class MLOpsLogger:
    """
    Enhanced logger for MLOps operations with structured logging support.
    """

    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = get_logger(name)
        self.log_file = log_file

        if log_file:
            try:
                # Add file handler if not already present
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if file handler already exists for this file
                existing_file_handler = None
                for handler in self.logger.handlers:
                    if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path.absolute()):
                        existing_file_handler = handler
                        break

                if not existing_file_handler:
                    file_handler = logging.FileHandler(log_file, mode="a")
                    formatter = logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)
            except Exception as e:
                # If file logging fails, continue without file logging
                print(f"Warning: Could not setup file logging: {e}")
                print("Continuing without file logging...")

    def log_experiment(self, experiment_name: str, parameters: dict) -> None:
        """Log experiment parameters."""
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.logger.info(f"Parameters: {parameters}")

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log metrics with optional step information."""
        step_info = f" (step {step})" if step is not None else ""
        self.logger.info(f"Metrics{step_info}: {metrics}")

    def log_model_info(self, model_info: dict) -> None:
        """Log model information."""
        self.logger.info(f"Model info: {model_info}")

    def log_performance(self, performance_data: dict) -> None:
        """Log performance metrics."""
        self.logger.info(f"Performance: {performance_data}")

    def log_error_with_context(self, error: Exception, context: dict) -> None:
        """Log error with additional context."""
        self.logger.error(f"Error: {error}")
        self.logger.error(f"Context: {context}")

    def log_checkpoint(self, checkpoint_path: str, metrics: dict) -> None:
        """Log model checkpoint information."""
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.logger.info(f"Checkpoint metrics: {metrics}")

    def log_data_info(self, data_info: dict) -> None:
        """Log data information."""
        self.logger.info(f"Data info: {data_info}")

    def log_training_start(self, config: dict) -> None:
        """Log training start with configuration."""
        self.logger.info("=" * 50)
        self.logger.info("Starting training session")
        self.logger.info(f"Training configuration: {config}")
        self.logger.info("=" * 50)

    def log_training_end(self, final_metrics: dict, duration: float) -> None:
        """Log training end with final results."""
        self.logger.info("=" * 50)
        self.logger.info("Training completed")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Final metrics: {final_metrics}")
        self.logger.info("=" * 50)

    def log_evaluation_results(self, results: dict) -> None:
        """Log evaluation results."""
        self.logger.info("Evaluation Results:")
        for metric, value in results.items():
            self.logger.info(f"  {metric}: {value}")


def create_run_logger(run_name: str, log_dir: str = "logs") -> MLOpsLogger:
    """
    Create a logger for a specific run.

    Args:
        run_name: Name of the run
        log_dir: Directory for log files

    Returns:
        MLOpsLogger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{run_name}_{timestamp}.log"

    return MLOpsLogger(run_name, str(log_file))
