"""
Configuration management for the MLOps pipeline.

Provides centralized configuration handling with validation,
environment variable support, and hierarchical configuration loading.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataConfig:
    """Data loading configuration."""

    data_dir: str = "./data"
    batch_size: int = 64
    validation_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    download: bool = True
    transform_config: Optional[Dict[str, Any]] = None


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    num_classes: int = 10
    input_channels: int = 1
    conv_channels: tuple = (32, 64, 128)
    fc_hidden_dims: tuple = (256, 128)
    dropout_rate: float = 0.5
    use_batch_norm: bool = True
    activation: str = "relu"


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    scheduler: str = "plateau"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    criterion: str = "cross_entropy"
    early_stopping_patience: int = 5
    save_checkpoints: bool = True
    checkpoint_dir: str = "artifacts/models/checkpoints"
    save_best_only: bool = True
    validation_frequency: int = 1
    device: str = "auto"
    mixed_precision: bool = False
    gradient_clip_val: Optional[float] = None


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    device: str = "auto"
    batch_size: int = 64
    num_classes: int = 10
    class_names: list = field(default_factory=lambda: [str(i) for i in range(10)])
    save_predictions: bool = True
    save_probabilities: bool = True
    calculate_roc_auc: bool = True
    detailed_report: bool = True


@dataclass
class ReleaseThresholds:
    """Model release thresholds."""

    min_accuracy: float = 0.95
    min_precision: float = 0.9
    min_recall: float = 0.9
    min_f1_score: float = 0.9
    max_loss: float = 0.2
    min_per_class_accuracy: float = 0.8
    max_inference_time_ms: float = 100.0
    min_confidence_threshold: float = 0.7


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_level: str = "INFO"
    log_file: Optional[str] = "logs/mlops_pipeline.log"
    log_format: Optional[str] = None
    console_logging: bool = True
    file_logging: bool = True


@dataclass
class MLOpsConfig:
    """Complete MLOps pipeline configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    release_thresholds: ReleaseThresholds = field(default_factory=ReleaseThresholds)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # General pipeline settings
    experiment_name: str = "mnist_cnn_experiment"
    random_seed: int = 42
    output_dir: str = "artifacts"
    save_intermediate_results: bool = True


class ConfigManager:
    """
    Configuration manager with support for multiple sources and environments.

    Features:
    - YAML and JSON configuration files
    - Environment variable overrides
    - Configuration validation
    - Hierarchical configuration merging
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_config(
        self,
        config_file: Optional[str] = None,
        env_prefix: str = "MLOPS_",
        validate: bool = True,
    ) -> MLOpsConfig:
        """
        Load configuration from multiple sources.

        Args:
            config_file: Optional path to configuration file
            env_prefix: Prefix for environment variables
            validate: Whether to validate configuration

        Returns:
            Complete MLOps configuration
        """
        try:
            logger.info("Loading configuration...")

            # Start with default configuration
            config_dict = asdict(MLOpsConfig())

            # Load from file if specified
            if config_file:
                file_config = self._load_config_file(config_file)
                config_dict = self._merge_configs(config_dict, file_config)

            # Override with environment variables
            env_config = self._load_env_config(env_prefix)
            config_dict = self._merge_configs(config_dict, env_config)

            # Create configuration object
            config = self._dict_to_config(config_dict)

            # Validate configuration
            if validate:
                self._validate_config(config)

            logger.info("Configuration loaded successfully")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")

    def _load_config_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            config_path = Path(config_file)

            # If path doesn't exist and is relative, try relative to current directory
            if not config_path.exists() and not config_path.is_absolute():
                config_path = Path.cwd() / config_file

            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_file}")
                return {}

            if (
                config_path.suffix.lower() == ".yaml"
                or config_path.suffix.lower() == ".yml"
            ):
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == ".json":
                with open(config_path, "r") as f:
                    return json.load(f) or {}
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )

        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
            raise ConfigurationError(f"Failed to load config file: {e}")

    def _load_env_config(self, prefix: str) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        try:
            env_config = {}

            for key, value in os.environ.items():
                if key.startswith(prefix):
                    # Convert environment variable to config key
                    config_key = key[len(prefix) :].lower()

                    # Parse nested keys (e.g., MLOPS_TRAINING_EPOCHS -> training.epochs)
                    key_parts = config_key.split("_")
                    current_dict = env_config

                    for part in key_parts[:-1]:
                        if part not in current_dict:
                            current_dict[part] = {}
                        current_dict = current_dict[part]

                    # Convert value to appropriate type
                    final_key = key_parts[-1]
                    current_dict[final_key] = self._parse_env_value(value)

            return env_config

        except Exception as e:
            logger.error(f"Failed to load environment configuration: {e}")
            return {}

    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type."""
        # Boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _merge_configs(
        self, base_config: Dict[str, Any], override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge configuration dictionaries."""
        result = base_config.copy()

        for key, value in override_config.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> MLOpsConfig:
        """Convert dictionary to configuration object."""
        try:
            # Extract nested configurations
            data_config = DataConfig(**config_dict.get("data", {}))
            model_config = ModelConfig(**config_dict.get("model", {}))
            training_config = TrainingConfig(**config_dict.get("training", {}))
            evaluation_config = EvaluationConfig(**config_dict.get("evaluation", {}))
            release_thresholds = ReleaseThresholds(
                **config_dict.get("release_thresholds", {})
            )
            logging_config = LoggingConfig(**config_dict.get("logging", {}))

            # Create main configuration
            main_config_dict = {
                k: v
                for k, v in config_dict.items()
                if k
                not in [
                    "data",
                    "model",
                    "training",
                    "evaluation",
                    "release_thresholds",
                    "logging",
                ]
            }

            config = MLOpsConfig(
                data=data_config,
                model=model_config,
                training=training_config,
                evaluation=evaluation_config,
                release_thresholds=release_thresholds,
                logging=logging_config,
                **main_config_dict,
            )

            return config

        except Exception as e:
            logger.error(f"Failed to convert dictionary to configuration: {e}")
            raise ConfigurationError(f"Configuration conversion failed: {e}")

    def _validate_config(self, config: MLOpsConfig) -> None:
        """Validate configuration values."""
        try:
            # Validate data configuration
            if config.data.batch_size <= 0:
                raise ConfigurationError("Batch size must be positive")

            if not (0.0 <= config.data.validation_split <= 1.0):
                raise ConfigurationError("Validation split must be between 0 and 1")

            # Validate model configuration
            if config.model.num_classes <= 0:
                raise ConfigurationError("Number of classes must be positive")

            if config.model.dropout_rate < 0.0 or config.model.dropout_rate > 1.0:
                raise ConfigurationError("Dropout rate must be between 0 and 1")

            # Validate training configuration
            if config.training.epochs <= 0:
                raise ConfigurationError("Number of epochs must be positive")

            if config.training.learning_rate <= 0:
                raise ConfigurationError("Learning rate must be positive")

            # Validate release thresholds
            if not (0.0 <= config.release_thresholds.min_accuracy <= 1.0):
                raise ConfigurationError("Minimum accuracy must be between 0 and 1")

            logger.info("Configuration validation passed")

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ConfigurationError(f"Invalid configuration: {e}")

    def save_config(self, config: MLOpsConfig, filename: str) -> None:
        """Save configuration to file."""
        try:
            filepath = self.config_dir / filename

            config_dict = asdict(config)

            if filename.endswith(".yaml") or filename.endswith(".yml"):
                with open(filepath, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif filename.endswith(".json"):
                with open(filepath, "w") as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported file format: {filename}")

            logger.info(f"Configuration saved to: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ConfigurationError(f"Configuration saving failed: {e}")

    def create_default_config(self, filename: str = "default_config.yaml") -> None:
        """Create a default configuration file."""
        try:
            default_config = MLOpsConfig()
            self.save_config(default_config, filename)
            logger.info(f"Default configuration created: {self.config_dir / filename}")

        except Exception as e:
            logger.error(f"Failed to create default configuration: {e}")
            raise ConfigurationError(f"Failed to create default configuration: {e}")


# Global configuration manager instance
config_manager = ConfigManager()


def load_config(config_file: Optional[str] = None, **kwargs) -> MLOpsConfig:
    """
    Convenience function to load configuration.

    Args:
        config_file: Optional path to configuration file
        **kwargs: Additional arguments for config loading

    Returns:
        MLOps configuration
    """
    return config_manager.load_config(config_file, **kwargs)
