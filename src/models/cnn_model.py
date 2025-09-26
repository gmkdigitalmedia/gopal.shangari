"""
CNN Model for MNIST classification with comprehensive validation and error handling.

Provides a flexible CNN architecture with configurable layers,
proper initialization, and extensive validation capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

from ..utils.exceptions import ModelError, ValidationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MNISTCNNModel(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.

    Features:
    - Configurable architecture
    - Batch normalization and dropout for regularization
    - Proper weight initialization
    - Model validation and diagnostics
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        conv_channels: Tuple[int, ...] = (32, 64, 128),
        fc_hidden_dims: Tuple[int, ...] = (256, 128),
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Initialize CNN model.

        Args:
            num_classes: Number of output classes (10 for MNIST)
            input_channels: Number of input channels (1 for grayscale MNIST)
            conv_channels: Tuple of output channels for convolutional layers
            fc_hidden_dims: Tuple of hidden dimensions for fully connected layers
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function name
        """
        super(MNISTCNNModel, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.fc_hidden_dims = fc_hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation

        self._validate_parameters()
        self._build_model()
        self._initialize_weights()

        logger.info(f"Initialized CNN model with {self.count_parameters()} parameters")

    def _validate_parameters(self) -> None:
        """Validate model parameters."""
        if self.num_classes <= 0:
            raise ValidationError("Number of classes must be positive")

        if self.input_channels <= 0:
            raise ValidationError("Input channels must be positive")

        if len(self.conv_channels) == 0:
            raise ValidationError("Must have at least one convolutional layer")

        if any(c <= 0 for c in self.conv_channels):
            raise ValidationError("All convolutional channels must be positive")

        if len(self.fc_hidden_dims) == 0:
            raise ValidationError("Must have at least one fully connected layer")

        if any(d <= 0 for d in self.fc_hidden_dims):
            raise ValidationError("All hidden dimensions must be positive")

        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValidationError("Dropout rate must be between 0 and 1")

        if self.activation_name not in ['relu', 'leaky_relu', 'tanh', 'sigmoid']:
            raise ValidationError(f"Unsupported activation: {self.activation_name}")

    def _get_activation(self) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations[self.activation_name]

    def _build_model(self) -> None:
        """Build the CNN architecture."""
        try:
            # Convolutional layers
            self.conv_layers = nn.ModuleList()
            self.bn_layers = nn.ModuleList() if self.use_batch_norm else None

            in_channels = self.input_channels
            for i, out_channels in enumerate(self.conv_channels):
                # Convolutional layer
                conv_layer = nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=3, padding=1
                )
                self.conv_layers.append(conv_layer)

                # Batch normalization
                if self.use_batch_norm:
                    self.bn_layers.append(nn.BatchNorm2d(out_channels))

                in_channels = out_channels

            # Adaptive pooling to handle different input sizes
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

            # Calculate the size after convolutions and pooling
            conv_output_size = self.conv_channels[-1] * 4 * 4

            # Fully connected layers
            self.fc_layers = nn.ModuleList()
            self.fc_bn_layers = nn.ModuleList() if self.use_batch_norm else None

            fc_input_size = conv_output_size
            for i, hidden_dim in enumerate(self.fc_hidden_dims):
                fc_layer = nn.Linear(fc_input_size, hidden_dim)
                self.fc_layers.append(fc_layer)

                if self.use_batch_norm:
                    self.fc_bn_layers.append(nn.BatchNorm1d(hidden_dim))

                fc_input_size = hidden_dim

            # Output layer
            self.output_layer = nn.Linear(fc_input_size, self.num_classes)

            # Dropout
            self.dropout = nn.Dropout(self.dropout_rate)

            # Activation
            self.activation = self._get_activation()

            logger.info("Model architecture built successfully")

        except Exception as e:
            raise ModelError(f"Failed to build model architecture: {e}")

    def _initialize_weights(self) -> None:
        """Initialize model weights using appropriate methods."""
        try:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(module.bias, 0)

            logger.info("Model weights initialized successfully")

        except Exception as e:
            raise ModelError(f"Failed to initialize weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        try:
            # Validate input
            if x.dim() != 4:
                raise ValidationError(f"Expected 4D input tensor, got {x.dim()}D")

            if x.size(1) != self.input_channels:
                raise ValidationError(
                    f"Expected {self.input_channels} input channels, got {x.size(1)}"
                )

            # Convolutional layers
            for i, conv_layer in enumerate(self.conv_layers):
                x = conv_layer(x)

                if self.use_batch_norm:
                    x = self.bn_layers[i](x)

                x = self.activation(x)
                x = F.max_pool2d(x, kernel_size=2)

            # Adaptive pooling
            x = self.adaptive_pool(x)

            # Flatten
            x = x.view(x.size(0), -1)

            # Fully connected layers
            for i, fc_layer in enumerate(self.fc_layers):
                x = fc_layer(x)

                if self.use_batch_norm:
                    x = self.fc_bn_layers[i](x)

                x = self.activation(x)
                x = self.dropout(x)

            # Output layer
            x = self.output_layer(x)

            return x

        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise ModelError(f"Forward pass failed: {e}")

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': 'CNN',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'conv_channels': self.conv_channels,
            'fc_hidden_dims': self.fc_hidden_dims,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation_name,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

    def validate_model(self, input_shape: Tuple[int, ...] = (1, 1, 28, 28)) -> bool:
        """
        Validate model by running a forward pass with dummy data.

        Args:
            input_shape: Shape of input tensor for validation

        Returns:
            True if validation passes, False otherwise
        """
        try:
            logger.info("Validating model architecture...")

            # Create dummy input
            dummy_input = torch.randn(*input_shape)

            # Set model to evaluation mode
            self.eval()

            with torch.no_grad():
                output = self(dummy_input)

                # Check output shape
                expected_shape = (input_shape[0], self.num_classes)
                if output.shape != expected_shape:
                    logger.error(f"Output shape mismatch: expected {expected_shape}, got {output.shape}")
                    return False

                # Check for NaN or infinite values
                if torch.isnan(output).any() or torch.isinf(output).any():
                    logger.error("Model output contains NaN or infinite values")
                    return False

            logger.info("Model validation passed")
            return True

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def save_model(self, filepath: str, include_config: bool = True) -> None:
        """
        Save model state and configuration.

        Args:
            filepath: Path to save the model
            include_config: Whether to save model configuration
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save model state
            save_dict = {
                'model_state_dict': self.state_dict(),
                'model_info': self.get_model_info()
            }

            torch.save(save_dict, filepath)

            # Save configuration separately
            if include_config:
                config_path = filepath.with_suffix('.json')
                with open(config_path, 'w') as f:
                    json.dump(self.get_model_info(), f, indent=2)

            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise ModelError(f"Model saving failed: {e}")

    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu') -> 'MNISTCNNModel':
        """
        Load model from file.

        Args:
            filepath: Path to the saved model
            device: Device to load the model on

        Returns:
            Loaded model instance
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise ModelError(f"Model file not found: {filepath}")

            # Load saved data
            checkpoint = torch.load(filepath, map_location=device)
            model_info = checkpoint['model_info']

            # Create model instance
            model = cls(
                num_classes=model_info['num_classes'],
                input_channels=model_info['input_channels'],
                conv_channels=tuple(model_info['conv_channels']),
                fc_hidden_dims=tuple(model_info['fc_hidden_dims']),
                dropout_rate=model_info['dropout_rate'],
                use_batch_norm=model_info['use_batch_norm'],
                activation=model_info['activation']
            )

            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])

            logger.info(f"Model loaded from {filepath}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelError(f"Model loading failed: {e}")


def create_model(config: Dict[str, Any]) -> MNISTCNNModel:
    """
    Factory function to create a model from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model instance
    """
    try:
        model = MNISTCNNModel(**config)

        # Validate the created model
        if not model.validate_model():
            raise ModelError("Model validation failed after creation")

        return model

    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise ModelError(f"Model creation failed: {e}")