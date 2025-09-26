"""
Test cases for MNIST CNN Model.

Tests model creation, validation, forward pass, and serialization.
"""

import pytest
import torch
import tempfile
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.cnn_model import MNISTCNNModel, create_model
from src.utils.exceptions import ModelError, ValidationError


class TestMNISTCNNModel:
    """Test cases for MNISTCNNModel."""

    def test_model_creation_default(self):
        """Test model creation with default parameters."""
        model = MNISTCNNModel()
        assert model is not None
        assert model.num_classes == 10
        assert model.input_channels == 1

    def test_model_creation_custom(self):
        """Test model creation with custom parameters."""
        model = MNISTCNNModel(
            num_classes=5,
            conv_channels=(16, 32),
            fc_hidden_dims=(64,),
            dropout_rate=0.3
        )
        assert model.num_classes == 5
        assert model.conv_channels == (16, 32)
        assert model.fc_hidden_dims == (64,)
        assert model.dropout_rate == 0.3

    def test_model_validation(self):
        """Test model validation."""
        model = MNISTCNNModel()
        assert model.validate_model() is True

    def test_invalid_parameters(self):
        """Test model creation with invalid parameters."""
        with pytest.raises(ValidationError):
            MNISTCNNModel(num_classes=0)

        with pytest.raises(ValidationError):
            MNISTCNNModel(dropout_rate=1.5)

        with pytest.raises(ValidationError):
            MNISTCNNModel(conv_channels=())

    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = MNISTCNNModel()
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 28, 28)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_parameter_counting(self):
        """Test parameter counting."""
        model = MNISTCNNModel()
        param_count = model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)

    def test_model_info(self):
        """Test model info generation."""
        model = MNISTCNNModel()
        info = model.get_model_info()

        assert 'model_type' in info
        assert 'total_parameters' in info
        assert 'num_classes' in info
        assert info['model_type'] == 'CNN'
        assert info['num_classes'] == 10

    def test_model_serialization(self):
        """Test model saving and loading."""
        model = MNISTCNNModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pt"

            # Save model
            model.save_model(str(model_path))
            assert model_path.exists()

            # Load model
            loaded_model = MNISTCNNModel.load_model(str(model_path))
            assert loaded_model is not None

            # Test loaded model with batch size > 1 to avoid batch norm issues
            test_input = torch.randn(2, 1, 28, 28)
            with torch.no_grad():
                model.eval()  # Set to eval mode to avoid batch norm training issues
                loaded_model.eval()
                original_output = model(test_input)
                loaded_output = loaded_model(test_input)

            # Outputs should be identical
            torch.testing.assert_close(original_output, loaded_output)

    def test_create_model_factory(self):
        """Test model factory function."""
        config = {
            'num_classes': 10,
            'conv_channels': (16, 32),
            'fc_hidden_dims': (64,)
        }

        model = create_model(config)
        assert model is not None
        assert model.num_classes == 10
        assert model.conv_channels == (16, 32)

    def test_invalid_input_shape(self):
        """Test model behavior with invalid input shapes."""
        model = MNISTCNNModel()

        # Test with wrong number of dimensions
        with pytest.raises((ModelError, ValidationError, RuntimeError)):
            invalid_input = torch.randn(28, 28)  # Missing batch and channel dims
            model(invalid_input)

        # Test with wrong number of channels
        with pytest.raises((ModelError, ValidationError, RuntimeError)):
            invalid_input = torch.randn(1, 3, 28, 28)  # 3 channels instead of 1
            model(invalid_input)


class TestModelTraining:
    """Test cases for model training components."""

    def test_model_train_mode(self):
        """Test switching between train and eval modes."""
        model = MNISTCNNModel()

        # Test train mode
        model.train()
        assert model.training is True

        # Test eval mode
        model.eval()
        assert model.training is False

    def test_model_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = MNISTCNNModel()
        model.train()

        input_tensor = torch.randn(2, 1, 28, 28, requires_grad=True)
        target = torch.randint(0, 10, (2,))

        output = model(input_tensor)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


if __name__ == "__main__":
    pytest.main([__file__])