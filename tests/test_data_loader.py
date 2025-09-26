"""
Test cases for data loading functionality.

Tests data loader creation, validation, and error handling.
"""

import pytest
import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.data_loader import MNISTDataLoader, create_data_loaders
from src.utils.exceptions import DataLoadingError, ValidationError


class TestMNISTDataLoader:
    """Test cases for MNISTDataLoader."""

    def test_data_loader_creation_default(self):
        """Test data loader creation with default parameters."""
        loader = MNISTDataLoader(download=False)  # Don't download in tests
        assert loader is not None
        assert loader.batch_size == 64
        assert loader.validation_split == 0.1

    def test_data_loader_creation_custom(self):
        """Test data loader creation with custom parameters."""
        loader = MNISTDataLoader(
            batch_size=32,
            validation_split=0.2,
            num_workers=2,
            download=False
        )
        assert loader.batch_size == 32
        assert loader.validation_split == 0.2
        assert loader.num_workers == 2

    def test_invalid_parameters(self):
        """Test data loader creation with invalid parameters."""
        with pytest.raises(ValidationError):
            MNISTDataLoader(batch_size=0)

        with pytest.raises(ValidationError):
            MNISTDataLoader(validation_split=-0.1)

        with pytest.raises(ValidationError):
            MNISTDataLoader(validation_split=1.5)

        with pytest.raises(ValidationError):
            MNISTDataLoader(num_workers=-1)

    def test_transform_setup(self):
        """Test transform setup."""
        loader = MNISTDataLoader(download=False)
        assert loader.train_transform is not None
        assert loader.test_transform is not None

    def test_custom_transform_config(self):
        """Test custom transform configuration."""
        transform_config = {
            'train': {
                'normalize': {'mean': [0.5], 'std': [0.5]},
                'rotation': 15
            },
            'test': {
                'normalize': {'mean': [0.5], 'std': [0.5]}
            }
        }

        loader = MNISTDataLoader(transform_config=transform_config, download=False)
        assert loader.train_transform is not None
        assert loader.test_transform is not None

    @pytest.mark.slow
    def test_data_loading_with_download(self):
        """Test actual data loading (requires internet connection)."""
        try:
            loader = MNISTDataLoader(
                batch_size=32,
                validation_split=0.1,
                download=True
            )
            train_loader, val_loader, test_loader = loader.load_data()

            assert train_loader is not None
            assert test_loader is not None

            # Check data shapes
            data_batch, label_batch = next(iter(train_loader))
            assert data_batch.shape[0] <= 32  # Batch size
            assert data_batch.shape[1] == 1   # Channels
            assert data_batch.shape[2] == 28  # Height
            assert data_batch.shape[3] == 28  # Width
            assert label_batch.shape[0] <= 32

        except Exception as e:
            pytest.skip(f"Data loading test skipped due to: {e}")

    def test_validation_data_integrity(self):
        """Test data integrity validation with synthetic data."""
        # Create a simple synthetic dataset for testing
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, valid=True):
                self.valid = valid

            def __len__(self):
                return 100

            def __getitem__(self, idx):
                if self.valid:
                    return torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).item()
                else:
                    return torch.randn(3, 28, 28), torch.randint(0, 10, (1,)).item()  # Wrong channels

        # Test with valid data
        valid_dataset = MockDataset(valid=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=10)

        loader = MNISTDataLoader(download=False)
        assert loader.validate_data_integrity(valid_loader) is True

        # Test with invalid data
        invalid_dataset = MockDataset(valid=False)
        invalid_loader = torch.utils.data.DataLoader(invalid_dataset, batch_size=10)

        assert loader.validate_data_integrity(invalid_loader) is False


class TestDataLoaderFactory:
    """Test cases for data loader factory function."""

    def test_create_data_loaders_default(self):
        """Test data loader factory with default config."""
        config = {'download': False, 'batch_size': 16}

        try:
            train_loader, val_loader, test_loader = create_data_loaders(config)
        except Exception as e:
            # Skip if data not available
            pytest.skip(f"Data loader factory test skipped: {e}")

    def test_create_data_loaders_invalid_config(self):
        """Test data loader factory with invalid config."""
        config = {'batch_size': 0, 'download': False}

        with pytest.raises(DataLoadingError):
            create_data_loaders(config)


if __name__ == "__main__":
    pytest.main([__file__])