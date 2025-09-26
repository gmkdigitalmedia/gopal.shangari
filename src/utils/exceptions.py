"""
Custom exceptions for the MLOps pipeline.

Provides specific exception types for different failure modes
to enable better error handling and debugging.
"""


class MLOpsError(Exception):
    """Base exception for MLOps pipeline errors."""

    pass


class DataLoadingError(MLOpsError):
    """Exception raised when data loading fails."""

    pass


class ValidationError(MLOpsError):
    """Exception raised when validation fails."""

    pass


class ModelError(MLOpsError):
    """Exception raised when model operations fail."""

    pass


class TrainingError(MLOpsError):
    """Exception raised when training fails."""

    pass


class EvaluationError(MLOpsError):
    """Exception raised when evaluation fails."""

    pass


class ConfigurationError(MLOpsError):
    """Exception raised when configuration is invalid."""

    pass


class ModelReleaseError(MLOpsError):
    """Exception raised when model release criteria are not met."""

    pass
