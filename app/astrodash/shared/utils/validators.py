from typing import Any, List, Dict, Optional
from pydantic import validator, ValidationError as PydanticValidationError
import numpy as np
import torch
import os
import json


class ValidationError(Exception):
    """Custom validation error for model validation."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


def validate_spectrum_data(x: List[float], y: List[float]) -> None:
    """Raise ValidationError if spectrum data is invalid."""
    if not x or not y or len(x) != len(y):
        raise ValidationError("Spectrum x and y must be non-empty and of equal length.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValidationError("Spectrum data contains NaN values.")


def validate_spectrum(x: List[float], y: List[float], redshift: Optional[float] = None) -> None:
    """
    Comprehensive validation for spectrum data including x, y, and redshift.

    Args:
        x: List of wavelength values
        y: List of flux values
        redshift: Optional redshift value

    Raises:
        ValidationError: If any validation fails
    """
    # Validate spectrum data (x and y)
    validate_spectrum_data(x, y)

    # Validate redshift if provided
    if redshift is not None:
        validate_redshift(redshift)


def validate_redshift(redshift: Any) -> float:
    """Validate and return a proper redshift value (>= 0)."""
    try:
        z = float(redshift)
        if z < 0:
            raise ValidationError("Redshift must be non-negative.")
        return z
    except Exception:
        raise ValidationError("Invalid redshift value.")


def validate_file_extension(filename: str, allowed: List[str] = [".dat", ".lnw", ".txt"]) -> None:
    """Raise ValidationError if file extension is not allowed."""
    if not any(filename.lower().endswith(ext) for ext in allowed):
        raise ValidationError(f"File extension not allowed. Allowed: {allowed}")


def non_empty_list(cls, v):
    """Pydantic validator: ensure a list is not empty."""
    if not v or not isinstance(v, list) or len(v) == 0:
        raise ValueError('List must not be empty')
    return v


def validate_user_model(model_path: str, input_shape: List[int], allowed_exts: List[str] = [".pth", ".pt"]) -> None:
    """
    Validate a user-uploaded model by checking file extension, loading with torch.jit, and running a dummy input.
    Raises ValidationError if any check fails.
    """
    validate_file_extension(model_path, allowed_exts)
    if not os.path.exists(model_path):
        raise ValidationError(f"Model file does not exist: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        dummy_input = torch.randn(*input_shape)
        with torch.no_grad():
            output = model(dummy_input)
        if not hasattr(output, 'shape') or output.shape[0] != 1:
            raise ValidationError(f"Model output shape {getattr(output, 'shape', None)} is invalid.")
    except Exception as e:
        raise ValidationError(f"Failed to load or validate user model: {e}")


def validate_user_model_basic(model_path: Optional[str], class_mapping_path: Optional[str], input_shape_path: Optional[str]) -> None:
    """
    Basic validation for UserModel fields.

    Args:
        model_path: Path to the model file
        class_mapping_path: Path to the class mapping file
        input_shape_path: Path to the input shape file

    Raises:
        ValidationError: If any required field is missing
    """
    if not model_path:
        raise ValidationError("Model path is required")
    if not class_mapping_path:
        raise ValidationError("Class mapping path is required")
    if not input_shape_path:
        raise ValidationError("Input shape path is required")


def validate_class_mapping(class_mapping: Dict[str, int]) -> None:
    """
    Validate class mapping structure and integrity.

    Args:
        class_mapping: Dictionary mapping class names to indices

    Raises:
        ValidationError: If class mapping is invalid
    """
    if not isinstance(class_mapping, dict) or not class_mapping:
        raise ValidationError("Class mapping must be a non-empty dictionary")

    # Check for valid indices
    indices = list(class_mapping.values())
    if not all(isinstance(idx, int) and idx >= 0 for idx in indices):
        raise ValidationError("All class mapping values must be non-negative integers")

    # Check for unique indices
    if len(indices) != len(set(indices)):
        raise ValidationError("Class mapping indices must be unique")

    # Check for consecutive indices starting from 0
    expected_indices = set(range(len(indices)))
    if set(indices) != expected_indices:
        raise ValidationError("Class mapping indices must be consecutive starting from 0")

    # Check for valid class names
    for class_name in class_mapping.keys():
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValidationError("All class names must be non-empty strings")


def validate_input_shape(input_shape: List[int]) -> None:
    """
    Validate input shape for model compatibility.

    Args:
        input_shape: List of integers representing input shape

    Raises:
        ValidationError: If input shape is invalid
    """
    if not isinstance(input_shape, list) or not input_shape:
        raise ValidationError("Input shape must be a non-empty list")

    if not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
        raise ValidationError("All input shape dimensions must be positive integers")

    # Check for reasonable dimensions (prevent extremely large shapes)
    for dim in input_shape:
        if dim > 10000:  # Arbitrary limit
            raise ValidationError(f"Input dimension {dim} is too large. Maximum allowed: 10000")


def validate_model_compatibility(
    model_path: str,
    input_shapes: List[List[int]],
    class_mapping: Dict[str, int]
) -> Dict[str, Any]:
    """
    Comprehensive model validation including loading, shape checking, and output verification.

    Args:
        model_path: Path to the model file
        input_shapes: List of input shapes for each model input
        class_mapping: Dictionary mapping class names to indices

    Returns:
        Dictionary containing validation results and model info

    Raises:
        ValidationError: If validation fails
    """
    try:
        # Validate inputs first
        validate_class_mapping(class_mapping)
        for shape in input_shapes:
            validate_input_shape(shape)

        # Load and validate model
        if not os.path.exists(model_path):
            raise ValidationError(f"Model file does not exist: {model_path}")

        # Try loading as TorchScript
        try:
            model = torch.jit.load(model_path, map_location="cpu")
            model.eval()
            model_type = "torchscript"
        except Exception as e:
            raise ValidationError(f"Failed to load model as TorchScript: {e}")

        # Prepare dummy inputs
        dummy_inputs = []
        for shape in input_shapes:
            dummy_inputs.append(torch.randn(*shape))

        # Run inference
        with torch.no_grad():
            if len(dummy_inputs) == 1:
                output = model(dummy_inputs[0])
            elif len(dummy_inputs) == 2:
                output = model(dummy_inputs[0], dummy_inputs[1])
            elif len(dummy_inputs) == 3:
                output = model(dummy_inputs[0], dummy_inputs[1], dummy_inputs[2])
            else:
                output = model(*dummy_inputs)

        # Extract output shape
        if hasattr(output, 'shape'):
            output_shape = list(output.shape)
        else:
            output_shape = [1]  # Fallback for scalar outputs

        # Validate output shape matches class mapping
        n_classes = len(class_mapping)
        if output_shape[-1] != n_classes:
            raise ValidationError(
                f"Model output shape {output_shape} does not match "
                f"number of classes {n_classes} in class mapping."
            )

        # Collect model information
        model_info = {
            "input_shapes": input_shapes,
            "output_shape": output_shape,
            "n_classes": n_classes,
            "model_type": model_type,
            "file_size": os.path.getsize(model_path)
        }

        # Try to get model parameters count
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            })
        except Exception:
            pass  # Not critical for validation

        return model_info

    except Exception as e:
        if isinstance(e, ValidationError):
            raise e
        raise ValidationError(f"Model validation failed: {str(e)}")


def validate_json_string(json_str: str, expected_type: type = dict) -> Any:
    """
    Validate and parse a JSON string.

    Args:
        json_str: JSON string to validate
        expected_type: Expected type after parsing

    Returns:
        Parsed JSON object

    Raises:
        ValidationError: If JSON is invalid or wrong type
    """
    from astrodash.config.logging import get_logger
    logger = get_logger(__name__)

    logger.info(f"Parsing JSON string: '{json_str}', expected type: {expected_type.__name__}")

    try:
        parsed = json.loads(json_str)
        logger.info(f"JSON parsed successfully: {parsed}, type: {type(parsed)}")

        if not isinstance(parsed, expected_type):
            logger.error(f"Type mismatch: expected {expected_type.__name__}, got {type(parsed).__name__}")
            raise ValidationError(f"Expected {expected_type.__name__}, got {type(parsed).__name__}")

        logger.info(f"JSON validation passed: {parsed}")
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise ValidationError(f"Invalid JSON format: {e}")
    except Exception as e:
        logger.error(f"JSON validation error: {e}")
        raise ValidationError(f"JSON validation failed: {e}")


def validate_model_upload_request(
    filename: str,
    class_mapping_str: str,
    input_shape_str: str
) -> tuple[Dict[str, int], List[int]]:
    """
    Validate a complete model upload request.

    Args:
        filename: Name of the uploaded file
        class_mapping_str: JSON string containing class mapping
        input_shape_str: JSON string containing input shape

    Returns:
        Tuple of (class_mapping, input_shape)

    Raises:
        ValidationError: If any validation fails
    """
    from astrodash.config.logging import get_logger
    logger = get_logger(__name__)

    logger.info(f"Validating model upload request: filename={filename}")
    logger.info(f"Class mapping string: {class_mapping_str}")
    logger.info(f"Input shape string: {input_shape_str}")

    # Validate file extension
    validate_file_extension(filename, [".pth", ".pt"])

    # Parse and validate class mapping
    try:
        class_mapping = validate_json_string(class_mapping_str, dict)
        validate_class_mapping(class_mapping)
        logger.info(f"Class mapping validation passed: {class_mapping}")
    except Exception as e:
        logger.error(f"Class mapping validation failed: {e}")
        raise ValidationError(f"Invalid class mapping: {e}")

    # Parse and validate input shape
    try:
        input_shape = validate_json_string(input_shape_str, list)
        logger.info(f"Input shape parsed: {input_shape}, type: {type(input_shape)}")

        # Handle both single input shape and multiple input shapes
        if input_shape and isinstance(input_shape[0], list):
            # Multiple input shapes: [[1, 1024], [1, 1024], [1, 1]]
            logger.info(f"Validating multiple input shapes: {input_shape}")
            for i, shape in enumerate(input_shape):
                logger.info(f"Validating input shape {i}: {shape}")
                validate_input_shape(shape)
        else:
            # Single input shape: [1, 1024]
            logger.info(f"Validating single input shape: {input_shape}")
            validate_input_shape(input_shape)

        logger.info(f"Input shape validation passed: {input_shape}")
    except Exception as e:
        logger.error(f"Input shape validation failed: {e}")
        raise ValidationError(f"Invalid input shape: {e}")

    return class_mapping, input_shape
