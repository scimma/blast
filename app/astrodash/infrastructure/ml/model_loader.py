import torch
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from astrodash.config.logging import get_logger

logger = get_logger(__name__)

class ModelLoader:
    """
    Infrastructure component for loading and validating PyTorch models.
    Handles model loading, shape inference, and validation operations.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        logger.info(f"ModelLoader initialized with device: {self.device}")

    def load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load a PyTorch model from file.

        Args:
            model_path: Path to the model file (.pth or .pt)

        Returns:
            Loaded PyTorch model

        Raises:
            ValueError: If model cannot be loaded
        """
        try:
            if not os.path.exists(model_path):
                raise ValueError(f"Model file does not exist: {model_path}")

            # Basic file validation
            file_size = os.path.getsize(model_path)
            if file_size == 0:
                raise ValueError(f"Model file is empty: {model_path}")
            if file_size < 100:  # Very small files are likely not valid models
                raise ValueError(f"Model file is too small ({file_size} bytes) to be a valid PyTorch model: {model_path}")

            # Try loading as TorchScript first
            try:
                model = torch.jit.load(model_path, map_location=self.device)
                logger.info(f"Loaded TorchScript model from {model_path}")
                model.eval()
                return model
            except Exception as e:
                logger.warning(f"Failed to load as TorchScript: {e}")

                # Try loading as state dict
                try:
                    state_dict = torch.load(model_path, map_location=self.device)
                    logger.info(f"Loaded state dict from {model_path}")

                    # For now, we'll raise an error for state dict models
                    # since we need the model architecture to reconstruct them
                    raise ValueError(
                        f"State dict models are not yet supported. "
                        f"Please export your model as TorchScript using one of these methods:\n"
                        f"1. For a trained model: torch.jit.script(model)\n"
                        f"2. For a traced model: torch.jit.trace(model, example_input)\n"
                        f"3. Save with: torch.jit.save(scripted_model, 'model.pt')\n"
                        f"Original error: {str(e)}"
                    )
                except Exception as state_dict_error:
                    if "State dict models are not yet supported" in str(state_dict_error):
                        raise state_dict_error
                    else:
                        raise ValueError(
                            f"Failed to load model from {model_path}. "
                            f"Model must be a valid TorchScript (.pt) file.\n"
                            f"Common issues:\n"
                            f"1. Model is not saved as TorchScript\n"
                            f"2. Model file is corrupted\n"
                            f"3. Model was saved with incompatible PyTorch version\n"
                            f"TorchScript error: {str(e)}\n"
                            f"State dict error: {str(state_dict_error)}"
                        )

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise ValueError(f"Failed to load model: {str(e)}")

    def validate_model_with_inputs(
        self,
        model: torch.nn.Module,
        input_shapes: List[List[int]],
        class_mapping: Dict[str, int]
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Validate model by running dummy inputs and checking output shape.

        Args:
            model: Loaded PyTorch model
            input_shapes: List of input shapes for each model input
            class_mapping: Dictionary mapping class names to indices

        Returns:
            Tuple of (output_shape, model_info)

        Raises:
            ValueError: If validation fails
        """
        try:
            # Detect model input requirements
            num_inputs = len(input_shapes)  # Default to provided input shapes

            if hasattr(model, 'forward'):
                # Try to inspect the signature, but handle TorchScript models gracefully
                try:
                    import inspect
                    sig = inspect.signature(model.forward)
                    # Count non-self parameters
                    num_inputs = len([p for p in sig.parameters.values() if p.name != 'self' and p.default == inspect.Parameter.empty])
                    logger.info(f"Detected {num_inputs} inputs from model signature")
                except Exception as e:
                    logger.warning(f"Could not inspect model signature: {e}. Using provided input shapes.")
                    num_inputs = len(input_shapes)

            # If model expects more inputs than provided, create additional dummy inputs
            if num_inputs > len(input_shapes):
                logger.info(f"Model expects {num_inputs} inputs but only {len(input_shapes)} provided. Creating additional dummy inputs.")

                # For transformer models, create appropriate dummy inputs
                if num_inputs == 3:  # wavelength, flux, redshift
                    # Use the provided input shape for flux, create wavelength and redshift
                    flux_shape = input_shapes[0]
                    wavelength_shape = flux_shape  # Same shape as flux
                    redshift_shape = [flux_shape[0], 1]  # [batch_size, 1]

                    dummy_inputs = [
                        torch.randn(*wavelength_shape, device=self.device),  # wavelength
                        torch.randn(*flux_shape, device=self.device),        # flux
                        torch.randn(*redshift_shape, device=self.device)     # redshift
                    ]
                elif num_inputs == 4:  # wavelength, flux, redshift, mask
                    # Use the provided input shape for flux, create others
                    flux_shape = input_shapes[0]
                    wavelength_shape = flux_shape  # Same shape as flux
                    redshift_shape = [flux_shape[0], 1]  # [batch_size, 1]
                    mask_shape = flux_shape  # Same shape as flux

                    dummy_inputs = [
                        torch.randn(*wavelength_shape, device=self.device),  # wavelength
                        torch.randn(*flux_shape, device=self.device),        # flux
                        torch.randn(*redshift_shape, device=self.device),    # redshift
                        torch.ones(*mask_shape, device=self.device)          # mask (all ones)
                    ]
                else:
                    # Generic case: create additional dummy inputs with the same shape
                    base_shape = input_shapes[0]
                    dummy_inputs = [torch.randn(*shape, device=self.device) for shape in input_shapes]
                    for i in range(len(input_shapes), num_inputs):
                        dummy_inputs.append(torch.randn(*base_shape, device=self.device))
            else:
                # Use provided input shapes
                dummy_inputs = [torch.randn(*shape, device=self.device) for shape in input_shapes]

            # Run inference with error handling
            with torch.no_grad():
                try:
                    if len(dummy_inputs) == 1:
                        output = model(dummy_inputs[0])
                    elif len(dummy_inputs) == 2:
                        output = model(dummy_inputs[0], dummy_inputs[1])
                    elif len(dummy_inputs) == 3:
                        output = model(dummy_inputs[0], dummy_inputs[1], dummy_inputs[2])
                    elif len(dummy_inputs) == 4:
                        output = model(dummy_inputs[0], dummy_inputs[1], dummy_inputs[2], dummy_inputs[3])
                    else:
                        output = model(*dummy_inputs)
                except Exception as e:
                    # If the first attempt fails, try with just the provided input shapes
                    logger.warning(f"Model validation with {len(dummy_inputs)} inputs failed: {e}. Trying with original input shapes.")
                    dummy_inputs = [torch.randn(*shape, device=self.device) for shape in input_shapes]
                    if len(dummy_inputs) == 1:
                        output = model(dummy_inputs[0])
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
                raise ValueError(
                    f"Model output shape {output_shape} does not match "
                    f"number of classes {n_classes} in class mapping."
                )

            # Collect model information
            model_info = {
                "input_shapes": input_shapes,
                "actual_input_shapes": [list(inp.shape) for inp in dummy_inputs],
                "output_shape": output_shape,
                "n_classes": n_classes,
                "device": str(self.device),
                "model_type": "torchscript" if isinstance(model, torch.jit.ScriptModule) else "pytorch"
            }

            logger.info(f"Model validation successful. Output shape: {output_shape}")
            return output_shape, model_info

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise ValueError(f"Model validation failed: {str(e)}")

    def extract_model_metadata(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Extract metadata from a loaded model.

        Args:
            model: Loaded PyTorch model

        Returns:
            Dictionary containing model metadata
        """
        metadata = {
            "model_type": "torchscript" if isinstance(model, torch.jit.ScriptModule) else "pytorch",
            "device": str(self.device),
            "training": model.training,
        }

        # Try to get model parameters count
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            metadata.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            })
        except Exception as e:
            logger.warning(f"Could not extract parameter count: {e}")

        return metadata

    def cleanup_model(self, model: torch.nn.Module) -> None:
        """
        Clean up model resources.

        Args:
            model: Model to clean up
        """
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")

class ModelValidator:
    """
    Validator for model files and metadata.
    """

    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str] = [".pth", ".pt"]) -> None:
        """
        Validate file extension.

        Args:
            filename: Name of the file to validate
            allowed_extensions: List of allowed extensions

        Raises:
            ValueError: If extension is not allowed
        """
        if not filename:
            raise ValueError("Filename cannot be empty")

        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise ValueError(f"File extension {file_ext} not allowed. Allowed: {allowed_extensions}")

    @staticmethod
    def validate_class_mapping(class_mapping: Dict[str, int]) -> None:
        """
        Validate class mapping structure.

        Args:
            class_mapping: Dictionary mapping class names to indices

        Raises:
            ValueError: If class mapping is invalid
        """
        if not isinstance(class_mapping, dict) or not class_mapping:
            raise ValueError("Class mapping must be a non-empty dictionary")

        # Check for valid indices
        indices = list(class_mapping.values())
        if not all(isinstance(idx, int) and idx >= 0 for idx in indices):
            raise ValueError("All class mapping values must be non-negative integers")

        # Check for unique indices
        if len(indices) != len(set(indices)):
            raise ValueError("Class mapping indices must be unique")

        # Check for consecutive indices starting from 0
        expected_indices = set(range(len(indices)))
        if set(indices) != expected_indices:
            raise ValueError("Class mapping indices must be consecutive starting from 0")

    @staticmethod
    def validate_input_shape(input_shape: List[int]) -> None:
        """
        Validate input shape.

        Args:
            input_shape: List of integers representing input shape

        Raises:
            ValueError: If input shape is invalid
        """
        if not isinstance(input_shape, list) or not input_shape:
            raise ValueError("Input shape must be a non-empty list")

        if not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
            raise ValueError("All input shape dimensions must be positive integers")
