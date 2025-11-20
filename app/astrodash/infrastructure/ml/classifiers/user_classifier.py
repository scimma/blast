import torch
import numpy as np
from typing import Any, Optional
from astrodash.infrastructure.ml.classifiers.base import BaseClassifier
from astrodash.infrastructure.storage.model_storage import ModelStorage
from astrodash.config.settings import get_settings, Settings
from astrodash.config.logging import get_logger

logger = get_logger(__name__)

class UserClassifier(BaseClassifier):
    def __init__(self, user_model_id: str, model_storage: ModelStorage, config: Settings = None):
        super().__init__(config)
        self.user_model_id = user_model_id
        self.config = config or get_settings()
        self.model_storage = model_storage
        self.model = None
        self.class_map = None
        self.input_shape = None
        self._load_model_and_metadata()

    def _load_model_and_metadata(self):
        """Load user model and metadata using ModelStorage."""
        try:
            # Use ModelStorage to get file paths and load data
            model_path = self.model_storage.get_model_path(self.user_model_id)
            self.class_map = self.model_storage.load_class_mapping(self.user_model_id)
            self.input_shape = self.model_storage.load_input_shape(self.user_model_id)

            # Load the model
            self.model = torch.jit.load(model_path, map_location='cpu')
            self.model.eval()

            logger.info(f"Loaded user model {self.user_model_id} with input shape {self.input_shape}")
        except FileNotFoundError as e:
            logger.error(f"User model files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load user model {self.user_model_id}: {e}")
            raise

    def _infer_sequence_length(self) -> int:
        """Infer the target sequence length from input_shape metadata.
        Handles cases where input_shape is a list of shapes (e.g., [[1,1024],[1,1024],[1,1]]).
        """
        shape = self.input_shape
        # list of shapes for multiple inputs
        if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
            candidate_lengths = []
            for sub in shape:
                if isinstance(sub, (list, tuple)) and len(sub) >= 2:
                    for dim in sub[1:]:
                        if isinstance(dim, int) and dim > 1:
                            candidate_lengths.append(dim)
                            break
            if candidate_lengths:
                return int(max(candidate_lengths))
        # single shape like [batch, seq_len]
        if isinstance(shape, (list, tuple)):
            for dim in shape:
                if isinstance(dim, int) and dim > 1:
                    return int(dim)
        return 1024

    async def classify(self, spectrum: Any) -> dict:
        try:
            flux = np.array(spectrum.y)
            wavelength = np.array(spectrum.x)
            redshift = getattr(spectrum, 'redshift', 0.0) or 0.0

            # Handle different input shapes based on model type. So far only CNN and Transformer.
            if len(self.input_shape) == 4:  # [batch, channels, height, width] - CNN style
                flat_size = np.prod(self.input_shape[1:])
                flux_flat = np.zeros(flat_size)
                n = min(len(flux), flat_size)
                flux_flat[:n] = flux[:n]
                model_input = torch.tensor(flux_flat, dtype=torch.float32).reshape(self.input_shape)
                with torch.no_grad():
                    output = self.model(model_input)
                probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
            else:  # Transformer style - needs wavelength, flux, redshift
                target_length = self._infer_sequence_length()
                logger.info(f"Transformer model input shape: {self.input_shape}, inferred target length: {int(target_length)}")

                if len(flux) != target_length:
                    x_old = np.linspace(0, 1, len(flux))
                    x_new = np.linspace(0, 1, int(target_length))
                    flux = np.interp(x_new, x_old, flux)
                    wavelength = np.interp(x_new, x_old, wavelength)
                wavelength_tensor = torch.tensor(wavelength, dtype=torch.float32).unsqueeze(0)  # [1, target_length]
                flux_tensor = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)              # [1, target_length]
                redshift_tensor = torch.tensor([redshift], dtype=torch.float32)               # [1]
                with torch.no_grad():
                    output = self.model(wavelength_tensor, flux_tensor, redshift_tensor)
                probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
            idx_to_label = {v: k for k, v in self.class_map.items()}
            top_indices = np.argsort(probs)[::-1][:3]
            matches = []
            for idx in top_indices:
                class_name = idx_to_label.get(idx, f'unknown_class_{idx}')
                matches.append({
                    'type': class_name,
                    'age': 'N/A',  # User models don't classify age
                    'probability': float(probs[idx]),
                    'redshift': redshift,
                    'rlap': None,  # Not calculated for user-uploaded models
                    'reliable': bool(probs[idx] > self.config.user_model_reliability_threshold)
                })
            best_match = matches[0] if matches else {'type': 'Unknown', 'age': 'N/A', 'probability': 0.0}
            return {
                "best_matches": matches,
                "best_match": best_match,
                "reliable_matches": best_match.get('reliable', False) if best_match else False,
                "user_model_id": self.user_model_id
            }
        except Exception as e:
            logger.error(f"Error using user-uploaded model: {e}", exc_info=True)
            raise
