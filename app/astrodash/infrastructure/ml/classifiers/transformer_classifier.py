import os
import torch
import numpy as np
from typing import Any, Optional, Dict
from astrodash.infrastructure.ml.classifiers.base import BaseClassifier
from astrodash.infrastructure.ml.data_processor import TransformerSpectrumProcessor
from astrodash.infrastructure.ml.classifiers.architectures import spectraTransformerEncoder
from astrodash.shared.utils.helpers import interpolate_to_1024
from astrodash.config.settings import get_settings, Settings
from astrodash.config.logging import get_logger

logger = get_logger(__name__)

class TransformerClassifier(BaseClassifier):
    """
    Yuqing's Transformer classifier for supernova spectra.
    """
    def __init__(self, config: Settings = None, processor: Optional[TransformerSpectrumProcessor] = None):
        super().__init__(config)
        self.config = config or get_settings()
        self.processor = processor or TransformerSpectrumProcessor(target_length=self.config.nw)
        self.model_path = self.config.transformer_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        # Label mapping for classes
        self.label_mapping = self.config.label_mapping
        if not self.label_mapping:
            logger.error("Label mapping not configured")
            raise ValueError("Label mapping must be configured")

        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        logger.debug(f"Using label mapping: {self.label_mapping} with {len(self.label_mapping)} classes")

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the transformer model with proper initialization"""
        if not self.model_path:
            logger.error("Transformer model path not configured")
            self.model = None
            return

        if not os.path.exists(self.model_path):
            logger.error(f"Transformer model file not found at {self.model_path}")
            self.model = None
            return

        try:
            # Use configurable hyperparameters from settings
            model = spectraTransformerEncoder(
                bottleneck_length=self.config.transformer_bottleneck_length,
                model_dim=self.config.transformer_model_dim,
                num_heads=self.config.transformer_num_heads,
                num_layers=self.config.transformer_num_layers,
                num_classes=len(self.label_mapping),  # Dynamic based on label mapping
                ff_dim=self.config.transformer_ff_dim,
                dropout=self.config.transformer_dropout,
                selfattn=self.config.transformer_selfattn
            ).to(self.device)

            # Load the state dict
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()

            self.model = model
            logger.debug(f"Transformer model loaded successfully from {self.model_path} with {sum(p.numel() for p in model.parameters())} parameters")

        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            self.model = None

    def classify_sync(self, spectrum: Any) -> dict:
        """Synchronous CPU-bound classification used from async via to_thread."""
        if self.model is None:
            logger.error("Transformer model is not loaded. Returning empty result.")
            return {}

        try:
            # Extract data from spectrum
            x = np.array(spectrum.x)
            y = np.array(spectrum.y)
            redshift = getattr(spectrum, 'redshift', 0.0) or 0.0

            # Extract and preprocess data
            wavelength_data = interpolate_to_1024(x)  # wavelength
            flux_data = interpolate_to_1024(y)        # flux

            # Convert to tensors with proper shape
            wavelength = torch.tensor(wavelength_data, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, 1024]
            flux = torch.tensor(flux_data, dtype=torch.float32).unsqueeze(0).to(self.device)              # [1, 1024]
            redshift_tensor = torch.tensor([[redshift]], dtype=torch.float32).to(self.device)             # [1, 1]

            logger.debug(f"Input shapes - wavelength: {wavelength.shape}, flux: {flux.shape}, redshift: {redshift_tensor.shape}")
            logger.debug(f"Redshift value: {redshift}")

            # Run inference
            with torch.no_grad():
                logits = self.model(wavelength, flux, redshift_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            logger.debug(f"Model output probabilities: {probs}")

            # Get top 3 predictions using the correct label mapping
            top_indices = np.argsort(probs)[::-1][:3]
            matches = []

            for idx in top_indices:
                class_name = self.idx_to_label.get(idx, f'unknown_class_{idx}')
                matches.append({
                    'type': class_name,
                    'probability': float(probs[idx]),
                    'redshift': redshift,
                    'rlap': None,  # Not calculated for transformer model (RLAP requires template matching)
                    'reliable': probs[idx] > 0.5  # Simple reliability threshold
                })

            best_match = matches[0] if matches else {}

            logger.debug(f"Classification results - best match: {best_match}")

            return {
                'best_matches': matches,
                'best_match': best_match,
                'reliable_matches': best_match.get('reliable', False) if best_match else False
            }

        except Exception as e:
            logger.error(f"Error during transformer classification: {e}")
            return {}

    async def classify(self, spectrum: Any) -> dict:
        """Async wrapper that runs the synchronous CPU-bound classify in a thread."""
        import asyncio
        return await asyncio.to_thread(self.classify_sync, spectrum)

    def load_model_from_state_dict(self, state_dict, model_config: Dict[str, Any]):
        """
        Load a model from a state dict with the specified configuration.

        Args:
            state_dict: PyTorch state dict containing model weights
            model_config: Dictionary containing model hyperparameters
        """
        try:
            # Initialize the model with the provided config
            model = spectraTransformerEncoder(**model_config).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()

            self.model = model
            logger.info(f"Transformer model loaded from state dict with {sum(p.numel() for p in model.parameters())} parameters")

        except Exception as e:
            logger.error(f"Failed to load transformer model from state dict: {e}")
            self.model = None

    def update_model_from_state_dict(self, state_dict, model_config: Dict[str, Any]):
        """
        Update the current model with a new state dict.

        Args:
            state_dict: PyTorch state dict containing model weights
            model_config: Dictionary containing model hyperparameters
        """
        self.load_model_from_state_dict(state_dict, model_config)
