import os
import torch
import numpy as np
import pickle
from typing import Any, Optional
from astrodash.infrastructure.ml.classifiers.base import BaseClassifier
from astrodash.infrastructure.ml.data_processor import DashSpectrumProcessor
from astrodash.infrastructure.ml.classifiers.architectures import AstroDashPyTorchNet
from astrodash.config.settings import get_settings, Settings
from astrodash.infrastructure.ml.dash_utils import combined_prob
from astrodash.config.logging import get_logger

logger = get_logger(__name__)

class DashClassifier(BaseClassifier):
    """
    Production-grade Dash (CNN) classifier for supernova spectra.
    Uses dependency injection for processor and config.
    """
    def __init__(self, config: Settings = None, processor: Optional[DashSpectrumProcessor] = None):
        super().__init__(config)
        self.config = config or get_settings()
        self.processor = processor
        self.model = None
        self.model_path = self.config.dash_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self.nw = self.config.nw
        self.w0 = self.config.w0
        self.w1 = self.config.w1
        if not self.processor:
            self.processor = DashSpectrumProcessor(self.w0, self.w1, self.nw)
        self.type_names_list = self._load_type_names()

    def _load_type_names(self):
        """Load type names from training parameters file using settings."""
        try:
            training_params_path = self.config.dash_training_params_path

            if not os.path.exists(training_params_path):
                logger.error(f"Training parameters file not found at {training_params_path}")
                return []

            with open(training_params_path, 'rb') as f:
                pars = pickle.load(f, encoding='latin1')

            # Extract type list and generate type names like original
            type_list = pars.get('typeList', [])
            min_age = pars.get('minAge', -5)
            max_age = pars.get('maxAge', 15)
            age_bin_size = pars.get('ageBinSize', 4)

            age_labels = []
            age_bin_prev = 0
            age_label_min = min_age
            for age in np.arange(min_age, max_age, 0.5):
                age_bin = int(round(age / age_bin_size)) - int(round(min_age / age_bin_size))
                if age_bin != age_bin_prev:
                    age_label_max = int(round(age))
                    age_labels.append(f"{int(age_label_min)} to {age_label_max}")
                    age_label_min = age_label_max
                age_bin_prev = age_bin
            age_labels.append(f"{int(age_label_min)} to {int(max_age)}")

            # Generate type names like old dash
            type_names = []
            for t_type in type_list:
                for age_label in age_labels:
                    type_names.append(f"{t_type}: {age_label}")

            logger.debug(f"Loaded {len(type_names)} type names from training parameters")
            return type_names

        except Exception as e:
            logger.error(f"Failed to load type names from training parameters: {e}")
            return []

    def _classification_split(self, classification_string):
        """Split classification string like 'Ia: 2 to 6' into type and age."""
        parts = classification_string.split(': ')
        return "", parts[0], parts[1]

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"Dash model not found at {self.model_path}. Classifier will not work.")
            self.model = None
            return
        state_dict = torch.load(self.model_path, map_location=self.device)
        # Check for the new layer name first, fall back to old name for compatibility
        if 'output.weight' in state_dict:
            n_types = state_dict['output.weight'].shape[0]
        else:
            n_types = state_dict['classifier.3.weight'].shape[0]
        self.model = self.load_model_from_state_dict(state_dict, n_types)
        logger.debug(f"Dash model loaded from {self.model_path}")

    def classify_sync(self, spectrum: Any) -> dict:
        """Synchronous CPU-bound classification used from async via to_thread."""
        if self.model is None:
            logger.error("Dash model is not loaded. Returning empty result.")
            return {}
        # Assume spectrum.x, spectrum.y, spectrum.redshift
        x = np.array(spectrum.x)
        y = np.array(spectrum.y)
        z = getattr(spectrum, 'redshift', 0.0) or 0.0
        # Preprocess
        processed_flux, min_idx, max_idx, z = self.processor.process(x, y, z)
        input_tensor = torch.from_numpy(processed_flux).float().reshape(1, -1)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        softmax = outputs[0].cpu().numpy()

        # Only use the first n_bins outputs (corresponding to actual galaxy types)
        # The model may have been trained with more classes but only the first n_bins are valid
        n_bins = len(self.type_names_list)
        softmax = softmax[:n_bins]

        logger.debug(f"Softmax shape: {softmax.shape}, type_names_list length: {len(self.type_names_list)}")
        logger.debug(f"Using first {n_bins} outputs from model")

        # Process ALL classifications for future combination
        all_indices = np.argsort(softmax)[::-1]  # Sort all indices
        matches = []

        logger.debug(f"Processing all {len(all_indices)} classifications")

        for idx in all_indices:
            if idx < len(self.type_names_list):
                classification = self.type_names_list[idx]
                _, sn_type, sn_age = self._classification_split(classification)
                matches.append({
                    'type': sn_type,
                    'age': sn_age,
                    'probability': float(softmax[idx]),
                    'redshift': z,
                    'rlap': None,
                    'reliable': False
                })
                logger.debug(f"Added match {len(matches)}: {sn_type} ({sn_age}) - {float(softmax[idx]):.3f}")
            else:
                logger.warning(f"Index {idx} out of range for type_names_list (length: {len(self.type_names_list)})")

        if not matches:
            logger.warning("No valid matches found. Returning mock classification.")
            return {
                'best_matches': [],
                'best_match': {
                    'type': 'Unknown',
                    'age': 'Unknown',
                    'probability': 0.0,
                    'redshift': z,
                    'rlap': None
                },
                'reliable_matches': False
            }

        # Use combined_prob like the original DASH package with ALL matches
        best_match_list_for_prob = [[m['type'], m['age'], m['probability']] for m in matches]
        best_type, best_age, prob_total, reliable_flag = combined_prob(best_match_list_for_prob)

        # Update the best match with combined probability
        best_match = {
            'type': best_type,
            'age': best_age,
            'probability': prob_total,
            'redshift': z,
            'rlap': None,
            'reliable': reliable_flag
        }

        # Update matches to mark the best one as reliable
        for m in matches:
            if m['type'] == best_type and m['age'] == best_age:
                m['reliable'] = reliable_flag

        # Check if RLAP calculation is requested
        calculate_rlap = getattr(spectrum, 'calculate_rlap', False)
        if not calculate_rlap and hasattr(spectrum, 'meta') and spectrum.meta:
            calculate_rlap = spectrum.meta.get('processing_params', {}).get('calculate_rlap', False)
        logger.debug(f"RLAP calculation requested: {calculate_rlap}")

        if calculate_rlap:
            try:
                # Import RLAP functions
                from astrodash.infrastructure.ml.rlap_calculator import prepare_log_wavelength_and_templates, get_templates_for_type_age, normalize_age_bin, get_nonzero_minmax, calculate_rlap_with_redshift

                # Prepare templates and log wavelength grid
                log_wave, input_flux_log, snTemplates, dwlog, nw, w0, w1 = prepare_log_wavelength_and_templates(spectrum)

                # Calculate RLAP for all top 3 matches
                for i, match in enumerate(matches[:3]):
                    sn_type = match['type']
                    age = match['age']
                    age_norm = normalize_age_bin(age)


                    if sn_type in snTemplates:
                        logger.debug(f"Available age bins for {sn_type}: {list(snTemplates[sn_type].keys())}")
                    else:
                        logger.warning(f"SN type '{sn_type}' not found in templates")

                    # Get templates for this match
                    template_fluxes, template_names, template_minmax_indexes = get_templates_for_type_age(snTemplates, sn_type, age_norm, log_wave)

                    if template_fluxes:
                        try:
                            logger.debug(f"Computing RLAP for match {i+1} with found templates")
                            input_minmax_index = get_nonzero_minmax(input_flux_log)
                            rlap_label, used_redshift, rlap_warning = calculate_rlap_with_redshift(
                                log_wave, input_flux_log, template_fluxes, template_names, template_minmax_indexes, input_minmax_index,
                                redshift=match['redshift'] if hasattr(spectrum, 'known_z') and spectrum.known_z else None
                            )
                            match['rlap'] = rlap_label
                            match['rlap_warning'] = rlap_warning
                        except Exception as e:
                            logger.error(f"Error during RLAP calculation for match {i+1} ({sn_type}): {e}")
                            match['rlap'] = "N/A"
                            match['rlap_warning'] = True
                            logger.debug(f"Setting RLAP to 'N/A' for match {i+1} due to calculation error")
                    else:
                        logger.error(f"No valid templates found for RLap calculation for match {i+1}.")
                        match['rlap'] = "N/A"
                        match['rlap_warning'] = True
                        logger.debug(f"Setting RLAP to 'N/A' for match {i+1} due to no templates")

                # Update best_match with RLAP from the best match
                best_match['rlap'] = matches[0]['rlap']
                best_match['rlap_warning'] = matches[0]['rlap_warning']

                logger.debug(f"RLap score calculated: {best_match.get('rlap', 'N/A')} (warning: {best_match.get('rlap_warning', False)})")

            except Exception as e:
                logger.error(f"Error during RLAP calculation setup: {e}")
                # Set RLAP to "N/A" if calculation fails
                for m in matches:
                    m['rlap'] = "N/A"
                    m['rlap_warning'] = True
                best_match['rlap'] = "N/A"
                best_match['rlap_warning'] = True
        else:
            # Set default RLAP values when calculation is skipped
            for m in matches:
                m['rlap'] = None
                m['rlap_warning'] = False
            best_match['rlap'] = None
            best_match['rlap_warning'] = False

        # Return only top 3 matches for display, but use all for probability calculation
        result = {
            'best_matches': matches[:3],  # Only return top 3 for display
            'best_match': best_match,
            'reliable_matches': reliable_flag
        }

        return result

    async def classify(self, spectrum: Any) -> dict:
        """
        Async wrapper that runs the synchronous CPU-bound classify in a thread.
        """
        import asyncio
        return await asyncio.to_thread(self.classify_sync, spectrum)

    def load_model_from_state_dict(self, state_dict, n_classes):
        """
        Load a model from a state dict with the specified number of classes.

        Args:
            state_dict: PyTorch state dict containing model weights
            n_classes: Number of output classes for the model

        Returns:
            AstroDashPyTorchNet: Loaded and configured model
        """
        # Use im_width=32 as in the default model
        model = AstroDashPyTorchNet(n_types=n_classes, im_width=32)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def update_model_from_state_dict(self, state_dict, n_classes):
        """
        Update the current model with a new state dict.

        Args:
            state_dict: PyTorch state dict containing model weights
            n_classes: Number of output classes for the model
        """
        self.model = self.load_model_from_state_dict(state_dict, n_classes)
        logger.info(f"Dash model updated with new state dict (n_classes={n_classes})")
