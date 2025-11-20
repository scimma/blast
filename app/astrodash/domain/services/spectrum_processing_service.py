from typing import Dict, Any, Optional, Tuple
from astrodash.domain.models.spectrum import Spectrum
from astrodash.infrastructure.ml.data_processor import DashSpectrumProcessor, TransformerSpectrumProcessor
from astrodash.shared.utils.helpers import interpolate_to_1024, normalise_spectrum
from astrodash.config.settings import Settings, get_settings
from astrodash.config.logging import get_logger
from astrodash.core.exceptions import SpectrumProcessingException
import numpy as np

logger = get_logger(__name__)

class SpectrumProcessingService:
    """
    Domain service for processing spectra with custom parameters.
    Handles smoothing, redshift application, wavelength filtering, and preprocessing.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.dash_processor = DashSpectrumProcessor(
            w0=self.settings.w0,
            w1=self.settings.w1,
            nw=self.settings.nw
        )
        self.transformer_processor = TransformerSpectrumProcessor(
            target_length=self.settings.nw
        )
        logger.debug("SpectrumProcessingService initialized with settings")

    async def process_spectrum_with_params(
        self,
        spectrum: Spectrum,
        params: Dict[str, Any]
    ) -> Spectrum:
        """
        Process spectrum with custom parameters.

        Args:
            spectrum: Input spectrum to process
            params: Processing parameters including:
                - smoothing: Smoothing factor (0 = no smoothing)
                - knownZ: Whether redshift is known
                - zValue: Known redshift value
                - minWave: Minimum wavelength filter
                - maxWave: Maximum wavelength filter
                - calculateRlap: Whether to calculate RLAP

        Returns:
            Processed spectrum

        Raises:
            SpectrumProcessingException: If spectrum processing fails
        """
        try:
            # Extract parameters
            smoothing = params.get('smoothing', 0)
            known_z = params.get('knownZ', False)
            z_value = params.get('zValue')
            min_wave = params.get('minWave')
            max_wave = params.get('maxWave')
            calculate_rlap = params.get('calculateRlap', False)

            # Convert spectrum data to numpy arrays
            x = np.array(spectrum.x)
            y = np.array(spectrum.y)

            logger.debug(f"Processing spectrum with {len(x)} points")
            logger.debug(f"Parameters: smoothing={smoothing}, z_value={z_value}, min_wave={min_wave}, max_wave={max_wave}")

            # Apply wavelength filtering
            if min_wave is not None or max_wave is not None:
                x, y = self._apply_wavelength_filter(x, y, min_wave, max_wave)
                logger.debug(f"Applied wavelength filter: {len(y)} points remaining")

            # Apply smoothing
            if smoothing > 0:
                y = self._apply_smoothing(x, y, smoothing)
                logger.debug(f"Applied smoothing with factor {smoothing}")

            # Normalize spectrum (matching old backend behavior)
            y = normalise_spectrum(y)
            logger.debug("Applied spectrum normalization")

            # Apply redshift if provided
            if z_value is not None:
                spectrum.redshift = float(z_value)
                logger.debug(f"Applied redshift: {z_value}")

            # Update spectrum with processed data
            spectrum.x = x.tolist()
            spectrum.y = y.tolist()

            # Add processing metadata
            if not hasattr(spectrum, 'meta'):
                spectrum.meta = {}
            spectrum.meta.update({
                'processing_params': {
                    'smoothing': smoothing,
                    'known_z': known_z,
                    'z_value': z_value,
                    'min_wave': min_wave,
                    'max_wave': max_wave,
                    'calculate_rlap': calculate_rlap
                }
            })

            logger.debug("Spectrum processing completed successfully")
            return spectrum

        except Exception as e:
            logger.error(f"Error processing spectrum with parameters: {e}")
            raise SpectrumProcessingException(f"Spectrum processing failed: {str(e)}")

    def _apply_wavelength_filter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        min_wave: Optional[float],
        max_wave: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply wavelength filtering to spectrum data.

        Args:
            x: Wavelength array
            y: Flux array
            min_wave: Minimum wavelength (None = no lower limit)
            max_wave: Maximum wavelength (None = no upper limit)

        Returns:
            Tuple of (filtered_wavelength, filtered_flux)
        """
        if min_wave is None and max_wave is None:
            return x, y

        mask = np.ones(len(x), dtype=bool)

        if min_wave is not None:
            mask &= (x >= min_wave)

        if max_wave is not None:
            mask &= (x <= max_wave)

        # Filter both wavelength and flux arrays
        x_filtered = x[mask]
        y_filtered = y[mask]

        logger.debug(f"Wavelength filtering: {len(x)} -> {len(x_filtered)} points")
        if len(x_filtered) > 0:
            logger.debug(f"Filtered wavelength range: {x_filtered.min():.2f} - {x_filtered.max():.2f}")

        return x_filtered, y_filtered

    def _apply_smoothing(
        self,
        x: np.ndarray,
        y: np.ndarray,
        smoothing_factor: int
    ) -> np.ndarray:
        """
        Apply smoothing to spectrum data.

        Args:
            x: Wavelength array
            y: Flux array
            smoothing_factor: Smoothing factor (higher = more smoothing)

        Returns:
            Smoothed flux array
        """
        if smoothing_factor <= 0:
            return y

        try:
            from scipy.signal import savgol_filter

            # Calculate window length based on smoothing factor
            window_length = min(smoothing_factor * 2 + 1, len(y))
            if window_length < 3:
                window_length = 3

            # Ensure window length is odd
            if window_length % 2 == 0:
                window_length += 1

            # Apply Savitzky-Golay filter
            y_smoothed = savgol_filter(y, window_length, 3)

            return y_smoothed

        except ImportError:
            logger.warning("scipy not available, skipping smoothing")
            return y
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}, returning original data")
            return y

    def prepare_for_model(
        self,
        spectrum: Spectrum,
        model_type: str
    ) -> Dict[str, np.ndarray]:
        """
        Prepare spectrum data for model input.

        Args:
            spectrum: Spectrum to prepare
            model_type: Type of model ('dash', 'transformer', 'user_uploaded')

        Returns:
            Dictionary with prepared data
        """
        try:
            x = np.array(spectrum.x)
            y = np.array(spectrum.y)
            z = getattr(spectrum, 'redshift', 0.0) or 0.0

            if model_type == 'dash':
                # Use Dash processor
                processed_y, min_idx, max_idx, processed_z = self.dash_processor.process(
                    wave=x,
                    flux=y,
                    z=z,
                    smooth=0,  # Smoothing already applied
                    min_wave=None,  # Filtering already applied
                    max_wave=None
                )
                return {
                    'x': x,
                    'y': processed_y,
                    'redshift': processed_z,
                    'min_idx': min_idx,
                    'max_idx': max_idx
                }

            elif model_type == 'transformer':
                # Use Transformer processor
                processed_x, processed_y, processed_z = self.transformer_processor.process(x, y, z)
                return {
                    'x': processed_x,
                    'y': processed_y,
                    'redshift': processed_z
                }

            else:
                # For user models, return as-is (they handle their own preprocessing)
                return {
                    'x': x,
                    'y': y,
                    'redshift': z
                }

        except Exception as e:
            logger.error(f"Error preparing spectrum for model {model_type}: {e}")
            raise ValueError(f"Model preparation failed: {str(e)}")
