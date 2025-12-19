import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import splrep, splev
from typing import Tuple, Optional, Union
from astrodash.config.logging import get_logger
from astrodash.shared.utils.validators import validate_spectrum, ValidationError

logger = get_logger(__name__)

class DashSpectrumProcessor:
    """
    Handles all preprocessing for the Dash (CNN) classifier.
    Includes normalization, wavelength binning, continuum removal, mean zeroing, and apodization.
    """

    # Configuration constants
    DEFAULT_EDGE_WIDTH = 50
    DEFAULT_EDGE_RATIO = 4
    DEFAULT_OUTER_VAL = 0.5
    MIN_FILTER_SIZE = 3

    def __init__(self, w0: float, w1: float, nw: int, num_spline_points: int = 13):
        """
        Initialize the DashSpectrumProcessor.

        Args:
            w0: Minimum wavelength in Angstroms
            w1: Maximum wavelength in Angstroms
            nw: Number of wavelength bins
            num_spline_points: Number of points for spline fitting

        Raises:
            ValueError: If parameters are invalid
        """
        if w0 <= 0 or w1 <= 0 or w0 >= w1:
            raise ValueError(f"Invalid wavelength range: w0={w0}, w1={w1}")
        if nw <= 0:
            raise ValueError(f"Invalid number of bins: nw={nw}")
        if num_spline_points < 3:
            raise ValueError(f"Invalid spline points: {num_spline_points} (minimum 3)")

        self.w0 = float(w0)
        self.w1 = float(w1)
        self.nw = int(nw)
        self.num_spline_points = int(num_spline_points)

        logger.info(f"DashSpectrumProcessor initialized: w0={w0}, w1={w1}, nw={nw}")

    def process(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        z: float,
        smooth: int = 0,
        min_wave: Optional[float] = None,
        max_wave: Optional[float] = None
    ) -> Tuple[np.ndarray, int, int, float]:
        """
        Full preprocessing pipeline for Dash classifier.

        Args:
            wave: Wavelength array in Angstroms
            flux: Flux array (arbitrary units)
            z: Redshift value
            smooth: Smoothing factor (0 = no smoothing)
            min_wave: Minimum wavelength cutoff
            max_wave: Maximum wavelength cutoff

        Returns:
            Tuple of (processed_flux, min_idx, max_idx, z)

        Raises:
            ValidationError: If processing fails or spectrum is out of range
        """
        try:
            validate_spectrum(wave.tolist(), flux.tolist(), z)

            # Process spectrum
            flux_processed = self.normalise_spectrum(flux)
            flux_processed = self.limit_wavelength_range(wave, flux_processed, min_wave, max_wave)

            # Apply smoothing if requested
            if smooth > 0:
                flux_processed = self._apply_smoothing(wave, flux_processed, smooth)

            # Derive redshift and validate range
            wave_deredshifted = wave / (1 + z)
            if len(wave_deredshifted) < 2:
                raise ValidationError("Spectrum is out of classification range after deredshifting")

            # Apply processing pipeline
            binned_wave, binned_flux, min_idx, max_idx = self.log_wavelength_binning(wave_deredshifted, flux_processed)
            new_flux, _ = self.continuum_removal(binned_wave, binned_flux, min_idx, max_idx)
            mean_zero_flux = self.mean_zero(new_flux, min_idx, max_idx)
            apodized_flux = self.apodize(mean_zero_flux, min_idx, max_idx)
            flux_norm = self.normalise_spectrum(apodized_flux)
            flux_norm = self.zero_non_overlap_part(flux_norm, min_idx, max_idx, self.DEFAULT_OUTER_VAL)

            logger.debug(f"Processing completed: min_idx={min_idx}, max_idx={max_idx}")
            return flux_norm, min_idx, max_idx, z

        except ValidationError:
            # Re-raise ValidationError as-is
            raise
        except Exception as e:
            logger.error(f"Spectrum processing failed: {str(e)}")
            raise ValidationError(f"Spectrum processing failed: {str(e)}") from e

    def _apply_smoothing(self, wave: np.ndarray, flux: np.ndarray, smooth: int) -> np.ndarray:
        """Apply median filtering for smoothing."""
        try:
            wavelength_density = (np.max(wave) - np.min(wave)) / len(wave)
            w_density = (self.w1 - self.w0) / self.nw
            filter_size = int(w_density / wavelength_density * smooth / 2) * 2 + 1

            if filter_size >= self.MIN_FILTER_SIZE:
                flux_smoothed = medfilt(flux, kernel_size=filter_size)
                logger.debug(f"Applied smoothing with filter size {filter_size}")
                return flux_smoothed
            else:
                logger.warning(f"Filter size {filter_size} too small, skipping smoothing")
                return flux
        except Exception as e:
            logger.warning(f"Smoothing failed: {str(e)}, returning original flux")
            return flux

    @staticmethod
    def normalise_spectrum(flux: np.ndarray) -> np.ndarray:
        """
        Normalize flux array to [0, 1] range.

        Args:
            flux: Input flux array

        Returns:
            Normalized flux array

        Raises:
            ValidationError: If normalization fails
        """
        if len(flux) == 0:
            raise ValidationError("Cannot normalize empty array")

        flux_min, flux_max = np.min(flux), np.max(flux)

        if not np.isfinite(flux_min) or not np.isfinite(flux_max):
            raise ValidationError("Array contains non-finite values")

        if np.isclose(flux_min, flux_max):
            logger.warning("Normalizing spectrum: constant flux array")
            return np.zeros(len(flux))

        # Avoid division by zero
        if flux_max <= flux_min:
            raise ValidationError(f"Invalid flux range: min={flux_min}, max={flux_max}")

        return (flux - flux_min) / (flux_max - flux_min)

    @staticmethod
    def limit_wavelength_range(
        wave: np.ndarray,
        flux: np.ndarray,
        min_wave: Optional[float],
        max_wave: Optional[float]
    ) -> np.ndarray:
        """
        Limit flux values outside specified wavelength range.

        Args:
            wave: Wavelength array
            flux: Flux array
            min_wave: Minimum wavelength cutoff
            max_wave: Maximum wavelength cutoff

        Returns:
            Modified flux array
        """
        flux_out = np.copy(flux)

        if min_wave is not None and np.isfinite(min_wave):
            min_idx = np.clip((np.abs(wave - min_wave)).argmin(), 0, len(flux_out) - 1)
            flux_out[:min_idx] = 0

        if max_wave is not None and np.isfinite(max_wave):
            max_idx = np.clip((np.abs(wave - max_wave)).argmin(), 0, len(flux_out) - 1)
            flux_out[max_idx:] = 0

        return flux_out

    def log_wavelength_binning(self, wave: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Bin flux to log-wavelength grid.

        Args:
            wave: Input wavelength array
            flux: Input flux array

        Returns:
            Tuple of (binned_wavelength, binned_flux, min_index, max_index)
        """
        try:
            dwlog = np.log(self.w1 / self.w0) / self.nw
            wlog = self.w0 * np.exp(np.arange(0, self.nw) * dwlog)
            binned_flux = np.interp(wlog, wave, flux, left=0, right=0)

            # Find non-zero region
            non_zero_indices = np.where(binned_flux != 0)[0]

            if len(non_zero_indices) == 0:
                min_index = max_index = 0
            else:
                min_index = non_zero_indices[0]
                max_index = non_zero_indices[-1]

            return wlog, binned_flux, min_index, max_index

        except Exception as e:
            logger.error(f"Wavelength binning failed: {str(e)}")
            raise ValidationError(f"Wavelength binning failed: {str(e)}") from e

    def continuum_removal(self, wave: np.ndarray, flux: np.ndarray, min_idx: int, max_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove continuum from spectrum using spline fitting.

        Args:
            wave: Wavelength array
            flux: Flux array
            min_idx: Start index of valid region
            max_idx: End index of valid region

        Returns:
            Tuple of (continuum_subtracted_flux, continuum)
        """
        try:
            # Validate indices
            min_idx = np.clip(min_idx, 0, len(flux) - 1)
            max_idx = np.clip(max_idx, min_idx, len(flux) - 1)

            wave_region = wave[min_idx:max_idx + 1]
            flux_region = flux[min_idx:max_idx + 1]

            if len(wave_region) > self.num_spline_points:
                # Use spline fitting
                indices = np.linspace(0, len(wave_region) - 1, self.num_spline_points, dtype=int)
                spline = splrep(wave_region[indices], flux_region[indices], k=3)
                continuum = splev(wave_region, spline)
            else:
                # Use mean for short regions
                continuum = np.full_like(flux_region, np.mean(flux_region))

            # Create full continuum array
            full_continuum = np.zeros_like(flux)
            full_continuum[min_idx:max_idx + 1] = continuum

            return flux - full_continuum, full_continuum

        except Exception as e:
            logger.error(f"Continuum removal failed: {str(e)}")
            raise ValidationError(f"Continuum removal failed: {str(e)}") from e

    @staticmethod
    def mean_zero(flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
        """
        Zero-mean the flux array within the specified region.

        Args:
            flux: Input flux array
            min_idx: Start index of valid region
            max_idx: End index of valid region

        Returns:
            Zero-meaned flux array
        """
        flux_out = np.copy(flux)

        # Validate indices
        min_idx = np.clip(min_idx, 0, len(flux_out) - 1)
        max_idx = np.clip(max_idx, min_idx, len(flux_out) - 1)

        # Set regions outside valid range to edge values
        flux_out[:min_idx] = flux_out[min_idx]
        flux_out[max_idx:] = flux_out[max_idx]

        # Subtract mean from valid region
        valid_mean = np.mean(flux_out[min_idx:max_idx + 1])
        flux_out = flux_out - valid_mean

        return flux_out

    @staticmethod
    def apodize(flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
        """
        Apply apodization to reduce edge effects.

        Args:
            flux: Input flux array
            min_idx: Start index of valid region
            max_idx: End index of valid region

        Returns:
            Apodized flux array
        """
        apodized = np.copy(flux)

        # Validate indices
        min_idx = np.clip(min_idx, 0, len(apodized) - 1)
        max_idx = np.clip(max_idx, min_idx, len(apodized) - 1)

        # Calculate edge width
        edge_width = min(DashSpectrumProcessor.DEFAULT_EDGE_WIDTH, (max_idx - min_idx) // DashSpectrumProcessor.DEFAULT_EDGE_RATIO)

        if edge_width > 0:
            for i in range(edge_width):
                factor = 0.5 * (1 + np.cos(np.pi * i / edge_width))

                # Apply to left edge
                left_idx = min_idx + i
                if 0 <= left_idx < len(apodized):
                    apodized[left_idx] *= factor

                # Apply to right edge
                right_idx = max_idx - i
                if 0 <= right_idx < len(apodized):
                    apodized[right_idx] *= factor

        return apodized

    @staticmethod
    def zero_non_overlap_part(
        array: np.ndarray,
        min_index: int,
        max_index: int,
        outer_val: float = 0.0
    ) -> np.ndarray:
        """
        Set regions outside the valid range to a specified value.

        Args:
            array: Input array
            min_index: Start index of valid region
            max_index: End index of valid region
            outer_val: Value to set outside valid region

        Returns:
            Modified array
        """
        sliced_array = np.copy(array)

        # Validate indices
        min_index = np.clip(min_index, 0, len(sliced_array) - 1)
        max_index = np.clip(max_index, min_index, len(sliced_array) - 1)

        # Set outer regions
        sliced_array[:min_index] = outer_val
        sliced_array[max_index:] = outer_val

        return sliced_array


class TransformerSpectrumProcessor:
    """
    Handles preprocessing for the Transformer classifier.
    Includes interpolation to target length and normalization.
    """

    def __init__(self, target_length: int = 1024):
        """
        Initialize the TransformerSpectrumProcessor.

        Args:
            target_length: Target length for interpolation

        Raises:
            ValueError: If target_length is invalid
        """
        if target_length <= 0:
            raise ValueError(f"Invalid target length: {target_length}")

        self.target_length = int(target_length)
        logger.info(f"TransformerSpectrumProcessor initialized with target length: {target_length}")

    def process(self, x: Union[np.ndarray, list], y: Union[np.ndarray, list], redshift: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Interpolate and normalize spectrum data for transformer input.

        Args:
            x: Wavelength array
            y: Flux array
            redshift: Redshift value

        Returns:
            Tuple of (interpolated_x, normalized_y, redshift)

        Raises:
            ValidationError: If processing fails
        """
        try:
            validate_spectrum(x if isinstance(x, list) else x.tolist(),
                           y if isinstance(y, list) else y.tolist(),
                           redshift)

            # Convert to numpy arrays
            x_array = np.asarray(x, dtype=np.float64)
            y_array = np.asarray(y, dtype=np.float64)

            # Interpolate to target length
            x_interp = self._interpolate_to_length(x_array, self.target_length)
            y_interp = self._interpolate_to_length(y_array, self.target_length)

            # Normalize flux
            y_norm = self._normalize(y_interp)

            logger.debug(f"Transformer processing completed: input_length={len(x)}, output_length={self.target_length}")
            return x_interp, y_norm, redshift

        except ValidationError:
            # Re-raise ValidationError as-is
            raise
        except Exception as e:
            logger.error(f"Transformer processing failed: {str(e)}")
            raise ValidationError(f"Transformer processing failed: {str(e)}") from e

    def _interpolate_to_length(self, arr: np.ndarray, length: int) -> np.ndarray:
        """
        Interpolate array to target length.

        Args:
            arr: Input array
            length: Target length

        Returns:
            Interpolated array
        """
        if len(arr) == length:
            return arr

        # Create normalized coordinate systems
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, length)

        # Interpolate
        return np.interp(x_new, x_old, arr)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalize array to [0, 1] range.

        Args:
            arr: Input array

        Returns:
            Normalized array

        Raises:
            ValidationError: If normalization fails
        """
        if len(arr) == 0:
            raise ValidationError("Cannot normalize empty array")

        arr_min, arr_max = np.min(arr), np.max(arr)

        if not np.isfinite(arr_min) or not np.isfinite(arr_max):
            raise ValidationError("Array contains non-finite values")

        if np.isclose(arr_min, arr_max):
            logger.warning("Normalizing transformer input: constant array")
            return np.zeros(len(arr))

        # Avoid division by zero
        if arr_max <= arr_min:
            raise ValidationError(f"Invalid array range: min={arr_min}, max={arr_max}")

        return (arr - arr_min) / (arr_max - arr_min)
