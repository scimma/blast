import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from astrodash.shared.utils.redshift import get_median_redshift
from astrodash.shared.utils.helpers import prepare_log_wavelength_and_templates, get_nonzero_minmax, normalize_age_bin
from astrodash.config.settings import get_settings
from astrodash.config.logging import get_logger

logger = get_logger(__name__)

class RedshiftService:
    """
    Service for estimating redshift from a spectrum using DASH (CNN) templates.
    Note: Redshift estimation is only available for DASH models as they are the only
    models that have the required spectral templates.
    """
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.template_path = self.settings.template_path

    async def estimate_redshift_from_spectrum(
        self,
        x: List[float],
        y: List[float],
        sn_type: str,
        age_bin: str,
        model_type: str = "dash"
    ) -> Dict[str, Any]:
        """
        Estimate redshift from spectrum data using DASH templates.

        Args:
            x: Wavelength array
            y: Flux array
            sn_type: Supernova type (e.g., 'Ia', 'Ib', 'II')
            age_bin: Age bin (e.g., '2 to 6')
            model_type: Model type (must be 'dash' for redshift estimation)

        Returns:
            Dictionary with estimated redshift and error
        """
        try:
            # Validate that only DASH models are supported
            if model_type.lower() != "dash":
                return {
                    "estimated_redshift": None,
                    "estimated_redshift_error": None,
                    "message": f"Redshift estimation is only available for DASH (CNN) models. Received model type: {model_type}"
                }

            logger.info(f"Estimating redshift for {sn_type} {age_bin} using DASH templates")

            # Convert to numpy arrays
            x = np.array(x)
            y = np.array(y)

            # Prepare templates and log-wavelength grid
            log_wave, input_flux_log, snTemplates, dwlog, nw, w0, w1 = prepare_log_wavelength_and_templates(
                {"x": x, "y": y}, self.template_path
            )

            # Get templates for this type/age
            template_fluxes, template_names, template_minmax_indexes = self._get_templates_for_type_age(
                snTemplates, sn_type, age_bin, log_wave
            )

            if not template_fluxes:
                return {
                    "estimated_redshift": None,
                    "estimated_redshift_error": None,
                    "message": "No valid DASH templates found for this type/age."
                }

            # Get input spectrum minmax index
            input_minmax_index = get_nonzero_minmax(input_flux_log)

            # Estimate redshift
            est_z, _, _, est_z_err = get_median_redshift(
                input_flux_log, template_fluxes, nw, dwlog,
                input_minmax_index, template_minmax_indexes, template_names, outerVal=0.5
            )

            return {
                "estimated_redshift": float(est_z) if est_z is not None else None,
                "estimated_redshift_error": float(est_z_err) if est_z_err is not None else None,
                "message": "Redshift estimated successfully using DASH templates" if est_z is not None else "Redshift estimation failed"
            }

        except Exception as e:
            logger.error(f"Error estimating redshift: {e}", exc_info=True)
            return {
                "estimated_redshift": None,
                "estimated_redshift_error": None,
                "message": f"Redshift estimation failed: {str(e)}"
            }

    def _get_templates_for_type_age(
        self,
        snTemplates: Dict[str, Any],
        sn_type: str,
        age_bin: str,
        log_wave: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[Tuple[int, int]]]:
        """
        Get DASH templates for a specific SN type and age bin.

        Args:
            snTemplates: Template data from npz file
            sn_type: Supernova type
            age_bin: Age bin
            log_wave: Log-wavelength grid

        Returns:
            Tuple of (template_fluxes, template_names, template_minmax_indexes)
        """
        template_fluxes = []
        template_names = []
        template_minmax_indexes = []

        # Normalize age bin
        age_norm = normalize_age_bin(age_bin)

        if sn_type in snTemplates:
            age_bin_keys = [str(k).strip() for k in snTemplates[sn_type].keys()]
            if age_norm.strip() in age_bin_keys:
                real_key = [k for k in snTemplates[sn_type].keys() if str(k).strip() == age_norm.strip()][0]
                snInfo = snTemplates[sn_type][real_key].get('snInfo', None)

                if isinstance(snInfo, np.ndarray) and snInfo.shape[0] > 0:
                    for i in range(snInfo.shape[0]):
                        template_wave = snInfo[i][0]
                        template_flux = snInfo[i][1]

                        # Interpolate template to log-wavelength grid
                        interp_flux = np.interp(log_wave, template_wave, template_flux, left=0, right=0)

                        # Find nonzero region
                        nonzero = np.where(interp_flux != 0)[0]
                        if len(nonzero) > 0:
                            tmin, tmax = nonzero[0], nonzero[-1]
                        else:
                            tmin, tmax = 0, len(interp_flux) - 1

                        template_fluxes.append(interp_flux)
                        template_names.append(f"{sn_type}:{age_norm}")
                        template_minmax_indexes.append((tmin, tmax))

        logger.info(f"Found {len(template_fluxes)} DASH templates for {sn_type} {age_bin}")
        return template_fluxes, template_names, template_minmax_indexes

    async def estimate_redshift(
        self,
        input_flux: np.ndarray,
        temp_fluxes: List[np.ndarray],
        nw: int,
        dwlog: float,
        input_minmax_index: Any,
        temp_minmax_indexes: List[Any],
        temp_names: List[str],
        outer_val: float = 0.5
    ) -> Tuple[Optional[float], Optional[Dict[str, Any]], Optional[str], Optional[float]]:
        """
        Estimate the median redshift for the input spectrum using provided templates.
        Returns (median_redshift, crossCorrs, medianName, stdRedshift)
        """
        return get_median_redshift(
            input_flux, temp_fluxes, nw, dwlog, input_minmax_index, temp_minmax_indexes, temp_names, outer_val
        )
