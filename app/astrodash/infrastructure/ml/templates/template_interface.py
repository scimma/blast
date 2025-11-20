"""
Abstract interface for spectrum template handlers.
Defines the contract for template operations across different model types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from astrodash.core.exceptions import TemplateNotFoundException

class SpectrumTemplateInterface(ABC):
    """
    Abstract interface for spectrum template handlers.

    All methods should raise TemplateNotFoundException when templates are not found.
    """

    @abstractmethod
    def get_template_spectrum(self, sn_type: str, age_bin: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get template spectrum for given SN type and age bin.

        Args:
            sn_type: Supernova type (e.g., 'Ia', 'II')
            age_bin: Age bin identifier

        Returns:
            Tuple of (wavelength_array, flux_array)

        Raises:
            TemplateNotFoundException: If template is not found
        """
        pass

    @abstractmethod
    def get_all_templates(self) -> Dict[str, Any]:
        """
        Get all available templates.

        Returns:
            Dictionary containing all template data
        """
        pass

    @abstractmethod
    def validate_template(self, sn_type: str, age_bin: str) -> bool:
        """
        Validate if template exists for given SN type and age bin.

        Args:
            sn_type: Supernova type
            age_bin: Age bin identifier

        Returns:
            True if template exists and is valid, False otherwise
        """
        pass
