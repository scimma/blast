from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
from astrodash.domain.models.spectrum import Spectrum
from astrodash.config.settings import get_settings
from astrodash.config.logging import get_logger
from astrodash.core.exceptions import (
    TemplateNotFoundException,
    FileNotFoundException,
    ModelConfigurationException,
    SpectrumValidationException,
    FileReadException
)
import numpy as np
import os

logger = get_logger(__name__)

class SpectrumRepository(ABC):
    """
    Abstract repository interface for spectrum data.
    Follows the repository pattern for decoupling domain and infrastructure.

    All methods should raise appropriate exceptions when operations fail:
    - FileReadException: When file reading fails
    - SpectrumValidationException: When spectrum validation fails
    - FileNotFoundException: When files are not found
    """

    @abstractmethod
    def save(self, spectrum: Spectrum) -> Spectrum:
        """
        Save a spectrum to persistent storage.

        Args:
            spectrum: The Spectrum to save

        Returns:
            The saved Spectrum with updated fields

        Raises:
            SpectrumValidationException: If the spectrum is invalid
        """
        pass

    @abstractmethod
    def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        """
        Retrieve a spectrum by its unique ID.

        Args:
            spectrum_id: The unique identifier of the spectrum

        Returns:
            The Spectrum if found, None otherwise
        """
        pass

    @abstractmethod
    def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        """
        Retrieve a spectrum by its OSC reference.

        Args:
            osc_ref: The OSC reference identifier

        Returns:
            The Spectrum if found, None otherwise
        """
        pass

    @abstractmethod
    def get_from_file(self, file: Any) -> Optional[Spectrum]:
        """
        Read and parse a spectrum from a file.

        Args:
            file: File object (UploadFile or file-like object)

        Returns:
            The parsed Spectrum if successful, None otherwise

        Raises:
            FileReadException: If file reading fails
            SpectrumValidationException: If parsed spectrum is invalid
        """
        pass
