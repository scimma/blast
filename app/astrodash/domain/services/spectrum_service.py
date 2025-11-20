from typing import Any, Optional
from astrodash.domain.models.spectrum import Spectrum
from astrodash.domain.repositories.spectrum_repository import SpectrumRepository
from astrodash.config.settings import Settings, get_settings
from astrodash.shared.utils.validators import validate_file_extension, validate_spectrum
from astrodash.config.logging import get_logger
from astrodash.core.exceptions import (
    SpectrumValidationException,
    FileReadException,
    OSCServiceException,
    ValidationException
)
import asyncio

logger = get_logger(__name__)

class SpectrumService:
    def __init__(self, file_repo: SpectrumRepository, osc_repo: SpectrumRepository, db_repo: SpectrumRepository, settings: Optional[Settings] = None):
        """Service for spectrum operations. Injects repositories and settings."""
        self.file_repo = file_repo
        self.osc_repo = osc_repo
        self.db_repo = db_repo
        self.settings = settings or get_settings()

    async def get_spectrum_from_file(self, file: Any) -> Spectrum:
        logger.debug(f"Getting spectrum from file: {getattr(file, 'filename', 'unknown')}")
        spectrum = await asyncio.to_thread(self.file_repo.get_from_file, file)
        logger.debug(f"Repository returned spectrum: {spectrum}")

        if not spectrum:
            logger.error("Repository returned None spectrum")
            raise FileReadException(getattr(file, 'filename', 'unknown'), "Invalid or unreadable spectrum file.")

        try:
            validate_spectrum(spectrum.x, spectrum.y, spectrum.redshift)
            logger.debug("Spectrum validation passed")
        except Exception as e:
            logger.error(f"Spectrum validation failed: {e}")
            raise SpectrumValidationException("Invalid or unreadable spectrum file.")

        return spectrum

    async def get_spectrum_from_osc(self, osc_ref: str) -> Spectrum:
        logger.debug(f"Spectrum service: Starting to get spectrum for OSC reference: {osc_ref}")

        # Fetch from DB if stored
        try:
            db_spectrum = await asyncio.to_thread(self.db_repo.get_by_osc_ref, osc_ref)
        except Exception as e:
            logger.debug(f"Spectrum service: DB lookup by osc_ref failed: {e}")
            db_spectrum = None

        if db_spectrum:
            logger.debug(f"Spectrum service: Found spectrum in DB for {osc_ref}: {db_spectrum.id}")
            try:
                validate_spectrum(db_spectrum.x, db_spectrum.y, db_spectrum.redshift)
                logger.debug(f"Spectrum service: Successfully validated DB spectrum for {osc_ref}")
            except Exception as e:
                logger.error(f"Spectrum service: Stored spectrum validation failed for {osc_ref}: {e}")
                raise OSCServiceException(
                    f"Stored spectrum invalid for reference: {osc_ref}. Consider re-fetching."
                )
            return db_spectrum

        # Fallback to OSC repository
        logger.debug(f"Spectrum service: Not found in DB. Fetching from OSC for {osc_ref}")
        spectrum = await asyncio.to_thread(self.osc_repo.get_by_osc_ref, osc_ref)
        logger.debug(f"Spectrum service: OSC repository returned spectrum: {spectrum}")

        if not spectrum:
            logger.error(f"Spectrum service: OSC repository returned None spectrum for {osc_ref}")
            raise OSCServiceException(f"Could not retrieve valid spectrum data from OSC for reference: {osc_ref}. The spectrum may not exist or the OSC API may be unavailable.")

        try:
            validate_spectrum(spectrum.x, spectrum.y, spectrum.redshift)
            logger.debug(f"Spectrum service: Successfully retrieved and validated OSC spectrum for {osc_ref}")
        except Exception as e:
            logger.error(f"Spectrum service: Spectrum validation failed for {osc_ref}: {e}")
            raise OSCServiceException(f"Could not retrieve valid spectrum data from OSC for reference: {osc_ref}. The spectrum may not exist or the OSC API may be unavailable.")

        return spectrum

    async def get_spectrum_data(self, file: Optional[Any] = None, osc_ref: Optional[str] = None) -> Spectrum:
        """
        Get spectrum data from file or OSC reference.

        Args:
            file: UploadFile object or file-like object
            osc_ref: OSC reference string

        Returns:
            Spectrum object

        Raises:
            ValidationException: If no valid spectrum source is provided
            SpectrumValidationException: If spectrum is invalid
            FileReadException: If file cannot be read
            OSCServiceException: If OSC service fails
        """
        try:
            if file:
                logger.debug(f"Processing uploaded file: {getattr(file, 'filename', 'unknown')}")
                # Validate file extension (now supports .fits as well)
                validate_file_extension(getattr(file, 'filename', ''), [".dat", ".lnw", ".txt", ".fits"])
                return await self.get_spectrum_from_file(file)
            elif osc_ref:
                logger.debug(f"Processing OSC reference: {osc_ref}")
                return await self.get_spectrum_from_osc(osc_ref)
            else:
                raise ValidationException("No spectrum file or OSC reference provided")
        except (ValidationException, SpectrumValidationException, FileReadException, OSCServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting spectrum data: {e}")
            raise SpectrumValidationException(f"Spectrum data error: {str(e)}")

    async def save_spectrum(self, spectrum: Spectrum) -> Spectrum:
        """
        Save spectrum to database.

        Args:
            spectrum: Spectrum object to save

        Returns:
            Spectrum object (after saving)

        Raises:
            SpectrumValidationException: If spectrum validation fails
        """
        try:
            logger.debug(f"Saving spectrum {spectrum.id} to database")

            # Validate spectrum before saving
            validate_spectrum(spectrum.x, spectrum.y, spectrum.redshift)

            # Save to database using async wrapper
            saved_spectrum = await asyncio.to_thread(self.db_repo.save, spectrum)

            logger.debug(f"Successfully saved spectrum {spectrum.id} to database")
            return saved_spectrum

        except Exception as e:
            logger.error(f"Error saving spectrum to database: {e}")
            raise SpectrumValidationException(f"Failed to save spectrum: {str(e)}")
