from astrodash.infrastructure.ml.templates import SpectrumTemplateInterface
from astrodash.config.logging import get_logger
from astrodash.core.exceptions import ValidationException
import asyncio
from typing import Dict, Any, List

logger = get_logger(__name__)

class TemplateAnalysisService:
    """
    Service for analyzing and validating spectrum templates.
    Provides methods to get template statistics and validate template requests.
    """

    def __init__(self, template_handler: SpectrumTemplateInterface):
        """Initialize with a template handler."""
        self.template_handler = template_handler
        logger.info("TemplateAnalysisService initialized")

    async def get_analysis_options(self) -> Dict[str, Any]:
        """
        Get available analysis options from templates.

        Returns:
            Dictionary with SN types and their valid age bins

        Raises:
            ValidationException: If template data is invalid
        """
        try:
            logger.info("Getting analysis options from templates")

            # Get all available templates
            templates = await asyncio.to_thread(self.template_handler.get_all_templates)

            # Validate and extract valid SN types and age bins
            valid_options = self._validate_and_extract_options(templates)

            # Extract SN types list
            sn_types = list(valid_options.keys())

            logger.info(f"Found {len(sn_types)} valid SN types: {sn_types}")

            return {
                'sn_types': sn_types,
                'age_bins_by_type': valid_options
            }

        except Exception as e:
            logger.error(f"Error getting analysis options: {e}")
            raise ValidationException(f"Failed to get analysis options: {str(e)}")

    def _validate_and_extract_options(self, templates: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract SN types and age bins from templates.
        Validation is delegated to the template handler.

        Args:
            templates: Raw template data from repository

        Returns:
            Dictionary mapping SN types to valid age bins
        """
        valid_options = {}

        for sn_type, age_bins in templates.items():
            valid_bins = []

            for age_bin in age_bins.keys():
                # Delegate validation to template handler
                if self.template_handler.validate_template(sn_type, age_bin):
                    valid_bins.append(age_bin)

            # Only include SN types that have valid age bins
            if valid_bins:
                valid_options[sn_type] = valid_bins
                logger.debug(f"Valid SN type '{sn_type}' with {len(valid_bins)} age bins")

        return valid_options

    async def get_template_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available templates.

        Returns:
            Dictionary with template statistics
        """
        try:
            templates = await asyncio.to_thread(self.template_handler.get_all_templates)

            total_sn_types = len(templates)
            total_age_bins = sum(len(age_bins) for age_bins in templates.values())
            valid_sn_types = 0
            valid_age_bins = 0

            # Count valid templates using template handler validation
            for sn_type, age_bins in templates.items():
                valid_bins_for_type = 0
                for age_bin in age_bins.keys():
                    if self.template_handler.validate_template(sn_type, age_bin):
                        valid_age_bins += 1
                        valid_bins_for_type += 1

                if valid_bins_for_type > 0:
                    valid_sn_types += 1

            return {
                'total_sn_types': total_sn_types,
                'total_age_bins': total_age_bins,
                'valid_sn_types': valid_sn_types,
                'valid_age_bins': valid_age_bins,
                'validity_rate': {
                    'sn_types': valid_sn_types / total_sn_types if total_sn_types > 0 else 0,
                    'age_bins': valid_age_bins / total_age_bins if total_age_bins > 0 else 0
                }
            }

        except Exception as e:
            logger.error(f"Error getting template statistics: {e}")
            raise ValidationException(f"Failed to get template statistics: {str(e)}")

    async def validate_template_request(self, sn_type: str, age_bin: str) -> bool:
        """
        Validate if a template request is valid.

        Args:
            sn_type: Supernova type to validate
            age_bin: Age bin to validate

        Returns:
            True if template exists and is valid, False otherwise
        """
        try:
            return await asyncio.to_thread(self.template_handler.validate_template, sn_type, age_bin)
        except Exception as e:
            logger.error(f"Error validating template request: {e}")
            return False
