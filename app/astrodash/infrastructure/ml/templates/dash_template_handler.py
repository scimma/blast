"""
DASH-specific template handler implementation.
Handles loading and validation of DASH model templates.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import os
from astrodash.infrastructure.ml.templates.template_interface import SpectrumTemplateInterface
from astrodash.core.exceptions import TemplateNotFoundException
from astrodash.config.logging import get_logger

logger = get_logger(__name__)

class DASHSpectrumTemplate(SpectrumTemplateInterface):
    """DASH-specific template handler."""

    def __init__(self, template_path: str):
        self.template_path = template_path
        self._templates: Optional[Dict[str, Any]] = None
        logger.info(f"DASHSpectrumTemplate initialized with path: {template_path}")

    def get_template_spectrum(self, sn_type: str, age_bin: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get template spectrum for DASH model."""
        try:
            templates = self._load_templates()

            if sn_type not in templates:
                raise TemplateNotFoundException(sn_type)

            if age_bin not in templates[sn_type]:
                raise TemplateNotFoundException(sn_type, age_bin)

            entry = templates[sn_type][age_bin]
            sn_info = entry.get('snInfo', None)

            if not isinstance(sn_info, np.ndarray) or sn_info.shape[0] == 0:
                raise TemplateNotFoundException(sn_type, age_bin)

            # Get the first template (placeholder for now)
            template = sn_info[0]
            wave = template[0]
            flux = template[1]

            logger.info(f"Template spectrum loaded for {sn_type} / {age_bin}")
            return wave, flux

        except TemplateNotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error loading template spectrum: {e}")
            raise TemplateNotFoundException(sn_type, age_bin)

    def get_all_templates(self) -> Dict[str, Any]:
        """Get all DASH templates."""
        try:
            return self._load_templates()
        except Exception as e:
            logger.error(f"Error loading all templates: {e}")
            raise

    def validate_template(self, sn_type: str, age_bin: str) -> bool:
        """Validate DASH template."""
        try:
            templates = self._load_templates()
            return (sn_type in templates and
                   age_bin in templates[sn_type] and
                   self._is_valid_entry(templates[sn_type][age_bin]))
        except Exception as e:
            logger.error(f"Error validating template: {e}")
            return False

    def _load_templates(self) -> Dict[str, Any]:
        """Load templates from astrodash.file."""
        if self._templates is None:
            try:
                data = np.load(self.template_path, allow_pickle=True)
                sn_templates_raw = data['snTemplates'].item()
                self._templates = {str(k): v for k, v in sn_templates_raw.items()}
                logger.info(f"Templates loaded: {list(self._templates.keys())}")
            except Exception as e:
                logger.error(f"Error loading templates from {self.template_path}: {e}")
                raise

        return self._templates

    def _is_valid_entry(self, entry: Any) -> bool:
        """Check if template entry is valid."""
        try:
            sn_info = entry.get('snInfo', None)
            return (isinstance(sn_info, np.ndarray) and
                   sn_info.shape and
                   len(sn_info.shape) == 2 and
                   sn_info.shape[0] > 0 and
                   sn_info.shape[1] == 4)
        except Exception:
            return False
