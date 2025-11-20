"""
Template handling infrastructure for spectrum templates.
Contains interfaces and implementations for different model types.
"""

from .template_interface import SpectrumTemplateInterface
from .dash_template_handler import DASHSpectrumTemplate
from .transformer_template_handler import TransformerSpectrumTemplate
from .template_factory import create_spectrum_template_handler

__all__ = [
    'SpectrumTemplateInterface',
    'DASHSpectrumTemplate',
    'TransformerSpectrumTemplate',
    'create_spectrum_template_handler'
]
