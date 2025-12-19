from functools import lru_cache
from fastapi import Depends
from typing import Generator
from sqlalchemy.orm import Session

# Config and logging imports
from astrodash.config.settings import Settings, get_settings

# Database imports
from astrodash.infrastructure.database.session import get_db
from astrodash.infrastructure.database.sqlalchemy_spectrum_repository import SQLAlchemySpectrumRepository
from astrodash.infrastructure.database.sqlalchemy_model_repository import SQLAlchemyModelRepository

# Storage imports
from astrodash.infrastructure.storage.file_spectrum_repository import FileSpectrumRepository, OSCSpectrumRepository

# ML imports
from astrodash.infrastructure.ml.model_factory import ModelFactory

# Domain services imports
from astrodash.domain.services.template_analysis_service import TemplateAnalysisService
from astrodash.domain.services.line_list_service import LineListService
from astrodash.domain.services.spectrum_processing_service import SpectrumProcessingService
from astrodash.domain.services.batch_processing_service import BatchProcessingService
from astrodash.domain.services.spectrum_service import SpectrumService
from astrodash.domain.services.classification_service import ClassificationService

# Template infrastructure imports
from astrodash.infrastructure.ml.templates import create_spectrum_template_handler

# Settings dependency
@lru_cache()
def get_app_settings() -> Settings:
    """Get the application settings (singleton)."""
    return get_settings()

# Spectrum repository dependencies (file-based)
def get_file_spectrum_repo(settings: Settings = Depends(get_app_settings)) -> FileSpectrumRepository:
    """Dependency to get file-based spectrum repository."""
    return FileSpectrumRepository(config=settings)

def get_osc_spectrum_repo(settings: Settings = Depends(get_app_settings)) -> OSCSpectrumRepository:
    """Dependency to get OSC spectrum repository."""
    return OSCSpectrumRepository(config=settings)

# Model factory dependency
def get_model_factory(settings: Settings = Depends(get_app_settings)) -> ModelFactory:
    """Dependency to get model factory."""
    return ModelFactory(config=settings)

# Model storage dependency
def get_model_storage(settings: Settings = Depends(get_app_settings)):
    """Dependency to get model storage."""
    from astrodash.infrastructure.storage.model_storage import ModelStorage
    return ModelStorage(settings.user_model_dir)

# SQLAlchemy repository dependencies
def get_sqlalchemy_model_repository(db: Session = Depends(get_db)) -> SQLAlchemyModelRepository:
    """Dependency to get SQLAlchemy model repository."""
    return SQLAlchemyModelRepository(db)

def get_sqlalchemy_spectrum_repository(db: Session = Depends(get_db)) -> SQLAlchemySpectrumRepository:
    """Dependency to get SQLAlchemy spectrum repository."""
    return SQLAlchemySpectrumRepository(db)

# Service dependencies
def get_template_analysis_service() -> TemplateAnalysisService:
    """Dependency to get template analysis service."""
    # Create template handler for DASH model (which has templates)
    template_handler = create_spectrum_template_handler('dash')
    return TemplateAnalysisService(template_handler)

def get_line_list_service() -> LineListService:
    """Dependency to get line list service."""
    return LineListService()

def get_spectrum_processing_service(
    settings = Depends(get_app_settings)
) -> SpectrumProcessingService:
    """Dependency to get spectrum processing service."""
    from astrodash.domain.services.spectrum_processing_service import SpectrumProcessingService
    return SpectrumProcessingService(settings)

def get_classification_service(
    model_factory = Depends(get_model_factory)
) -> ClassificationService:
    """Dependency to get classification service."""
    return ClassificationService(model_factory)

def get_model_service(
    model_repo = Depends(get_sqlalchemy_model_repository),
    model_storage = Depends(get_model_storage)
):
    """Dependency to get model service with storage."""
    from astrodash.domain.services.model_service import ModelService
    return ModelService(model_repo, model_storage)

def get_redshift_service(
    settings = Depends(get_app_settings)
):
    """Dependency to get redshift service."""
    from astrodash.domain.services.redshift_service import RedshiftService
    return RedshiftService(settings)

def get_spectrum_service(
    file_repo = Depends(get_file_spectrum_repo),
    osc_repo = Depends(get_osc_spectrum_repo),
    db_repo = Depends(get_sqlalchemy_spectrum_repository),
    settings = Depends(get_app_settings)
):
    """Dependency to get spectrum service."""
    from astrodash.domain.services.spectrum_service import SpectrumService
    return SpectrumService(file_repo, osc_repo, db_repo, settings)

def get_batch_processing_service(
    spectrum_service = Depends(get_spectrum_service),
    classification_service = Depends(get_classification_service),
    processing_service = Depends(get_spectrum_processing_service)
) -> BatchProcessingService:
    """Dependency to get batch processing service."""
    return BatchProcessingService(spectrum_service, classification_service, processing_service)
