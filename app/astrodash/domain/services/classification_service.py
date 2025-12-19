from typing import Optional, Dict, Any
from astrodash.domain.models.spectrum import Spectrum
from astrodash.domain.models.classification import Classification
from astrodash.infrastructure.ml.model_factory import ModelFactory
from astrodash.infrastructure.ml.classifiers.base import BaseClassifier
from astrodash.config.settings import Settings, get_settings
from astrodash.config.logging import get_logger
from astrodash.core.exceptions import ClassificationException

logger = get_logger(__name__)

class ClassificationService:
    def __init__(self, model_factory: ModelFactory, settings: Optional[Settings] = None):
        """Service for classification operations. Injects model factory and settings."""
        self.model_factory = model_factory
        self.settings = settings or get_settings()

    async def classify_spectrum(
        self,
        spectrum: Spectrum,
        model_type: str,
        user_model_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        classifier: Optional[BaseClassifier] = None
    ) -> Classification:
        """
        Classify spectrum using the specified model type or user-uploaded model.

        Args:
            spectrum: Spectrum to classify
            model_type: Type of model ('dash', 'transformer', 'user_uploaded')
            user_model_id: ID of user-uploaded model (if applicable)
            params: Additional parameters for classification (e.g., calculateRlap)

        Returns:
            Classification object with results

        Raises:
            ClassificationException: If classification fails or returns no results
        """
        # Determine the actual model type for user-uploaded models
        if user_model_id:
            model_type = "user_uploaded"
            logger.info(f"Using user-uploaded model: {user_model_id}")

        # Reuse provided classifier when available to avoid re-loading models repeatedly
        if classifier is None:
            classifier = self.model_factory.get_classifier(model_type, user_model_id)
        results = await classifier.classify(spectrum)

        if not results:
            raise ClassificationException("Classification failed or returned no results.")

        # Handle RLAP calculation for user-uploaded models
        if model_type == "user_uploaded" and params:
            calculate_rlap = params.get('calculateRlap', False)
            if calculate_rlap:
                logger.info("RLAP calculation requested but not supported by user-uploaded models")
                # Set RLAP to None for all matches
                for match in results.get("best_matches", []):
                    match["rlap"] = None

        return Classification(
            spectrum_id=getattr(spectrum, 'id', None),
            model_type=model_type,
            user_model_id=user_model_id,
            results=results
        )
