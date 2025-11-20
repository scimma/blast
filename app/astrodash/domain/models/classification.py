from typing import Optional, Dict, Any

class Classification:
    """
    Domain model representing the result of a spectrum classification.
    This model is independent of storage, API, or infrastructure concerns.
    """
    def __init__(
        self,
        spectrum_id: str,
        model_type: str,
        results: Dict[str, Any],
        user_model_id: Optional[str] = None,
        id: Optional[str] = None,
        meta: Optional[dict] = None
    ):
        self.id = id
        self.spectrum_id = spectrum_id
        self.model_type = model_type
        self.user_model_id = user_model_id
        self.results = results  # e.g., classification probabilities, best match, etc.
        self.meta = meta or {}

    def __repr__(self):
        return (
            f"Classification(id={self.id}, spectrum_id={self.spectrum_id}, model_type={self.model_type}, "
            f"user_model_id={self.user_model_id}, results_keys={list(self.results.keys())})"
        )
