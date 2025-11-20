from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class ClassificationSchema(BaseModel):
    """
    Pydantic schema for validating and serializing Classification data in the API layer.
    """
    id: Optional[str] = Field(None, description="Unique identifier for the classification result")
    spectrum_id: str = Field(..., description="ID of the classified spectrum")
    model_type: str = Field(..., description="Type of model used for classification (e.g., 'dash', 'transformer')")
    user_model_id: Optional[str] = Field(None, description="User-uploaded model ID, if applicable")
    results: Dict[str, Any] = Field(..., description="Classification results (probabilities, best match, etc.)")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "id": "cls-123",
                "spectrum_id": "abc123",
                "model_type": "dash",
                "user_model_id": None,
                "results": {
                    "best_type": "Ia",
                    "probabilities": {"Ia": 0.95, "II": 0.03, "Ib": 0.02},
                    "redshift": 0.123
                },
                "meta": {"timestamp": "2024-05-01T12:00:00Z"}
            }
        }
