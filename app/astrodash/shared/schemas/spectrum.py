from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SpectrumSchema(BaseModel):
    """
    Pydantic schema for validating and serializing Spectrum data in the API layer.
    """
    id: Optional[str] = Field(None, description="Unique identifier for the spectrum")
    osc_ref: Optional[str] = Field(None, description="OSC reference string, if applicable")
    file_name: Optional[str] = Field(None, description="Original file name, if applicable")
    x: List[float] = Field(..., description="Wavelength values")
    y: List[float] = Field(..., description="Flux values")
    redshift: Optional[float] = Field(None, description="Redshift value, if known")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "id": "abc123",
                "osc_ref": "osc-2021xyz",
                "file_name": "spectrum1.fits",
                "x": [3500.0, 3501.0, 3502.0],
                "y": [1.2, 1.3, 1.1],
                "redshift": 0.123,
                "meta": {"instrument": "Keck", "observer": "Jane Doe"}
            }
        }
