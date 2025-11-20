from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

class UserModelSchema(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier for the user model")
    name: Optional[str] = Field(None, description="Name of the user model")
    description: Optional[str] = Field(None, description="Description of the user model")
    owner: Optional[str] = Field(None, description="Owner of the user model")
    model_path: Optional[str] = Field(None, description="Path to the model weights file")
    class_mapping_path: Optional[str] = Field(None, description="Path to the class mapping file")
    input_shape_path: Optional[str] = Field(None, description="Path to the input shape file")
    created_at: Optional[datetime] = Field(None, description="Datetime when the model was created")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    # Additional fields for frontend compatibility
    model_id: Optional[str] = Field(None, description="Model ID (alias for id)")
    model_filename: Optional[str] = Field(None, description="Original filename of the uploaded model")
    class_mapping: Optional[Dict[str, int]] = Field(None, description="Class mapping dictionary")
    input_shape: Optional[List[Any]] = Field(None, description="Input shape for the model (can be List[int] or List[List[int]])")

    class Config:
        schema_extra = {
            "example": {
                "id": "932eed3d-4d0e-4594-a490-5fd4f5e7a344",
                "name": "My Custom SN Classifier",
                "description": "A user-uploaded model for supernova classification.",
                "owner": "user123",
                        "model_path": "/mnt/astrodash-data/user_models/932eed3d-4d0e-4594-a490-5fd4f5e7a344/932eed3d-4d0e-4594-a490-5fd4f5e7a344.pth",
        "class_mapping_path": "/mnt/astrodash-data/user_models/932eed3d-4d0e-4594-a490-5fd4f5e7a344/932eed3d-4d0e-4594-a490-5fd4f5e7a344.classes.json",
        "input_shape_path": "/mnt/astrodash-data/user_models/932eed3d-4d0e-4594-a490-5fd4f5e7a344/932eed3d-4d0e-4594-a490-5fd4f5e7a344.input_shape.json",
                "created_at": "2024-06-01T12:00:00Z",
                "meta": {"framework": "PyTorch", "num_classes": 5},
                "model_id": "932eed3d-4d0e-4594-a490-5fd4f5e7a344",
                "model_filename": "my_model.pt",
                "class_mapping": {"Ia": 0, "Ib": 1, "Ic": 2},
                "input_shape": [[1, 1024], [1, 1024], [1, 1]]
            }
        }

class ModelUploadResponse(BaseModel):
    """Enhanced response model for model uploads."""
    status: str
    message: str
    model_id: Optional[str] = None
    model_filename: Optional[str] = None
    class_mapping: Optional[Dict[str, int]] = None
    output_shape: Optional[List[int]] = None
    input_shape: Optional[List[int]] = None
    model_info: Optional[Dict[str, Any]] = None

class UserModelInfo(BaseModel):
    model_id: str
    description: str = ""

class ModelInfoResponse(BaseModel):
    """Response model for detailed model information."""
    model_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    uploaded_at: Optional[str] = None
    file_size_bytes: Optional[int] = None
    class_mapping: Optional[Dict[str, int]] = None
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    n_classes: Optional[int] = None
    model_type: Optional[str] = None
    total_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None
