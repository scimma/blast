from typing import Optional, Dict, Any
from datetime import datetime

class UserModel:
    def __init__(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        model_path: Optional[str] = None,
        class_mapping_path: Optional[str] = None,
        input_shape_path: Optional[str] = None,
        created_at: Optional[datetime] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.name = name
        self.description = description
        self.owner = owner
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path
        self.input_shape_path = input_shape_path
        self.created_at = created_at or datetime.utcnow()
        self.meta = meta or {}
