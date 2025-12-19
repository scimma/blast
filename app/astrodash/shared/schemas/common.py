from typing import Any, List, Generic, TypeVar, Optional
from pydantic import BaseModel, Field

T = TypeVar('T')

class APIResponse(BaseModel):
    """Generic API response schema."""
    message: str = Field(..., description="A human-readable message.")
    data: Optional[Any] = Field(None, description="Response data.")

class ErrorResponse(BaseModel):
    """Standard error response schema."""
    detail: str = Field(..., description="Error detail message.")

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response schema."""
    total: int = Field(..., description="Total number of items.")
    items: List[T] = Field(..., description="List of items on this page.")
    page: int = Field(..., description="Current page number.")
    size: int = Field(..., description="Number of items per page.")
