from http import HTTPStatus

HTTP_400_BAD_REQUEST = HTTPStatus.BAD_REQUEST.value
HTTP_404_NOT_FOUND = HTTPStatus.NOT_FOUND.value
HTTP_500_INTERNAL_SERVER_ERROR = HTTPStatus.INTERNAL_SERVER_ERROR.value
HTTP_422_UNPROCESSABLE_ENTITY = HTTPStatus.UNPROCESSABLE_ENTITY.value
HTTP_409_CONFLICT = HTTPStatus.CONFLICT.value

class AppException(Exception):
    """Base exception for the application."""
    def __init__(self, message: str, status_code: int = HTTP_400_BAD_REQUEST):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

# Data-related exceptions
class SpectrumNotFoundException(AppException):
    """Raised when a spectrum is not found."""
    def __init__(self, spectrum_id: str):
        super().__init__(f"Spectrum with ID '{spectrum_id}' not found.", status_code=HTTP_404_NOT_FOUND)

class ModelNotFoundException(AppException):
    """Raised when a model is not found."""
    def __init__(self, model_id: str):
        super().__init__(f"Model with ID '{model_id}' not found.", status_code=HTTP_404_NOT_FOUND)

class TemplateNotFoundException(AppException):
    """Raised when a template is not found."""
    def __init__(self, sn_type: str, age_bin: str = None):
        if age_bin:
            message = f"Template not found for SN type '{sn_type}' and age bin '{age_bin}'."
        else:
            message = f"Template not found for SN type '{sn_type}'."
        super().__init__(message, status_code=HTTP_404_NOT_FOUND)

class LineListNotFoundException(AppException):
    """Raised when the line list file is not found."""
    def __init__(self, file_path: str = None):
        message = "Line list file not found. Please ensure the sneLineList.txt file is available."
        if file_path:
            message += f" Expected at: {file_path}"
        super().__init__(message, status_code=HTTP_404_NOT_FOUND)

class ElementNotFoundException(AppException):
    """Raised when an element is not found in the line list."""
    def __init__(self, element: str):
        super().__init__(f"Element '{element}' not found in line list.", status_code=HTTP_404_NOT_FOUND)

# Processing and classification exceptions
class ClassificationException(AppException):
    """Raised for errors during classification."""
    def __init__(self, message: str = "Classification failed."):
        super().__init__(message, status_code=HTTP_400_BAD_REQUEST)

class SpectrumProcessingException(AppException):
    """Raised for errors during spectrum processing."""
    def __init__(self, message: str = "Spectrum processing failed."):
        super().__init__(message, status_code=HTTP_400_BAD_REQUEST)

class ModelProcessingException(AppException):
    """Raised for errors during model processing."""
    def __init__(self, message: str = "Model processing failed."):
        super().__init__(message, status_code=HTTP_400_BAD_REQUEST)

class BatchProcessingException(AppException):
    """Raised for errors during batch processing."""
    def __init__(self, message: str = "Batch processing failed."):
        super().__init__(message, status_code=HTTP_400_BAD_REQUEST)

# Validation exceptions
class ValidationException(AppException):
    """Raised for validation errors."""
    def __init__(self, message: str = "Validation failed."):
        super().__init__(message, status_code=HTTP_422_UNPROCESSABLE_ENTITY)

class FileValidationException(ValidationException):
    """Raised for file validation errors."""
    def __init__(self, message: str = "File validation failed."):
        super().__init__(message)

class ModelValidationException(ValidationException):
    """Raised for model validation errors."""
    def __init__(self, message: str = "Model validation failed."):
        super().__init__(message)

class SpectrumValidationException(ValidationException):
    """Raised for spectrum validation errors."""
    def __init__(self, message: str = "Spectrum validation failed."):
        super().__init__(message)

# Storage and file exceptions
class StorageException(AppException):
    """Raised for storage-related errors."""
    def __init__(self, message: str = "Storage error."):
        super().__init__(message, status_code=HTTP_500_INTERNAL_SERVER_ERROR)

class FileNotFoundException(AppException):
    """Raised when a file is not found."""
    def __init__(self, file_path: str):
        super().__init__(f"File not found: {file_path}", status_code=HTTP_404_NOT_FOUND)

class FileReadException(AppException):
    """Raised when there's an error reading a file."""
    def __init__(self, file_path: str, error: str = None):
        message = f"Error reading file: {file_path}"
        if error:
            message += f" - {error}"
        super().__init__(message, status_code=HTTP_400_BAD_REQUEST)

class UnsupportedFileFormatException(AppException):
    """Raised when an unsupported file format is provided."""
    def __init__(self, file_format: str, supported_formats: list = None):
        message = f"Unsupported file format: {file_format}"
        if supported_formats:
            message += f". Supported formats: {', '.join(supported_formats)}"
        super().__init__(message, status_code=HTTP_400_BAD_REQUEST)

# Configuration and setup exceptions
class ConfigurationException(AppException):
    """Raised for configuration errors."""
    def __init__(self, message: str = "Configuration error."):
        super().__init__(message, status_code=HTTP_500_INTERNAL_SERVER_ERROR)

class ModelConfigurationException(ConfigurationException):
    """Raised for model configuration errors."""
    def __init__(self, message: str = "Model configuration error."):
        super().__init__(message)

# External service exceptions
class ExternalServiceException(AppException):
    """Raised for external service errors."""
    def __init__(self, service: str, message: str = None):
        if message:
            error_message = f"{service} service error: {message}"
        else:
            error_message = f"{service} service error."
        super().__init__(error_message, status_code=HTTP_500_INTERNAL_SERVER_ERROR)

class OSCServiceException(ExternalServiceException):
    """Raised for OSC (Open Supernova Catalog) service errors."""
    def __init__(self, message: str = None):
        super().__init__("OSC", message)

# Resource and constraint exceptions
class ResourceNotFoundException(AppException):
    """Raised when a resource is not found."""
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(f"{resource_type} with ID '{resource_id}' not found.", status_code=HTTP_404_NOT_FOUND)

class ResourceConflictException(AppException):
    """Raised when there's a conflict with a resource."""
    def __init__(self, message: str = "Resource conflict."):
        super().__init__(message, status_code=HTTP_409_CONFLICT)

class ModelConflictException(ResourceConflictException):
    """Raised when there's a conflict with a model (e.g., duplicate name)."""
    def __init__(self, model_name: str):
        super().__init__(f"Model with name '{model_name}' already exists.")
