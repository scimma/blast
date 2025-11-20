from typing import Optional, List, Dict, Any, Tuple
from astrodash.domain.models.user_model import UserModel
from astrodash.domain.repositories.model_repository import ModelRepository
from astrodash.infrastructure.ml.model_loader import ModelLoader, ModelValidator
from astrodash.infrastructure.storage.model_storage import ModelStorage
from astrodash.shared.utils.validators import validate_model_upload_request, ValidationError, validate_user_model_basic
from astrodash.core.exceptions import (
    ModelNotFoundException,
    ModelValidationException,
    ModelConflictException,
    ConfigurationException,
    ValidationException
)
from astrodash.config.logging import get_logger
import uuid
import asyncio

logger = get_logger(__name__)

class ModelService:
    """
    Service layer for user-uploaded model operations.
    """

    def __init__(self, model_repo: ModelRepository, model_storage: ModelStorage = None):
        self.model_repo = model_repo
        self.model_storage = model_storage
        self.model_loader = ModelLoader()
        logger.info("ModelService initialized")

    async def upload_model(
        self,
        model_content: bytes,
        filename: str,
        class_mapping_str: str,
        input_shape_str: str,
        name: str = "",
        description: str = "",
        owner: str = None
    ) -> Tuple[UserModel, Dict[str, Any]]:
        """
        Upload and validate a new model.

        Args:
            model_content: Raw model file content
            filename: Name of the uploaded file
            class_mapping_str: JSON string containing class mapping
            input_shape_str: JSON string containing input shape
            description: Optional model description
            owner: Optional model owner

        Returns:
            Tuple of (UserModel, model_info)

        Raises:
            ModelValidationException: If validation fails
            ModelConflictException: If model with same name already exists
            ConfigurationException: If storage is not available
        """
        try:
            # Validate the upload request
            class_mapping, input_shape = validate_model_upload_request(
                filename, class_mapping_str, input_shape_str
            )

            # Generate unique model ID
            model_id = str(uuid.uuid4())

            # Wrap single input shape
            if input_shape and isinstance(input_shape[0], list):
                input_shapes = input_shape
            else:
                input_shapes = [input_shape]

            # Save model files to storage
            if self.model_storage:
                file_paths = self.model_storage.save_model_files(
                    model_id=model_id,
                    model_content=model_content,
                    class_mapping=class_mapping,
                    input_shape=input_shape,
                    metadata={"description": description, "owner": owner}
                )

                # Validate the saved model
                model_info = self._validate_saved_model(
                    model_id, input_shapes, class_mapping
                )
            else:
                # Fallback: validate in memory
                model_info = self._validate_model_in_memory(
                    model_content, input_shapes, class_mapping
                )
                file_paths = {
                    "model": f"temp_{model_id}.pth",
                    "class_mapping": f"temp_{model_id}.classes.json",
                    "input_shape": f"temp_{model_id}.input_shape.json",
                    "metadata": f"temp_{model_id}.meta.json"
                }

            # Create UserModel instance
            user_model = UserModel(
                id=model_id,
                name=name,
                description=description,
                owner=owner,
                model_path=file_paths["model"],
                class_mapping_path=file_paths["class_mapping"],
                input_shape_path=file_paths["input_shape"]
            )

            # Save to repository
            saved_model = await asyncio.to_thread(self.model_repo.save, user_model)

            return saved_model, model_info

        except Exception as e:
            logger.error(f"Error uploading model: {e}", exc_info=True)
            # Clean up any saved files on error
            if 'model_id' in locals() and self.model_storage:
                self.model_storage.cleanup_model_files(model_id)
            raise

    def _validate_saved_model(
        self,
        model_id: str,
        input_shapes: List[List[int]],
        class_mapping: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Validate a model that has been saved to storage.

        Args:
            model_id: Unique identifier for the model
            input_shapes: List of input shapes for each model input
            class_mapping: Dictionary mapping class names to indices

        Returns:
            Dictionary containing model information
        """
        try:
            model_path = self.model_storage.get_model_path(model_id)

            # Load and validate model
            model = self.model_loader.load_model(model_path)

            # Validate with inputs
            output_shape, model_info = self.model_loader.validate_model_with_inputs(
                model, input_shapes, class_mapping
            )

            # Extract additional metadata
            metadata = self.model_loader.extract_model_metadata(model)
            model_info.update(metadata)

            # Cleanup
            self.model_loader.cleanup_model(model)

            return model_info

        except Exception as e:
            logger.error(f"Model validation failed for {model_id}: {e}")
            raise ModelValidationException(f"Model validation failed: {str(e)}")

    def _validate_model_in_memory(
        self,
        model_content: bytes,
        input_shapes: List[List[int]],
        class_mapping: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Validate a model from memory content (fallback method).

        Args:
            model_content: Raw model file content
            input_shapes: List of input shapes for each model input
            class_mapping: Dictionary mapping class names to indices

        Returns:
            Dictionary containing model information
        """
        import tempfile
        import os

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
                temp_file.write(model_content)
                temp_path = temp_file.name

            try:
                # Load and validate model
                model = self.model_loader.load_model(temp_path)

                # Validate with inputs
                output_shape, model_info = self.model_loader.validate_model_with_inputs(
                    model, input_shapes, class_mapping
                )

                # Extract additional metadata
                metadata = self.model_loader.extract_model_metadata(model)
                model_info.update(metadata)

                # Cleanup
                self.model_loader.cleanup_model(model)

                return model_info

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.error(f"In-memory model validation failed: {e}")
            raise ModelValidationException(f"Model validation failed: {str(e)}")

    async def save_model(self, model: UserModel) -> UserModel:
        """
        Save a model to the repository with business rule validation.
        """
        # Validate model before saving
        try:
            validate_user_model_basic(model.model_path, model.class_mapping_path, model.input_shape_path)
        except Exception as e:
            raise ModelValidationException(f"Model validation failed: {str(e)}")

        # Check for duplicate names (if name is provided)
        if model.name:
            existing_models = await asyncio.to_thread(self.model_repo.get_by_owner, model.owner or "")
            for existing in existing_models:
                if existing.name == model.name:
                    raise ModelConflictException(f"Model with name '{model.name}' already exists")

        return await asyncio.to_thread(self.model_repo.save, model)

    async def get_model(self, model_id: str) -> UserModel:
        """
        Get a model by ID with additional validation.
        """
        model = await asyncio.to_thread(self.model_repo.get_by_id, model_id)
        if not model:
            raise ModelNotFoundException(model_id)

        # Verify model files still exist
        if self.model_storage and not self.model_storage.model_exists(model_id):
            logger.warning(f"Model files missing for {model_id}")
            # Could implement cleanup here if needed

        return model

    async def list_models(self) -> List[UserModel]:
        """
        List all models with optional filtering.
        """
        return await asyncio.to_thread(self.model_repo.list_all)

    async def delete_model(self, model_id: str) -> None:
        """
        Delete a model with cleanup.
        """
        model = await asyncio.to_thread(self.model_repo.get_by_id, model_id)
        if not model:
            raise ModelNotFoundException(model_id)

        # Clean up storage files
        if self.model_storage:
            self.model_storage.cleanup_model_files(model_id)

        # Delete from repository
        await asyncio.to_thread(self.model_repo.delete, model_id)
        logger.info(f"Model {model_id} deleted successfully")

    async def list_models_by_owner(self, owner: str) -> List[UserModel]:
        """
        List models by owner with validation.
        """
        if not owner:
            raise ValidationException("Owner cannot be empty")

        return await asyncio.to_thread(self.model_repo.get_by_owner, owner)

    async def update_model_metadata(self, model_id: str, updates: Dict[str, Any]) -> UserModel:
        """
        Update model metadata with validation.
        """
        model = await self.get_model(model_id)

        # Validate updates
        allowed_updates = {"name", "description", "meta"}
        invalid_updates = set(updates.keys()) - allowed_updates
        if invalid_updates:
            raise ModelValidationException(f"Invalid update fields: {invalid_updates}")

        # Apply updates
        for key, value in updates.items():
            if hasattr(model, key):
                setattr(model, key, value)

        # Update storage metadata if available
        if self.model_storage:
            self.model_storage.update_metadata(model_id, updates)

        # Save to repository
        return await asyncio.to_thread(self.model_repo.save, model)

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        """
        if not self.model_storage:
            raise ConfigurationException("Model storage not available")

        try:
            metadata = self.model_storage.load_model_metadata(model_id)
            class_mapping = self.model_storage.load_class_mapping(model_id)
            input_shape = self.model_storage.load_input_shape(model_id)
            file_size = self.model_storage.get_model_size(model_id)

            return {
                **metadata,
                "class_mapping": class_mapping,
                "input_shape": input_shape,
                "file_size": file_size,
                "model_id": model_id
            }
        except Exception as e:
            logger.error(f"Failed to get model info for {model_id}: {e}")
            raise ConfigurationException(f"Failed to get model info: {str(e)}")
