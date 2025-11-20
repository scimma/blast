import os
import json
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from astrodash.config.logging import get_logger

logger = get_logger(__name__)

class ModelStorage:
    """
    Infrastructure component for managing model file storage and metadata.
    Handles file operations, metadata persistence, and cleanup.
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelStorage initialized with base directory: {self.base_dir}")

    def save_model_files(
        self,
        model_id: str,
        model_content: bytes,
        class_mapping: Dict[str, int],
        input_shape: List[int],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Save model files and metadata to storage.

        Args:
            model_id: Unique identifier for the model
            model_content: Raw model file content
            class_mapping: Class name to index mapping
            input_shape: Model input shape
            metadata: Additional model metadata

        Returns:
            Dictionary mapping file types to their paths

        Raises:
            ValueError: If saving fails
        """
        try:
            # Create model directory
            model_dir = self.base_dir / model_id
            model_dir.mkdir(exist_ok=True)

            # Define file paths
            model_path = model_dir / f"{model_id}.pth"
            class_mapping_path = model_dir / f"{model_id}.classes.json"
            input_shape_path = model_dir / f"{model_id}.input_shape.json"
            metadata_path = model_dir / f"{model_id}.metadata.json"

            # Save model file
            with open(model_path, 'wb') as f:
                f.write(model_content)

            # Save class mapping
            with open(class_mapping_path, 'w') as f:
                json.dump(class_mapping, f, indent=2)

            # Save input shape
            with open(input_shape_path, 'w') as f:
                json.dump(input_shape, f, indent=2)

            # Save metadata
            model_metadata = {
                "model_id": model_id,
                "uploaded_at": datetime.utcnow().isoformat(),
                "file_size": len(model_content),
                "class_mapping": class_mapping,
                "input_shape": input_shape,
                **(metadata or {})
            }

            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)

            file_paths = {
                "model": str(model_path),
                "class_mapping": str(class_mapping_path),
                "input_shape": str(input_shape_path),
                "metadata": str(metadata_path)
            }

            logger.info(f"Model files saved successfully for model_id: {model_id}")
            return file_paths

        except Exception as e:
            logger.error(f"Failed to save model files for {model_id}: {e}")
            # Cleanup on failure
            self.cleanup_model_files(model_id)
            raise ValueError(f"Failed to save model files: {str(e)}")

    def load_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Load model metadata from astrodash.storage.

        Args:
            model_id: Unique identifier for the model

        Returns:
            Dictionary containing model metadata

        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        metadata_path = self.base_dir / model_id / f"{model_id}.metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found for model_id: {model_id}")

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata for {model_id}: {e}")
            raise ValueError(f"Failed to load model metadata: {str(e)}")

    def load_class_mapping(self, model_id: str) -> Dict[str, int]:
        """
        Load class mapping from astrodash.storage.

        Args:
            model_id: Unique identifier for the model

        Returns:
            Dictionary mapping class names to indices
        """
        class_mapping_path = self.base_dir / model_id / f"{model_id}.classes.json"

        if not class_mapping_path.exists():
            raise FileNotFoundError(f"Class mapping file not found for model_id: {model_id}")

        try:
            with open(class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
            return class_mapping
        except Exception as e:
            logger.error(f"Failed to load class mapping for {model_id}: {e}")
            raise ValueError(f"Failed to load class mapping: {str(e)}")

    def load_input_shape(self, model_id: str) -> List[int]:
        """
        Load input shape from astrodash.storage.

        Args:
            model_id: Unique identifier for the model

        Returns:
            List representing input shape
        """
        input_shape_path = self.base_dir / model_id / f"{model_id}.input_shape.json"

        if not input_shape_path.exists():
            raise FileNotFoundError(f"Input shape file not found for model_id: {model_id}")

        try:
            with open(input_shape_path, 'r') as f:
                input_shape = json.load(f)
            return input_shape
        except Exception as e:
            logger.error(f"Failed to load input shape for {model_id}: {e}")
            raise ValueError(f"Failed to load input shape: {str(e)}")

    def get_model_path(self, model_id: str) -> str:
        """
        Get the path to the model file.

        Args:
            model_id: Unique identifier for the model

        Returns:
            Path to the model file
        """
        model_path = self.base_dir / model_id / f"{model_id}.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for model_id: {model_id}")

        return str(model_path)

    def list_models(self) -> List[str]:
        """
        List all model IDs in storage.

        Returns:
            List of model IDs
        """
        try:
            model_ids = []
            for item in self.base_dir.iterdir():
                if item.is_dir() and (item / f"{item.name}.pth").exists():
                    model_ids.append(item.name)
            return model_ids
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def model_exists(self, model_id: str) -> bool:
        """
        Check if a model exists in storage.

        Args:
            model_id: Unique identifier for the model

        Returns:
            True if model exists, False otherwise
        """
        model_path = self.base_dir / model_id / f"{model_id}.pth"
        return model_path.exists()

    def cleanup_model_files(self, model_id: str) -> None:
        """
        Clean up all files for a specific model.

        Args:
            model_id: Unique identifier for the model
        """
        try:
            model_dir = self.base_dir / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info(f"Cleaned up model files for {model_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup model files for {model_id}: {e}")

    def get_model_size(self, model_id: str) -> int:
        """
        Get the size of the model file in bytes.

        Args:
            model_id: Unique identifier for the model

        Returns:
            Size of the model file in bytes
        """
        model_path = self.base_dir / model_id / f"{model_id}.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for model_id: {model_id}")

        return model_path.stat().st_size

    def update_metadata(self, model_id: str, updates: Dict[str, Any]) -> None:
        """
        Update model metadata.

        Args:
            model_id: Unique identifier for the model
            updates: Dictionary of metadata updates
        """
        try:
            metadata = self.load_model_metadata(model_id)
            metadata.update(updates)
            metadata["updated_at"] = datetime.utcnow().isoformat()

            metadata_path = self.base_dir / model_id / f"{model_id}.metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Updated metadata for model {model_id}")
        except Exception as e:
            logger.error(f"Failed to update metadata for {model_id}: {e}")
            raise ValueError(f"Failed to update metadata: {str(e)}")
