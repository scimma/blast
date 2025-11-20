from abc import ABC, abstractmethod
from typing import Optional, List
from astrodash.domain.models.user_model import UserModel
from astrodash.core.exceptions import ModelNotFoundException, ModelValidationException

class ModelRepository(ABC):
    """
    Abstract repository interface for user-uploaded models.
    Follows the repository pattern for decoupling domain and infrastructure.

    All methods should raise ModelNotFoundException when a model is not found,
    and ModelValidationException when validation fails.
    """

    @abstractmethod
    def save(self, model: UserModel) -> UserModel:
        """
        Save a user model to persistent storage.

        Args:
            model: The UserModel to save

        Returns:
            The saved UserModel with updated fields

        Raises:
            ModelValidationException: If the model is invalid
        """
        pass

    @abstractmethod
    def get_by_id(self, model_id: str) -> Optional[UserModel]:
        """
        Retrieve a user model by its unique ID.

        Args:
            model_id: The unique identifier of the model

        Returns:
            The UserModel if found, None otherwise
        """
        pass

    @abstractmethod
    def list_all(self) -> List[UserModel]:
        """
        List all user models in storage.

        Returns:
            List of all UserModel instances
        """
        pass

    @abstractmethod
    def delete(self, model_id: str) -> None:
        """
        Delete a user model by its unique ID.

        Args:
            model_id: The unique identifier of the model to delete

        Raises:
            ModelNotFoundException: If the model is not found
        """
        pass

    @abstractmethod
    def get_by_owner(self, owner: str) -> List[UserModel]:
        """
        List all user models owned by a specific user.

        Args:
            owner: The owner identifier

        Returns:
            List of UserModel instances owned by the specified user
        """
        pass
