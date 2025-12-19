from typing import Optional, List
from sqlalchemy.orm import Session
from astrodash.domain.repositories.model_repository import ModelRepository
from astrodash.domain.models.user_model import UserModel
from astrodash.infrastructure.database.models import UserModelDB
from astrodash.core.exceptions import ModelNotFoundException, ModelValidationException

class SQLAlchemyModelRepository(ModelRepository):
    """
    Concrete repository for user models using SQLAlchemy.
    Maps between UserModel domain model and UserModelDB ORM model.
    """
    def __init__(self, db: Session):
        self.db = db

    def save(self, model: UserModel) -> UserModel:
        # Check if model already exists
        existing_model = self.db.query(UserModelDB).filter(UserModelDB.id == model.id).first()

        if existing_model:
            # Update existing model
            existing_model.name = model.name
            existing_model.description = model.description
            existing_model.owner = model.owner
            existing_model.model_path = model.model_path
            existing_model.class_mapping_path = model.class_mapping_path
            existing_model.input_shape_path = model.input_shape_path
            existing_model.meta = model.meta
            db_model = existing_model
        else:
            # Create new model
            db_model = UserModelDB(
                id=model.id,
                name=model.name,
                description=model.description,
                owner=model.owner,
                model_path=model.model_path,
                class_mapping_path=model.class_mapping_path,
                input_shape_path=model.input_shape_path,
                created_at=model.created_at,
                meta=model.meta
            )
            self.db.add(db_model)

        self.db.commit()
        self.db.refresh(db_model)
        return self._to_domain(db_model)

    def get_by_id(self, model_id: str) -> Optional[UserModel]:
        db_model = self.db.query(UserModelDB).filter(UserModelDB.id == model_id).first()
        return self._to_domain(db_model) if db_model else None

    def list_all(self) -> List[UserModel]:
        db_models = self.db.query(UserModelDB).all()
        return [self._to_domain(m) for m in db_models]

    def delete(self, model_id: str) -> None:
        db_model = self.db.query(UserModelDB).filter(UserModelDB.id == model_id).first()
        if not db_model:
            raise ModelNotFoundException(model_id)

        self.db.delete(db_model)
        self.db.commit()

    def get_by_owner(self, owner: str) -> List[UserModel]:
        db_models = self.db.query(UserModelDB).filter(UserModelDB.owner == owner).all()
        return [self._to_domain(m) for m in db_models]

    def _to_domain(self, db_model: UserModelDB) -> UserModel:
        return UserModel(
            id=db_model.id,
            name=db_model.name,
            description=db_model.description,
            owner=db_model.owner,
            model_path=db_model.model_path,
            class_mapping_path=db_model.class_mapping_path,
            input_shape_path=db_model.input_shape_path,
            created_at=db_model.created_at,
            meta=db_model.meta
        )
