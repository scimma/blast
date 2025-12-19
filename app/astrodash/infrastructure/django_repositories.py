"""Django ORM backed repositories used by the AstroDash adapters."""
from __future__ import annotations

import uuid
from typing import Optional, List

from astrodash.domain.models.spectrum import Spectrum
from astrodash.domain.models.user_model import UserModel
from astrodash.domain.repositories.spectrum_repository import SpectrumRepository
from astrodash.domain.repositories.model_repository import ModelRepository
from astrodash.models import SpectrumRecord, UserModelRecord


class DjangoSpectrumRepository(SpectrumRepository):
    """Persist spectra using the Blast Django database."""

    def save(self, spectrum: Spectrum) -> Spectrum:
        if spectrum.id is None:
            spectrum.id = str(uuid.uuid4())
        defaults = {
            "osc_ref": spectrum.osc_ref,
            "file_name": spectrum.file_name,
            "x": spectrum.x,
            "y": spectrum.y,
            "redshift": spectrum.redshift,
            "meta": spectrum.meta or {},
        }
        SpectrumRecord.objects.update_or_create(id=spectrum.id, defaults=defaults)
        return spectrum

    def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        try:
            record = SpectrumRecord.objects.get(id=spectrum_id)
        except SpectrumRecord.DoesNotExist:
            return None
        return Spectrum(
            id=str(record.id),
            osc_ref=record.osc_ref,
            file_name=record.file_name,
            x=record.x,
            y=record.y,
            redshift=record.redshift,
            meta=record.meta or {},
        )

    def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        if not osc_ref:
            return None
        try:
            record = SpectrumRecord.objects.filter(osc_ref=osc_ref).latest("created_at")
        except SpectrumRecord.DoesNotExist:
            return None
        return Spectrum(
            id=str(record.id),
            osc_ref=record.osc_ref,
            file_name=record.file_name,
            x=record.x,
            y=record.y,
            redshift=record.redshift,
            meta=record.meta or {},
        )

    def get_from_file(self, file):  # pragma: no cover - delegated to FileSpectrumRepository
        raise NotImplementedError("Django repository does not parse files")


class DjangoModelRepository(ModelRepository):
    """Persist user uploaded models using Django ORM."""

    def save(self, model: UserModel) -> UserModel:
        defaults = {
            "name": model.name,
            "description": model.description,
            "owner": model.owner,
            "model_path": model.model_path,
            "class_mapping_path": model.class_mapping_path,
            "input_shape_path": model.input_shape_path,
            "meta": model.meta or {},
        }
        if not model.id:
            model.id = str(uuid.uuid4())
            UserModelRecord.objects.create(id=model.id, **defaults)
        else:
            UserModelRecord.objects.update_or_create(id=model.id, defaults=defaults)
        return model

    def get_by_id(self, model_id: str) -> Optional[UserModel]:
        try:
            record = UserModelRecord.objects.get(id=model_id)
        except UserModelRecord.DoesNotExist:
            return None
        return self._to_domain(record)

    def list_all(self) -> List[UserModel]:
        return [self._to_domain(obj) for obj in UserModelRecord.objects.all()]

    def delete(self, model_id: str) -> None:
        UserModelRecord.objects.filter(id=model_id).delete()

    def get_by_owner(self, owner: str) -> List[UserModel]:
        return [self._to_domain(obj) for obj in UserModelRecord.objects.filter(owner=owner)]

    @staticmethod
    def _to_domain(record: UserModelRecord) -> UserModel:
        return UserModel(
            id=str(record.id),
            name=record.name,
            description=record.description,
            owner=record.owner,
            model_path=record.model_path,
            class_mapping_path=record.class_mapping_path,
            input_shape_path=record.input_shape_path,
            meta=record.meta or {},
            created_at=record.created_at,
        )
