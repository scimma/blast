from django.db import models
from django.utils import timezone
import uuid


class SpectrumRecord(models.Model):
    """Persisted snapshot of a processed spectrum."""

    id = models.CharField(primary_key=True, max_length=64, default=lambda: str(uuid.uuid4()), editable=False)
    osc_ref = models.CharField(max_length=128, blank=True, null=True, db_index=True)
    file_name = models.CharField(max_length=255, blank=True, null=True)
    x = models.JSONField()
    y = models.JSONField()
    redshift = models.FloatField(blank=True, null=True)
    meta = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "astrodash_spectra"
        ordering = ["-created_at"]


class UserModelRecord(models.Model):
    """Metadata for a user-uploaded ML model stored on disk."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    owner = models.CharField(max_length=255, blank=True, null=True, db_index=True)
    model_path = models.CharField(max_length=512)
    class_mapping_path = models.CharField(max_length=512)
    input_shape_path = models.CharField(max_length=512)
    meta = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "astrodash_user_models"
        ordering = ["-created_at"]
