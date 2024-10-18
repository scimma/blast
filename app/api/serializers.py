from dataclasses import dataclass

from host import models
from rest_framework import serializers
from datetime import datetime, timezone
from django.conf import settings


def get_api_metadata():
    return {
        'app_version': settings.APP_VERSION,
        'date_accessed': datetime.now(timezone.utc),
    }


class CutoutField(serializers.RelatedField):
    def to_representation(self, value):
        return value.filter.name


class ModelSerializerWithMetadata(serializers.ModelSerializer):
    metadata = serializers.SerializerMethodField('generate_metadata')

    def generate_metadata(self, obj):
        return get_api_metadata()


class TransientSerializer(ModelSerializerWithMetadata):

    class Meta:
        model = models.Transient
        depth = 1
        exclude = [
            "tns_id",
            "tns_prefix",
            "tasks_initialized",
            "photometric_class",
            "processing_status",
            "added_by",
        ]
        read_only_fields = ('metadata',)


class HostSerializer(serializers.HyperlinkedModelSerializer):
    metadata = serializers.SerializerMethodField('generate_metadata')

    def generate_metadata(self, obj):
        return get_api_metadata()

    class Meta:
        model = models.Host
        depth = 1
        fields = ["name", "ra_deg", "dec_deg", "redshift", "milkyway_dust_reddening", "metadata"]
        read_only_fields = ('metadata',)


class ApertureSerializer(ModelSerializerWithMetadata):
    class Meta:
        model = models.Aperture
        depth = 1
        fields = "__all__"
        read_only_fields = ('metadata',)


class AperturePhotometrySerializer(ModelSerializerWithMetadata):
    class Meta:
        model = models.AperturePhotometry
        depth = 1
        fields = "__all__"
        read_only_fields = ('metadata',)


class SEDFittingResultSerializer(ModelSerializerWithMetadata):
    class Meta:
        model = models.SEDFittingResult
        depth = 1
        exclude = ["log_tau_16", "log_tau_50", "log_tau_84", "posterior"]
        read_only_fields = ('metadata',)


class CutoutSerializer(ModelSerializerWithMetadata):
    class Meta:
        model = models.Cutout
        depth = 1
        exclude = ["fits"]
        read_only_fields = ('metadata',)


class FilterSerializer(ModelSerializerWithMetadata):
    class Meta:
        model = models.Filter
        depth = 1
        fields = [
            "name",
            "pixel_size_arcsec",
            "image_fwhm_arcsec",
            "wavelength_eff_angstrom",
            "ab_offset",
        ]
        read_only_fields = ('metadata',)


class TaskRegisterSerializer(ModelSerializerWithMetadata):
    class Meta:
        model = models.TaskRegister
        depth = 1
        fields = "__all__"
        read_only_fields = ('metadata',)


class TaskSerializer(ModelSerializerWithMetadata):
    class Meta:
        model = models.Task
        depth = 1
        fields = ["name"]
