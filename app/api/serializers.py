from dataclasses import dataclass

from host import models
from rest_framework import serializers


class CutoutField(serializers.RelatedField):
    def to_representation(self, value):
        return value.filter.name


class TransientSerializer(serializers.ModelSerializer):
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


class HostSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = models.Host
        depth = 1
        fields = ["name", "ra_deg", "dec_deg", "redshift", "milkyway_dust_reddening"]


class ApertureSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Aperture
        depth = 1
        fields = "__all__"

class AperturePhotometrySerializer(serializers.ModelSerializer):
    class Meta:
        model = models.AperturePhotometry
        depth = 1
        fields = "__all__"

class SEDFittingResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.SEDFittingResult
        depth = 1
        exclude = ["log_tau_16", "log_tau_50", "log_tau_84", "posterior"]


class CutoutSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Cutout
        depth = 1
        exclude = ["fits"]


class FilterSerializer(serializers.ModelSerializer):
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


class TaskRegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.TaskRegister
        depth = 1
        fields = "__all__"


class TaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Task
        depth = 1
        fields = ["name"]
