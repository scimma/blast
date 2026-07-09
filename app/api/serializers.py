from host import models
from rest_framework import serializers
from django.urls import reverse

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
            "added_by"
            # "host",
        ]

    aliases = serializers.SerializerMethodField()

    def get_aliases(self, obj):
        aliases = models.Alias.objects.filter(transient=obj)
        return [alias.alias for alias in aliases]


class HostSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = models.Host
        depth = 1
        fields = [
            "name",
            "ra_deg",
            "dec_deg",
            "redshift",
            "milkyway_dust_reddening",
            "object_id",
            "catalog_name",
            "catalog_release",
            "aliases",
        ]

    aliases = serializers.SerializerMethodField()

    def get_aliases(self, obj):
        aliases = models.Alias.objects.filter(host=obj)
        return [alias.alias for alias in aliases]


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


class AliasSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Alias
        fields = ["alias", "transient", "host"]

    transient = serializers.SerializerMethodField()
    host = serializers.SerializerMethodField()

    def get_transient(self, obj):
        return obj.transient.name if obj.transient else None

    def get_host(self, obj):
        return obj.host.name if obj.host else None


class SEDFittingResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.SEDFittingResult
        depth = 1
        exclude = ["log_tau_16", "log_tau_50", "log_tau_84", "posterior"]


class CutoutSerializer(serializers.ModelSerializer):
    cutout_file = serializers.SerializerMethodField()
    class Meta:
        model = models.Cutout
        depth = 1
        exclude = ["fits"]

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret['cutout_file'] = self.get_cutout_file(instance)
        return ret
    
    def get_cutout_file(self, obj):
        request = self.context["request"]
        if not obj.fits:
            return None
        
        return request.build_absolute_uri(
            reverse(
                "cutout-download",
                kwargs={"pk":obj.pk,}
            )
        )


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
