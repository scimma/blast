from host import models
from rest_framework import serializers
from drf_spectacular.utils import extend_schema_field


class StatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Status
        fields = ["id", "message", "type"]


class CutoutField(serializers.RelatedField):
    def to_representation(self, value):
        return value.filter.name


class TransientSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Transient
        exclude = [
            "tns_id",
            "tns_prefix",
            "tasks_initialized",
            "photometric_class",
            "processing_status",
            "added_by"
        ]

    aliases = serializers.SerializerMethodField()

    @extend_schema_field(serializers.ListField(child=serializers.CharField()))
    def get_aliases(self, obj):
        aliases = models.Alias.objects.filter(transient=obj)
        return [alias.alias for alias in aliases]


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


class CutoutSerializer(serializers.ModelSerializer):
    filter = FilterSerializer(read_only=True)
    transient = TransientSerializer(read_only=True)

    class Meta:
        model = models.Cutout
        depth = 1
        exclude = ["fits"]


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

    @extend_schema_field(serializers.ListField(child=serializers.CharField()))
    def get_aliases(self, obj):
        aliases = models.Alias.objects.filter(host=obj)
        return [alias.alias for alias in aliases]


class ApertureSerializer(serializers.ModelSerializer):
    cutout = CutoutSerializer(read_only=True)
    transient = TransientSerializer(read_only=True)

    class Meta:
        model = models.Aperture
        depth = 1
        fields = "__all__"


class AperturePhotometrySerializer(serializers.ModelSerializer):

    aperture = ApertureSerializer(read_only=True)
    filter = FilterSerializer(read_only=True)
    transient = TransientSerializer(read_only=True)

    class Meta:
        model = models.AperturePhotometry
        depth = 1
        fields = "__all__"


class AliasSerializer(serializers.ModelSerializer):
    transient = TransientSerializer(read_only=True)
    host = HostSerializer(read_only=True)

    class Meta:
        model = models.Alias
        fields = ["alias", "transient", "host"]

    transient = serializers.SerializerMethodField()
    host = serializers.SerializerMethodField()

    @extend_schema_field(serializers.CharField(allow_null=True))
    def get_transient(self, obj):
        return obj.transient.name if obj.transient else None

    @extend_schema_field(serializers.CharField(allow_null=True))
    def get_host(self, obj):
        return obj.host.name if obj.host else None


class SEDFittingResultSerializer(serializers.ModelSerializer):
    transient = TransientSerializer(read_only=True)
    aperture = ApertureSerializer(read_only=True)

    class Meta:
        model = models.SEDFittingResult
        depth = 1
        exclude = ["log_tau_16", "log_tau_50", "log_tau_84", "posterior"]


class TaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Task
        fields = ["name"]


class TaskRegisterSerializer(serializers.ModelSerializer):
    task = TaskSerializer(read_only=True)
    status = StatusSerializer(read_only=True)
    transient = TransientSerializer(read_only=True)

    class Meta:
        model = models.TaskRegister
        depth = 1
        fields = "__all__"
