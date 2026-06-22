from host import models
from rest_framework import serializers
from drf_spectacular.utils import extend_schema_field
from drf_spectacular.types import OpenApiTypes

##################
# Nested Serializers

class UserNestedSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.User
        fields = ["id", "username"]


class SurveyNestedSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Survey
        fields = ["id", "name"]


class TaskNestedSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Task
        fields = ["id", "name"]


class StatusNestedSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Status
        fields = ["id", "message", "type"]


class HostNestedSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Host
        fields = [
            "id", "name", "ra_deg", "dec_deg",
            "redshift", "redshift_err",
            "photometric_redshift", "photometric_redshift_err",
            "milkyway_dust_reddening", "catalog_name", "catalog_release",
        ]


# Depends on SurveyNestedSerializer
class FilterNestedSerializer(serializers.ModelSerializer):
    survey = SurveyNestedSerializer(read_only=True)

    class Meta:
        model = models.Filter
        fields = [
            "id", "name", "survey",
            "wavelength_eff_angstrom", "wavelength_min_angstrom", "wavelength_max_angstrom",
            "pixel_size_arcsec", "image_fwhm_arcsec",
            "vega_zero_point_jansky", "magnitude_zero_point", "ab_offset",
        ]

class TransientNestedSerializer(serializers.ModelSerializer):
    host = HostNestedSerializer(read_only=True)
    added_by = UserNestedSerializer(read_only=True)

    class Meta:
        model = models.Transient
        fields = [
            "id", "name", "ra_deg", "dec_deg",
            "tns_id", "tns_prefix", "public_timestamp",
            "redshift", "spectroscopic_class", "photometric_class",
            "milkyway_dust_reddening", "processing_status", "progress",
            "host", "added_by",
        ]

class CutoutNestedSerializer(serializers.ModelSerializer):
    filter = FilterNestedSerializer(read_only=True)
    transient = TransientNestedSerializer(read_only=True)

    class Meta:
        model = models.Cutout
        fields = ["id", "name", "filter", "transient", "message", "cropped"]


# Depends on CutoutNestedSerializer and TransientNestedSerializer
class ApertureNestedSerializer(serializers.ModelSerializer):
    cutout = CutoutNestedSerializer(read_only=True)
    transient = TransientNestedSerializer(read_only=True)

    class Meta:
        model = models.Aperture
        fields = [
            "id", "name", "ra_deg", "dec_deg",
            "orientation_deg", "semi_major_axis_arcsec", "semi_minor_axis_arcsec",
            "type", "cutout", "transient",
        ]

class StarFormationHistoryResultNestedSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.StarFormationHistoryResult
        fields = [
            "id",
            "logsfr_16", "logsfr_50", "logsfr_84",
            "logsfr_tmin", "logsfr_tmax",
        ]

class CutoutField(serializers.RelatedField):
    def to_representation(self, value):
        return value.filter.name


class TransientSerializer(serializers.ModelSerializer):

    host = HostNestedSerializer(read_only=True)
    class Meta:
        model = models.Transient
        exclude = [
            "tns_id",
            "tns_prefix",
            "tasks_initialized",
            "photometric_class",
            "processing_status",
            "added_by"
#            "host",
        ]

    aliases = serializers.SerializerMethodField()

    @extend_schema_field(serializers.ListField(child=serializers.CharField()))
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

    @extend_schema_field(serializers.ListField(child=serializers.CharField()))
    def get_aliases(self, obj):
        aliases = models.Alias.objects.filter(host=obj)
        return [alias.alias for alias in aliases]


class ApertureSerializer(serializers.ModelSerializer):
    cutout = CutoutNestedSerializer(read_only=True)
    transient = TransientNestedSerializer(read_only=True)

    class Meta:
        model = models.Aperture
        depth = 1
        fields = "__all__"


class AperturePhotometrySerializer(serializers.ModelSerializer):
    aperture = ApertureNestedSerializer(read_only=True)
    filter = FilterNestedSerializer(read_only=True)
    transient = TransientNestedSerializer(read_only=True)

    class Meta:
        model = models.AperturePhotometry
        depth = 1
        fields = "__all__"


class AliasSerializer(serializers.ModelSerializer):
    transient = TransientNestedSerializer(read_only=True)
    host = HostNestedSerializer(read_only=True)

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
    transient = TransientNestedSerializer(read_only=True)
    aperture = ApertureNestedSerializer(read_only=True)
    logsfh = StarFormationHistoryResultNestedSerializer(many=True, read_only=True)

    class Meta:
        model = models.SEDFittingResult
        depth = 1
        exclude = ["log_tau_16", "log_tau_50", "log_tau_84", "posterior"]


class CutoutSerializer(serializers.ModelSerializer):
    filter = FilterNestedSerializer(read_only=True)
    transient = TransientNestedSerializer(read_only=True)
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
    task = TaskNestedSerializer(read_only=True)
    status = StatusNestedSerializer(read_only=True)
    transient = TransientNestedSerializer(read_only=True)
    class Meta:
        model = models.TaskRegister
        depth = 1
        fields = "__all__"


class TaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Task
        fields = ["name"]
