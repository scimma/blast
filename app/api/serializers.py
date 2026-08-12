from host import models
from rest_framework import serializers
from drf_spectacular.utils import extend_schema_field
from django.urls import reverse


class StatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Status
        fields = ["id", "message", "type"]


class CutoutField(serializers.RelatedField):
    def to_representation(self, value):
        return value.filter.name


class FilterSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Filter
        depth = 0
        fields = "__all__"


class SurveySerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Survey
        depth = 1
        fields = "__all__"


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


class TransientSerializer(serializers.ModelSerializer):
    host = HostSerializer(read_only=True)

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
        ]

    aliases = serializers.SerializerMethodField()

    @extend_schema_field(serializers.ListField(child=serializers.CharField()))
    def get_aliases(self, obj):
        aliases = models.Alias.objects.filter(transient=obj)
        return [alias.alias for alias in aliases]


class CutoutSerializer(serializers.ModelSerializer):
    filter = FilterSerializer(read_only=True)
    transient = TransientSerializer(read_only=True)
    cutout_file = serializers.SerializerMethodField()

    class Meta:
        model = models.Cutout
        depth = 1
        exclude = ["fits"]

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret['cutout_file'] = self.get_cutout_file(instance)
        return ret

    def get_cutout_file(self, obj) -> str:
        request = self.context["request"]
        if not obj.fits:
            return None

        return request.build_absolute_uri(
            reverse(
                "cutout-download",
                kwargs={"pk": obj.pk, }
            )
        )


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


class StarFormationHistoryResultSerializer(serializers.ModelSerializer):

    aperture = ApertureSerializer(read_only=True)
    transient = TransientSerializer(read_only=True)

    class Meta:
        model = models.StarFormationHistoryResult
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


class download_url_field(serializers.SerializerMethodField):
    # Like SerializerMethodField, but also passes self.field_name to the method, avoid hardcoding per-field wrappers.
    def to_representation(self, obj):
        method = getattr(self.parent, self.method_name)
        return method(obj, self.field_name)


class SEDFittingResultSerializer(serializers.ModelSerializer):
    transient = TransientSerializer(read_only=True)
    aperture = ApertureSerializer(read_only=True)

    chains_file = download_url_field(method_name='get_download_file')
    model_file = download_url_field(method_name='get_download_file')
    percentiles_file = download_url_field(method_name='get_download_file')

    class Meta:
        model = models.SEDFittingResult
        depth = 1
        exclude = ["log_tau_16", "log_tau_50", "log_tau_84", "posterior"]

    def to_representation(self, instance):
        # Hardcode download URL
        ret = super().to_representation(instance)
        ret['chains_file'] = self.get_download_file(instance, "chains")
        ret['model_file'] = self.get_download_file(instance, "model")
        ret['percentiles_file'] = self.get_download_file(instance, "percentiles")
        return ret

    def get_download_file(self, obj, file_type) -> str:
        request = self.context["request"]
        return request.build_absolute_uri(
            reverse("sedfittingresult-download", kwargs={"pk": obj.pk, "file_type": file_type})
        )


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


class MetadataSerializer(serializers.Serializer):
    app_version = serializers.CharField()
    export_time = serializers.DateTimeField()


class TransientEntitySerializer(serializers.Serializer):
    model = serializers.CharField()
    pk = serializers.IntegerField()
    fields = TransientSerializer()


class HostEntitySerializer(serializers.Serializer):
    model = serializers.CharField()
    pk = serializers.IntegerField()
    fields = HostSerializer()


class SEDFittingResultEntitySerializer(serializers.Serializer):
    class SEDFittingResultSerializerWithoutId(serializers.ModelSerializer):
        class Meta:
            model = models.SEDFittingResult
            depth = 0
            exclude = ["id"]
    model = serializers.CharField()
    pk = serializers.IntegerField()
    fields = SEDFittingResultSerializerWithoutId()


class AperturePhotometryEntitySerializer(serializers.Serializer):
    class AperturePhotometrySerializerWithoutId(serializers.ModelSerializer):
        class Meta:
            model = models.AperturePhotometry
            depth = 0
            exclude = ["id"]
    model = serializers.CharField()
    pk = serializers.IntegerField()
    fields = AperturePhotometrySerializerWithoutId()


class StarFormationHistoryResultEntitySerializer(serializers.Serializer):
    class StarFormationHistoryResultSerializerWithoutId(serializers.ModelSerializer):
        class Meta:
            model = models.StarFormationHistoryResult
            depth = 0
            exclude = ["id"]
    model = serializers.CharField()
    pk = serializers.IntegerField()
    fields = StarFormationHistoryResultSerializerWithoutId()


class ApertureEntitySerializer(serializers.Serializer):
    class ApertureSerializerWithoutId(serializers.ModelSerializer):
        class Meta:
            model = models.Aperture
            depth = 0
            exclude = ["id"]
    model = serializers.CharField()
    pk = serializers.IntegerField()
    fields = ApertureSerializerWithoutId()
    sedfittingresults = serializers.ListField(child=SEDFittingResultEntitySerializer())
    aperturephotometry = serializers.ListField(child=AperturePhotometryEntitySerializer())
    starformationhistoryresult = serializers.ListField(child=StarFormationHistoryResultEntitySerializer())


class CutoutEntitySerializer(serializers.Serializer):
    class CutoutSerializerWithoutId(serializers.ModelSerializer):
        class Meta:
            model = models.Cutout
            depth = 0
            exclude = ["id"]
    model = serializers.CharField()
    pk = serializers.IntegerField()
    fields = CutoutSerializerWithoutId()


class FilterEntitySerializer(serializers.Serializer):
    model = serializers.CharField()
    pk = serializers.IntegerField()
    fields = FilterSerializer()


class SurveyEntitySerializer(serializers.Serializer):
    class SurveySerializerWithoutId(serializers.ModelSerializer):
        class Meta:
            model = models.Survey
            depth = 1
            exclude = ["id"]

    model = serializers.CharField()
    pk = serializers.IntegerField()
    fields = SurveySerializerWithoutId()


class WorkflowTaskSerializer(serializers.Serializer):
    class StatusSerializerWithoutId(serializers.ModelSerializer):
        class Meta:
            model = models.Status
            exclude = ["id"]

    task_name = serializers.CharField()
    status = StatusSerializerWithoutId()
    user_warning = serializers.BooleanField()
    last_modified = serializers.DateTimeField()
    last_processing_time_seconds = serializers.FloatField()


class TransientDatasetSerializer(serializers.Serializer):
    metadata = MetadataSerializer()
    transient = TransientEntitySerializer()
    host = HostEntitySerializer()

    apertures = serializers.ListField(child=ApertureEntitySerializer())
    cutouts = serializers.ListField(child=CutoutEntitySerializer())
    filters = serializers.ListField(child=FilterEntitySerializer())
    surveys = serializers.ListField(child=SurveyEntitySerializer())
    workflow_tasks = serializers.ListField(child=WorkflowTaskSerializer())
