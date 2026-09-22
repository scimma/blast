from host import models
from datetime import datetime, timezone
from rest_framework import serializers
from drf_spectacular.utils import extend_schema_field
from django.urls import reverse
from django.conf import settings

TRANSIENT_EXCLUDED_FIELDS = [
    "tns_id",
    "tns_prefix",
    "tasks_initialized",
    "photometric_class",
    "processing_status",
    "added_by"
]


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
        exclude = TRANSIENT_EXCLUDED_FIELDS

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
    transient = serializers.SerializerMethodField()
    host = serializers.SerializerMethodField()

    class Meta:
        model = models.Alias
        fields = ["alias", "transient", "host"]
        depth = 1

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
        exclude = []


class DatasetSerializer(serializers.Serializer):
    metadata = serializers.SerializerMethodField()
    transient = serializers.SerializerMethodField()
    host = serializers.SerializerMethodField()
    surveys = serializers.SerializerMethodField()
    filters = serializers.SerializerMethodField()
    cutouts = serializers.SerializerMethodField()
    apertures = serializers.SerializerMethodField()
    host_spectra = serializers.SerializerMethodField()
    workflow_tasks = serializers.SerializerMethodField()

    class DatasetMetadataSerializer(serializers.Serializer):
        app_version = serializers.SerializerMethodField()
        export_time = serializers.SerializerMethodField()
        dataset_version = serializers.SerializerMethodField()

        @extend_schema_field(serializers.CharField())
        def get_app_version(self, obj):
            return f'v{settings.APP_VERSION}'

        @extend_schema_field(serializers.DateTimeField())
        def get_export_time(self, obj):
            return datetime.now(timezone.utc).isoformat()

        @extend_schema_field(serializers.IntegerField())
        def get_dataset_version(self, obj):
            dataset_version = models.DatasetRevision.objects.filter(transient=obj)
            if dataset_version:
                dataset_version = dataset_version[0].revision
            else:
                dataset_version = 0
            return dataset_version

    class DatasetTransientSerializer(serializers.ModelSerializer):
        class Meta:
            model = models.Transient
            depth = 0
            exclude = TRANSIENT_EXCLUDED_FIELDS

    class DatasetHostSerializer(serializers.ModelSerializer):
        class Meta:
            model = models.Host
            depth = 0
            exclude = []

        aliases = serializers.SerializerMethodField()

        @extend_schema_field(serializers.ListField(child=serializers.CharField()))
        def get_aliases(self, obj):
            aliases = models.Alias.objects.filter(host=obj)
            return [alias.alias for alias in aliases]

    class DatasetCutoutSerializer(serializers.ModelSerializer):
        class Meta:
            model = models.Cutout
            depth = 0
            exclude = ["id"]

    class DatasetApertureSerializer(serializers.Serializer):

        aperture = serializers.SerializerMethodField()
        sedfittingresults = serializers.SerializerMethodField()
        aperturephotometry = serializers.SerializerMethodField()
        starformationhistoryresult = serializers.SerializerMethodField()

        class ApertureSerializerFlat(serializers.ModelSerializer):
            class Meta:
                model = models.Aperture
                depth = 0
                exclude = []

        class DatasetAperturePhotometrySerializer(serializers.ModelSerializer):
            class Meta:
                model = models.AperturePhotometry
                depth = 0
                exclude = []

        class DatasetStarFormationHistoryResultSerializer(serializers.ModelSerializer):
            class Meta:
                model = models.StarFormationHistoryResult
                depth = 0
                exclude = []

        class DatasetSEDFittingResultSerializer(serializers.ModelSerializer):
            class Meta:
                model = models.SEDFittingResult
                depth = 0
                exclude = []

        @extend_schema_field(ApertureSerializerFlat)
        def get_aperture(self, obj):
            return self.ApertureSerializerFlat(obj).data

        @extend_schema_field(serializers.ListField(child=DatasetSEDFittingResultSerializer()))
        def get_sedfittingresults(self, obj):
            return [self.DatasetSEDFittingResultSerializer(record).data
                    for record in models.SEDFittingResult.objects.filter(aperture=obj)]

        @extend_schema_field(serializers.ListField(child=DatasetAperturePhotometrySerializer()))
        def get_aperturephotometry(self, obj):
            return [self.DatasetAperturePhotometrySerializer(record).data
                    for record in models.AperturePhotometry.objects.filter(aperture=obj)]

        @extend_schema_field(serializers.ListField(child=DatasetStarFormationHistoryResultSerializer()))
        def get_starformationhistoryresult(self, obj):
            return [self.DatasetStarFormationHistoryResultSerializer(record).data
                    for record in models.StarFormationHistoryResult.objects.filter(aperture=obj)]

    class DatasetHostSpectrumSerializer(serializers.ModelSerializer):
        class Meta:
            model = models.HostSpectrum
            depth = 0
            fields = "__all__"

    class DatasetTaskRegisterSerializer(serializers.ModelSerializer):

        class DatasetTaskStatusSerializer(serializers.ModelSerializer):
            class Meta:
                model = models.Status
                fields = ["message", "type"]

        task = TaskSerializer(read_only=True)
        status = DatasetTaskStatusSerializer(read_only=True)

        class Meta:
            model = models.TaskRegister
            depth = 0
            fields = "__all__"

    @extend_schema_field(DatasetMetadataSerializer)
    def get_metadata(self, transient_obj):
        return self.DatasetMetadataSerializer(transient_obj).data

    @extend_schema_field(DatasetTransientSerializer)
    def get_transient(self, transient_obj):
        return self.DatasetTransientSerializer(transient_obj).data

    @extend_schema_field(DatasetHostSerializer)
    def get_host(self, transient_obj):
        host_data = self.DatasetHostSerializer(transient_obj.host).data if transient_obj.host else None
        return host_data

    @extend_schema_field(serializers.ListField(child=SurveySerializer()))
    def get_surveys(self, transient_obj):
        return [SurveySerializer(record).data for record in models.Survey.objects.all()]

    @extend_schema_field(serializers.ListField(child=FilterSerializer()))
    def get_filters(self, transient_obj):
        return [FilterSerializer(record).data for record in models.Filter.objects.all()]

    @extend_schema_field(serializers.ListField(child=DatasetCutoutSerializer()))
    def get_cutouts(self, transient_obj):
        return [self.DatasetCutoutSerializer(record).data
                for record in models.Cutout.objects.filter(transient=transient_obj)]

    @extend_schema_field(serializers.ListField(child=DatasetApertureSerializer()))
    def get_apertures(self, transient_obj):
        return [self.DatasetApertureSerializer(record).data
                for record in models.Aperture.objects.filter(transient=transient_obj)]

    @extend_schema_field(serializers.ListField(child=DatasetHostSpectrumSerializer()))
    def get_host_spectra(self, transient_obj):
        return [self.DatasetHostSpectrumSerializer(record).data
                for record in models.HostSpectrum.objects.filter(host=transient_obj.host)]

    @extend_schema_field(serializers.ListField(child=DatasetTaskRegisterSerializer()))
    def get_workflow_tasks(self, transient_obj):
        return [self.DatasetTaskRegisterSerializer(record).data
                for record in models.TaskRegister.objects.filter(transient=transient_obj)]
