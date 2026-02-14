import os
import json
import tarfile
from io import BytesIO
from astropy.coordinates import SkyCoord
import django_filters
from django.conf import settings
from django.urls import reverse_lazy
from django.http import StreamingHttpResponse
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.http import JsonResponse
from django.shortcuts import render
from django_filters.rest_framework import DjangoFilterBackend
from django.contrib.auth.decorators import login_required, permission_required
from rest_framework import status
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from host.object_store import ObjectStore
from host.models import Aperture
from host.models import AperturePhotometry
from host.models import Cutout
from host.models import Filter
from host.models import SEDFittingResult
from host.models import TaskRegister
from host.models import Task
from host.models import Transient
from host.models import Host
from host.decorators import log_usage_metric
from host.host_utils import export_transient_info
from host.host_utils import delete_transient
from api.serializers import TransientSerializer
from api.serializers import ApertureSerializer
from api.serializers import CutoutSerializer
from api.serializers import FilterSerializer
from api.serializers import AperturePhotometrySerializer
from api.serializers import SEDFittingResultSerializer
from api.serializers import TaskRegisterSerializer
from api.serializers import TaskSerializer
from api.serializers import HostSerializer
from api.datamodel import unpack_component_groups
from api.datamodel import serialize_blast_science_data
from api.components import data_model_components

from host.log import get_logger
logger = get_logger(__name__)


############################################################
# Filter Sets
class TransientFilter(django_filters.FilterSet):
    redshift_lte = django_filters.NumberFilter(
        field_name="redshift", lookup_expr="lte")
    redshift_gte = django_filters.NumberFilter(
        field_name="redshift", lookup_expr="gte")
    host_redshift_lte = django_filters.NumberFilter(
        field_name="host__redshift", lookup_expr="lte")
    host_redshift_gte = django_filters.NumberFilter(
        field_name="host__redshift", lookup_expr="gte")
    host_photometric_redshift_lte = django_filters.NumberFilter(
        field_name="host__photometric_redshift", lookup_expr="lte")
    host_photometric_redshift_gte = django_filters.NumberFilter(
        field_name="host__photometric_redshift", lookup_expr="gte")

    class Meta:
        model = Transient
        fields = ("name",)


class HostFilter(django_filters.FilterSet):
    redshift_lte = django_filters.NumberFilter(
        field_name="redshift", lookup_expr="lte")
    redshift_gte = django_filters.NumberFilter(
        field_name="redshift", lookup_expr="gte")
    photometric_redshift_lte = django_filters.NumberFilter(
        field_name="photometric_redshift", lookup_expr="lte"
    )
    photometric_redshift_gte = django_filters.NumberFilter(
        field_name="photometric_redshift", lookup_expr="gte"
    )

    class Meta:
        model = Host
        fields = ("name",)


class ApertureFilter(django_filters.FilterSet):
    transient = django_filters.Filter(field_name="transient__name")

    class Meta:
        model = Aperture
        fields = ()


class TaskRegisterFilter(django_filters.FilterSet):
    transient = django_filters.Filter(field_name="transient__name")
    status = django_filters.Filter(field_name="status__message")
    task = django_filters.Filter(field_name="task__name")

    class Meta:
        model = TaskRegister
        fields = ()


class FilterFilter(django_filters.FilterSet):
    class Meta:
        model = Filter
        fields = ("name",)


class CutoutFilter(django_filters.FilterSet):
    filter = django_filters.Filter(field_name="filter__name")
    transient = django_filters.Filter(field_name="transient__name")

    class Meta:
        model = Cutout
        fields = ("name",)


class AperturePhotometryFilter(django_filters.FilterSet):
    filter = django_filters.Filter(field_name="filter__name")
    transient = django_filters.Filter(field_name="transient__name")

    class Meta:
        model = AperturePhotometry
        fields = ()


class SEDFittingResultFilter(django_filters.FilterSet):
    transient = django_filters.Filter(field_name="transient__name")
    aperture_type = django_filters.Filter(field_name="aperture__type")

    class Meta:
        model = SEDFittingResult
        fields = ()


############################################################
# ViewSets
class TransientViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Transient.objects.all()
    serializer_class = TransientSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = TransientFilter


class ApertureViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Aperture.objects.all()
    serializer_class = ApertureSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = ApertureFilter


class CutoutViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Cutout.objects.all()
    serializer_class = CutoutSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = CutoutFilter


class FilterViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Filter.objects.all()
    serializer_class = FilterSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = FilterFilter


class AperturePhotometryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = AperturePhotometry.objects.all()
    serializer_class = AperturePhotometrySerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = AperturePhotometryFilter


class SEDFittingResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = SEDFittingResult.objects.all()
    serializer_class = SEDFittingResultSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = SEDFittingResultFilter


class TaskRegisterViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = TaskRegister.objects.all()
    serializer_class = TaskRegisterSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = TaskRegisterFilter


class TaskViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer


class HostViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Host.objects.all()
    serializer_class = HostSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = HostFilter


def transient_exists(transient_name: str) -> bool:
    """
    Checks if a transient exists in the database.

    Parameters:
        transient_name (str): transient_name.
    Returns:
        exisit (bool): True if the transient exists false otherwise.
    """
    try:
        Transient.objects.get(name__exact=transient_name)
        exists = True
    except Transient.DoesNotExist:
        exists = False
    return exists


def ra_dec_valid(ra: str, dec: str) -> bool:
    """
    Checks if a given ra and dec coordinate is valid

    Parameters:
        ra (str): Right
    """
    try:
        ra, dec = float(ra), float(dec)
        SkyCoord(ra=ra, dec=dec, unit="deg")
        valid = True
    except Exception as err:
        logger.warning(f'Coordinates {(ra, dec)} are invalid: {err}')
        valid = False
    return valid


@api_view(["GET"])
@log_usage_metric()
def get_transient_science_payload(request, transient_name):
    if not transient_exists(transient_name):
        return Response(
            {"message": f"{transient_name} not in database"},
            status=status.HTTP_404_NOT_FOUND,
        )

    component_groups = [
        component_group(transient_name) for component_group in data_model_components
    ]
    components = unpack_component_groups(component_groups)
    data = serialize_blast_science_data(components)

    return Response(data, status=status.HTTP_200_OK)


@api_view(["POST"])
def post_transient(request, transient_name, transient_ra, transient_dec):
    if transient_exists(transient_name):
        return Response(
            {"message": f"{transient_name} already in database"},
            status=status.HTTP_409_CONFLICT,
        )

    if not ra_dec_valid(transient_ra, transient_dec):
        return Response(
            {"message": f"bad ra and dec: ra={transient_ra}, dec={transient_dec}"},
            status.HTTP_400_BAD_REQUEST,
        )

    data_string = (
        f"{transient_name}: ra = {float(transient_ra)}, dec= {float(transient_dec)}"
    )
    Transient.objects.create(
        name=transient_name,
        ra_deg=float(transient_ra),
        dec_deg=float(transient_dec),
        tns_id=1,
    )
    return Response(
        {"message": f"transient successfully posted: {data_string}"},
        status=status.HTTP_201_CREATED,
    )


# TODO: Secure this endpoint with Django REST Framework permission_classes
# @api_view(["PUT"])
# @permission_classes([IsAuthenticated])
# def launch_workflow(request, transient_name):
#     print(f'Launching transient workflow for "{transient_name}"...')
#     result = transient_workflow.delay(transient_name)
#     return Response({'message': f'Launched workflow for "{transient_name}": {result.task_id}'})


@log_usage_metric()
def export_transient_view(request=None, transient_name='', all=''):
    transient_info = export_transient_info(transient_name)
    if not transient_info:
        return render(request, "transient_404.html", status=404)
    logger.debug(f'''Exported transient dataset object:\n{json.dumps(transient_info, indent=2)}''')
    if not all:
        logger.info(f'Exporting only database objects for "{transient_name}", no data files.')
        return JsonResponse(transient_info)
    else:
        logger.info(f'Exporting all data for "{transient_name}", including files.')
        s3 = ObjectStore()
        tar_bytes_io = BytesIO()
        # Generate in-memory compressed archive file object of all data to stream
        with tarfile.open(fileobj=tar_bytes_io, mode="w:gz") as tar_fp:
            # Add transient dataset document to archive
            transient_info_fileobj = BytesIO(bytes(json.dumps(transient_info), 'utf-8'))
            transient_info_fileobj.seek(0)
            tarinfo = tarfile.TarInfo(name=f'{transient_name}.json')
            tarinfo.size = transient_info_fileobj.getbuffer().nbytes
            tar_fp.addfile(tarinfo, fileobj=transient_info_fileobj)
            # Download cutout FITS image files into memory
            for cutout in transient_info['cutouts']:
                canonical_path = cutout['fields']['fits']
                if not canonical_path:
                    continue
                object_key = os.path.join(settings.S3_BASE_PATH, canonical_path.strip('/'))
                cutout_fileobj = BytesIO(s3.get_object(path=object_key))
                # This assumes that the canonical paths for each cutout file are unique
                tarinfo = tarfile.TarInfo(
                    name=canonical_path.replace(os.path.join(settings.CUTOUT_ROOT, transient_name), 'cutouts'))
                tarinfo.size = cutout_fileobj.getbuffer().nbytes
                tar_fp.addfile(tarinfo, fileobj=cutout_fileobj)
            # Collect SED fit files into memory
            sedfittingresults = []
            for aperture in transient_info['apertures']:
                if aperture['sedfittingresults']:
                    sedfittingresults.extend(aperture['sedfittingresults'])
            for sedfittingresult in sedfittingresults:
                for sed_file in ['posterior', 'chains_file', 'percentiles_file', 'model_file']:
                    canonical_path = sedfittingresult['fields'][sed_file]
                    object_key = os.path.join(settings.S3_BASE_PATH, canonical_path.strip('/'))
                    sed_fileobj = BytesIO(s3.get_object(path=object_key))
                    # This assumes that the canonical paths for each sed file are unique
                    tarinfo = tarfile.TarInfo(
                        name=canonical_path.replace(os.path.join(settings.SED_OUTPUT_ROOT, transient_name), 'sed_data'))
                    tarinfo.size = sed_fileobj.getbuffer().nbytes
                    tar_fp.addfile(tarinfo, fileobj=sed_fileobj)
        tar_bytes_io.seek(0)
        response = StreamingHttpResponse(streaming_content=tar_bytes_io)
        response["Content-Disposition"] = f"attachment; filename={f'{transient_name}.tar.gz'}"
        return response


@login_required
@permission_required("host.delete_transient", raise_exception=True)
@log_usage_metric()
def delete_transient_view(request=None, transient_name=''):
    # Acquire the transient object or return 404 not found
    try:
        transient = Transient.objects.get(name__exact=transient_name)
    except Transient.DoesNotExist:
        return render(request, "transient_404.html", status=404)
    err_msg = delete_transient(transient=transient)
    if err_msg:
        return HttpResponse(status=500, content=err_msg)
    else:
        return HttpResponseRedirect(reverse_lazy("transient_list"))
