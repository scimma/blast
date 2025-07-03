import itertools

import django_filters
from astropy.coordinates import SkyCoord
from django_filters.rest_framework import DjangoFilterBackend
from host.models import *
from host.models import Transient
from rest_framework import generics
from rest_framework import status
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from . import datamodel
from . import serializers
from .components import data_model_components
from host.decorators import log_usage_metric


### Filter Sets ###

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


### Filter Sets ###
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


### ViewSets ###
class TransientViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Transient.objects.all()
    serializer_class = serializers.TransientSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = TransientFilter

class ApertureViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Aperture.objects.all()
    serializer_class = serializers.ApertureSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = ApertureFilter


class CutoutViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Cutout.objects.all()
    serializer_class = serializers.CutoutSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = CutoutFilter


class FilterViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Filter.objects.all()
    serializer_class = serializers.FilterSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = FilterFilter


class AperturePhotometryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = AperturePhotometry.objects.all()
    serializer_class = serializers.AperturePhotometrySerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = AperturePhotometryFilter


class SEDFittingResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = SEDFittingResult.objects.all()
    serializer_class = serializers.SEDFittingResultSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = SEDFittingResultFilter


class TaskRegisterViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = TaskRegister.objects.all()
    serializer_class = serializers.TaskRegisterSerializer
    filter_backends = (DjangoFilterBackend,)
    filterset_class = TaskRegisterFilter


class TaskViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Task.objects.all()
    serializer_class = serializers.TaskSerializer


class HostViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Host.objects.all()
    serializer_class = serializers.HostSerializer
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
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        valid = True
    except:
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
    components = datamodel.unpack_component_groups(component_groups)
    data = datamodel.serialize_blast_science_data(components)

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


# TO DO: Secure this endpoint with Django REST Framework permission_classes
# @api_view(["PUT"])
# @permission_classes([IsAuthenticated])
# def launch_workflow(request, transient_name):
#     print(f'Launching transient workflow for "{transient_name}"...')
#     result = transient_workflow.delay(transient_name)
#     return Response({'message': f'Launched workflow for "{transient_name}": {result.task_id}'}, status=status.HTTP_200_OK)
