"""
This module contains helper functions to serialize Blast data into a flat
science payload to be delivered by the Blast web API.
"""
import itertools
from dataclasses import dataclass
from typing import List

import django
from host import models
from rest_framework import serializers


@dataclass
class DataModelComponent:
    """
    A dataclass to store all information on how to
    serialise a Blast model into the Blast science payload.

    Attributes:
        prefix (str): What will be appended to the column name in the Blast
            science payload
        query (dict): A django query that will uniquely identify the data for
            the Blast science payload
        model (django.db.models.Model): The Blast model that the query will be
            passed to
        serializer (rest_framework.serializers.Serializer) The serializer
            associated with the model
    """

    prefix: str
    query: dict
    model: django.db.models.Model
    serializer: serializers.Serializer


def serialize_blast_science_data(datamodel: List[DataModelComponent]) -> dict:
    """
    Serializes all data associated with a transient into a flat structure.

    parameters:
        datamodel: (List[DataModelComponent]): datamodel to be serialized.
    returns:
        science_payload: Flat dictionary containing science data fields and
            values.
    """
    science_payload = {}
    for component in datamodel:
        prefix, serializer = component.prefix, component.serializer
        try:
            object = component.model.objects.get(**component.query)
            object_data = serializer(object).data
            object_dict = {prefix + name: value for name, value in object_data.items()}
        except:
            object_dict = {prefix + name: None for name in serializer().fields}

        science_payload = {**science_payload, **object_dict}
    return science_payload


def unpack_component_groups(
    component_groups: List[List[DataModelComponent]],
) -> List[DataModelComponent]:
    """
    Unpacks list of datamodel component groups into a flat structure

    parameters:
        component_groups (List[List[DataModelComponent]]): list of groups to be unpacked
    returns:
        flat_components (List[DataModelComponent]): flattened list of components
    """
    return list(itertools.chain(*component_groups))
