"""
This module contains some functions to get the object keys for certain components
"""

from .models import Cutout
from .models import SEDFittingResult
from django.db import models
from django.conf import settings

def get_cutout_file_object_key_from_cutout(cutout: Cutout):
    """
    Convert cutout object to an S3 object key
    """
    version = cutout.software_version if cutout.software_version else "0.0.0"
    workflow = cutout.workflow if cutout.workflow else "workflow_default"
    return f"""{cutout.transient.name}/v{version}/{workflow}/cutout_cdn/{cutout.filter.survey.name}/{cutout.filter}.fits"""


def get_cutout_file_local_path_from_cutout(cutout: Cutout, data_root = settings.INPUT_ROOT):
    """
    Convert cutout object to an S3 object key
    """
    version = cutout.software_version if cutout.software_version else "0.0.0"
    workflow = cutout.workflow if cutout.workflow else "workflow_default"
    return f"""{data_root}/{cutout.transient.name}/v{version}/{workflow}/cutout_cdn/{cutout.filter.survey.name}/{cutout.filter}.fits"""


def get_versions_sorted_transient(transient: str, model: models.Model):
    """
    Get the latest version of data for a transient
    """
    versions = model.objects.filter(transient__name__exact = transient).values("software_version").distinct()
    return versions
    

def get_sed_posterior_file_local_path_from_sed_fit_res(sed: SEDFittingResult, data_root = settings.INPUT_ROOT):
    version = sed.software_version if sed.software_version else "0.0.0"    
    workflow = sed.workflow if sed.workflow else "workflow_default"
    return f"""{data_root}/{sed.transient.name}/v{version}/{workflow}/sed_output/{sed.transient.name}_{sed.aperture.type}.h5"""


def get_sed_posterior_file_object_key_from_sed_fit_res(sed: SEDFittingResult):
    version = sed.software_version if sed.software_version else "0.0.0"    
    workflow = sed.workflow if sed.workflow else "workflow_default"
    return f"""{sed.transient.name}/v{version}/{workflow}/sed_output/{sed.transient.name}_{sed.aperture.type}.h5"""


def get_sed_chains_file_object_key_from_sed_fit_res(sed: SEDFittingResult):
    version = sed.software_version if sed.software_version else "0.0.0"    
    workflow = sed.workflow if sed.workflow else "workflow_default"
    return f"""{sed.transient.name}/v{version}/{workflow}/sed_output/{sed.transient.name}_{sed.aperture.type}_chain.npz"""


def get_sed_percentiles_file_object_key_from_sed_fit_res(sed: SEDFittingResult):
    version = sed.software_version if sed.software_version else "0.0.0"    
    workflow = sed.workflow if sed.workflow else "workflow_default"
    return f"""{sed.transient.name}/v{version}/{workflow}/sed_output/{sed.transient.name}_{sed.aperture.type}_perc.npz"""


def get_sed_model_file_object_key_from_sed_fit_res(sed: SEDFittingResult):
    version = sed.software_version if sed.software_version else "0.0.0"    
    workflow = sed.workflow if sed.workflow else "workflow_default"
    return f"""{sed.transient.name}/v{version}/{workflow}/sed_output/{sed.transient.name}_{sed.aperture.type}_modeldata.npz"""