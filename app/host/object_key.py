"""
This module contains some functions to get the object keys for certain components
"""

from .models import Cutout
from .models import SEDFittingResult

def get_cutout_file_object_key_from_cutout(cutout: Cutout):
    version = cutout.software_version if cutout.software_version else "0.0.0"
    workflow = cutout.workflow if cutout.workflow else "workflow_default"
    return f"""{cutout.transient.name}/v{version}/{workflow}/cutout_cdn/{cutout.filter.survey.name}/{cutout.filter}.fits"""

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