# Utils and wrappers for the prospector SED fitting code
import numpy as np

from .models import AperturePhotometry
from .models import Filter
from .photometric_calibration import jansky_to_maggies


def build_obs(transient, aperture_type):

    """
    This functions is required by prospector and should return
    a dictionary defined by
    https://prospect.readthedocs.io/en/latest/dataformat.html.

    """

    photometry = AperturePhotometry.objects.filter(
        transient=transient, aperture__type__exact=aperture_type
    )

    if not photometry.exists():
        raise ValueError(f"No host photometry of type {aperture_type}")

    if transient.host is None:
        raise ValueError("No host galaxy match")

    if transient.host.redshift is None:
        raise ValueError("No host galaxy redshift")

    filters, flux_maggies, flux_maggies_error = [], [], []

    for filter in Filter.objects.all():
        try:
            datapoint = photometry.get(filter=filter)
            filters.append(filter.transmission_curve())
            flux_maggies.append(jansky_to_maggies(datapoint.flux))
            flux_maggies_error.append(jansky_to_maggies(datapoint.flux_error))
        except AperturePhotometry.DoesNotExist or AperturePhotometry.MultipleObjectsReturned:
            raise

    obs_data = dict(
        wavelength=None,
        spectrum=None,
        unc=None,
        redshift=transient.host.redshift,
        maggies=np.array(flux_maggies),
        maggies_unc=np.array(flux_maggies_error),
        filters=filters,
    )

    return obs_data


def build_model(my, arguments):
    """
    Required by propector defined by
    https://prospect.readthedocs.io/en/latest/models.html
    """
    return 0.0


def build_sps(my, arguments):
    """
    Required by prospector defined by
    https://prospect.readthedocs.io/en/latest/usage.html
    """
    return 0.0


def build_noise(my, arguments):
    """
    Required by prospector defined by
    https://prospect.readthedocs.io/en/latest/usage.html
    """
    return 0.0
