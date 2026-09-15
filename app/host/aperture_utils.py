from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from host.host_utils import select_best_cutout
from host.plotting_utils import delete_cached_file
from host.plotting_utils import normalize_pixel_axes
from host.plotting_utils import _pixel_to_arcsec_matrix
from host.plotting_utils import temp_results_paths_from_canonical_path
from host.models import Aperture
from host.models import Cutout
from host.log import get_logger
from host.object_store import ObjectStore
logger = get_logger(__name__)

APERTURE_EDITABLE_FIELDS = (
    "ra_deg",
    "dec_deg",
    "semi_major_axis_arcsec",
    "semi_minor_axis_arcsec",
    "orientation_deg",
)

APERTURE_EQUALITY_TOLERANCE = 1e-9

def _resolve_cutout(transient, filter_name):
    if not filter_name:
        return select_best_cutout(transient.name)
    return Cutout.objects.filter(name__exact=f"{transient.name}_{filter_name}").first()

def _edit_float(edit, key):
    try:
        value = float(edit[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f'''"{key}" is missing or not a number''')
    if not np.isfinite(value):
        raise ValueError(f'''"{key}" must be finite''')
    return value

def _unchanged(stored, incoming):
    if stored is None:
        return False
    return np.isclose(
        float(stored), float(incoming), rtol=0.0, atol=APERTURE_EQUALITY_TOLERANCE,
    )

def _apply_aperture_edit(transient, wcs, edit):
    aperture_id = edit.get("apertureId")
    if aperture_id is None:
        raise ValueError("missing apertureId")

    aperture = Aperture.objects.get(id=aperture_id, transient=transient)

    geometry = aperture_sky_geometry(
        wcs,
        x=_edit_float(edit, "x"),
        y=_edit_float(edit, "y"),
        semi_major_px=_edit_float(edit, "semiMajor"),
        semi_minor_px=_edit_float(edit, "semiMinor"),
        theta_rad=_edit_float(edit, "thetaRadians"),
    )
    if (geometry["semi_major_axis_arcsec"] <= 0
            or geometry["semi_minor_axis_arcsec"] <= 0):
        raise ValueError(f"aperture {aperture_id} would have a non-positive axis")

    updates = {field: geometry[field] for field in APERTURE_EDITABLE_FIELDS}

    if aperture.type == "local":
        updates["semi_minor_axis_arcsec"] = updates["semi_major_axis_arcsec"]

    changed = {
        field: value
        for field, value in updates.items()
        if not _unchanged(getattr(aperture, field), value)
    }

    if changed:
        for field, value in changed.items():
            setattr(aperture, field, value)
        aperture.save(update_fields=list(changed))
        logger.info(
            f'''Aperture {aperture.id} (type "{aperture.type}", transient '''
            f'''"{transient.name}") edited by hand: {changed}'''
        )
        # Note: AperturePhotometry and SEDFittingResult rows for this aperture are now stale. 

    return {
        "apertureId": aperture.id,
        "apertureType": aperture.type,
        "changed": sorted(changed),
        "semiMajorArcsec": aperture.semi_major_axis_arcsec,
        "semiMinorArcsec": aperture.semi_minor_axis_arcsec,
        "orientationDeg": aperture.orientation_deg,
    }

def cutout_wcs(cutout):
    if cutout is None or not cutout.fits:
        return None
    local_tmp_path, object_key = temp_results_paths_from_canonical_path(cutout.fits.name)
    s3 = ObjectStore()
    if not s3.object_exists(object_key):
        logger.error(f'''Data object "{object_key}" not found for cutout "{cutout.name}".''')
        return None
    s3.download_object(path=object_key, file_path=local_tmp_path)
    try:
        with fits.open(local_tmp_path) as fits_file:
            return WCS(fits_file[0].header)
    except Exception as err:
        logger.error(f'''Error reading WCS from cutout "{cutout.name}": {err}''')
        return None
    finally:
        delete_cached_file(local_tmp_path)

def aperture_sky_geometry(wcs, x, y, semi_major_px, semi_minor_px, theta_rad):
    semi_major_px, semi_minor_px, theta_rad = normalize_pixel_axes(semi_major_px, semi_minor_px, theta_rad)

    cd = _pixel_to_arcsec_matrix(wcs)

    maj_dx = semi_major_px * np.cos(theta_rad)
    maj_dy = semi_major_px * np.sin(theta_rad)
    min_dx = -semi_minor_px * np.sin(theta_rad)
    min_dy = semi_minor_px * np.cos(theta_rad)

    maj_east = cd[0] * maj_dx + cd[1] * maj_dy
    maj_north = cd[2] * maj_dx + cd[3] * maj_dy
    min_east = cd[0] * min_dx + cd[1] * min_dy
    min_north = cd[2] * min_dx + cd[3] * min_dy

    sky_position = wcs.celestial.pixel_to_world(float(x), float(y))

    return {
        "ra_deg": float(sky_position.ra.deg),
        "dec_deg": float(sky_position.dec.deg),
        "semi_major_axis_arcsec": float(np.hypot(maj_east, maj_north)),
        "semi_minor_axis_arcsec": float(np.hypot(min_east, min_north)),
        "orientation_deg": float(np.degrees(np.arctan2(maj_east, maj_north)) % 180.0)
    }