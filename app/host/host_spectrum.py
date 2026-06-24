import time
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.ipac.ned import Ned
from django.conf import settings
from sparcl.client import SparclClient

from host.log import get_logger
from host.models import TaskLock

logger = get_logger(__name__)


def fetch_host_spectrum(position):
    """
    Fetch a spectrum for a galaxy at the given sky position using a hierarchical approach.

    Queries SPARCL first with top priority given to DESI-DR1, followed by SDSS-DR17, 
    and BOSS-DR17 as a third choice. If multiple records exist for the same specid, 
    prefers the 'main' survey. If no match is found in SPARCL, falls back to querying NED 
    as a last resort. All downloaded scalar metadata and physical units are dynamically 
    injected into the final FITS structure.

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Sky position to search around.

    Returns
    -------
    :result : dict or None
        Dict with keys 'hdulist', 'spectrum_id', 'redshift',
        'wavelength_min_angstrom', 'wavelength_max_angstrom', or None.
    """
    timeout = settings.QUERY_TIMEOUT
    time_start = time.time()
    logger.debug('''Acquiring host spectrum query lock...''')
    while timeout > time.time() - time_start:
        if TaskLock.objects.request_lock('host_spectrum_query'):
            break
        logger.debug('''Waiting to acquire host spectrum query lock...''')
        time.sleep(1)

    result = None
    try:
        ra = position.ra.deg
        dec = position.dec.deg
        radius_deg = 3.0 / 3600.0
        ra_min, ra_max = ra - radius_deg, ra + radius_deg
        dec_min, dec_max = dec - radius_deg, dec + radius_deg

        # Query SPARCL for DESI or SDSS or BOSS spectrum
        client = SparclClient(announcement=False, read_timeout=5500, connect_timeout=5600)
        outfields = ['sparcl_id', 'specid', 'data_release', 'survey', 'ra', 'dec']
        constraints = {
            'ra': [ra_min, ra_max],
            'dec': [dec_min, dec_max],
            'data_release': ['DESI-DR1', 'SDSS-DR17', 'BOSS-DR17'],
        }

        found = client.find(outfields=outfields, constraints=constraints, fmt='pandas')

        if found is not None and not found.empty:
            # Convert to native dicts for ultra-fast priority and deduplication processing
            records = found.to_dict('records')
            priority_map = {'DESI-DR1': 1, 'SDSS-DR17': 2, 'BOSS-DR17': 3}

            # Filter to the highest available priority tier
            best_priority = min(priority_map.get(r['data_release'], 999) for r in records)
            
            # Deduplicate by specid (preferring 'main') and calculate distance
            specid_map = {}
            for r in records:
                if priority_map.get(r['data_release'], 999) != best_priority:
                    continue
                
                specid = r['specid']
                if specid not in specid_map:
                    specid_map[specid] = r
                else:
                    if r['survey'] == 'main' and specid_map[specid]['survey'] != 'main':
                        specid_map[specid] = r

                # Simple Cartesian distance squared for precise angular sorting within 3"
                r['sep_sq'] = (r['ra'] - ra) ** 2 + (r['dec'] - dec) ** 2

            # Select the record closest to the target coordinates among the preferred tier
            best_record = min(specid_map.values(), key=lambda x: x['sep_sq'])
            selected_id = best_record['sparcl_id']

            # Expand include list to pull down all common science and catalog metadata
            include = [
                'sparcl_id', 'specid', 'data_release', 'redshift', 'flux', 'wavelength', 
                'ivar', 'mask', 'spectype', 'ra', 'dec', 'survey', 'wavemin', 'wavemax',
                'datasetgroup', 'dateobs', 'dateobs_center', 'exptime', 'instrument',
                'redshift_err', 'redshift_warning', 'site', 'specprimary', 'targetid',
                'telescope', 'wave_sigma', 'model'
            ]

            retrieved = client.retrieve(uuid_list=[selected_id], include=include)

            if retrieved and retrieved.records:
                record = retrieved.records[0]
                wavelength = np.asarray(record.wavelength, dtype=np.float64)
                flux = np.asarray(record.flux, dtype=np.float64)
                ivar = np.asarray(record.ivar, dtype=np.float64)
                spec_redshift = float(record.redshift) if record.redshift is not None else None

                # Extract units from the SPARCL response header metadata
                try:
                    units_info = retrieved.hdr['UNITS'].get(record.data_release, {})
                except (AttributeError, KeyError, TypeError):
                    units_info = {}
                wave_unit = units_info.get('wavelength', 'AA')
                flux_unit = units_info.get('flux', '1e-17 erg cm-2 s-1 AA-1')
                ivar_unit = units_info.get('ivar', '1e+34 cm4 s2 AA2 erg-2')

                # Create and systematically populate the primary FITS header
                primary_hdr = fits.Header()
                primary_hdr['BUNIT'] = (flux_unit, 'Physical units of the flux array')
                primary_hdr['ORIGIN'] = (f"{getattr(record, 'data_release', 'SPARCL')} (via SPARCL)", 'Data source')
                
                # Loop through all downloaded metadata fields and push scalars into the header
                array_fields = {'flux', 'wavelength', 'ivar', 'mask', 'model', 'wave_sigma'}
                for field in include:
                    if field in array_fields:
                        continue
                    val = getattr(record, field, None)
                    if val is not None and not isinstance(val, (np.ndarray, list)):
                        # Astropy automatically uses HIERARCH for long keyword strings
                        primary_hdr[field.upper()] = (val, f'SPARCL {field} metadata')

                # Dynamically construct columns list based on returned arrays
                fits_columns = [
                    fits.Column(name='wavelength', format='D', array=wavelength, unit=wave_unit),
                    fits.Column(name='flux',       format='E', array=flux,       unit=flux_unit),
                    fits.Column(name='ivar',       format='E', array=ivar,       unit=ivar_unit),
                ]

                # Append model spectrum if available
                if getattr(record, 'model', None) is not None:
                    model_array = np.asarray(record.model, dtype=np.float32)
                    fits_columns.append(fits.Column(name='model', format='E', array=model_array, unit=flux_unit))

                # Append quality bitmask array if available
                if getattr(record, 'mask', None) is not None:
                    mask_array = np.asarray(record.mask, dtype=np.int32)
                    fits_columns.append(fits.Column(name='mask', format='J', array=mask_array))

                hdulist = fits.HDUList([
                    fits.PrimaryHDU(header=primary_hdr),
                    fits.BinTableHDU.from_columns(fits_columns, name='SPECTRUM'),
                ])

                result = {
                    'hdulist': hdulist,
                    'spectrum_id': str(getattr(record, 'specid', record.sparcl_id)),
                    'redshift': spec_redshift,
                    'wavelength_min_angstrom': float(wavelength.min()),
                    'wavelength_max_angstrom': float(wavelength.max()),
                }
                logger.info(f'''Spectrum successfully fetched via SPARCL from {getattr(record, 'data_release', 'unknown')}.''')

        # Fallback to NED as a last resort if SPARCL yields nothing
        if result is None:
            logger.debug('''No matching spectrum found in SPARCL. Falling back to NED...''')
            ned_results = Ned.query_region(position, radius=3.0 * u.arcsec)
            
            if ned_results is not None and len(ned_results) > 0:
                ned_results = ned_results[ned_results['Redshift'].mask == False]  # noqa: E712
                
                if len(ned_results) > 0:
                    pos_results = SkyCoord(
                        ned_results['RA'].value, ned_results['DEC'].value, unit=u.deg
                    )
                    sep = position.separation(pos_results).arcsec
                    best_idx = int(np.argmin(sep))
                    object_name = str(ned_results[best_idx]['Object Name'])
                    redshift = float(ned_results[best_idx]['Redshift'])

                    spectra = Ned.get_spectra(object_name)
                    if spectra:
                        hdulist = spectra[0]
                        try:
                            crval1 = hdulist[0].header['CRVAL1']
                            cdelt1 = (hdulist[0].header.get('CD1_1') or
                                      hdulist[0].header.get('CDELT1'))
                            naxis1 = hdulist[0].header['NAXIS1']
                            wavelengths = 10 ** (crval1 + cdelt1 * np.arange(naxis1))
                            wl_min = float(wavelengths[0])
                            wl_max = float(wavelengths[-1])
                        except Exception:
                            wl_min, wl_max = None, None

                        result = {
                            'hdulist': hdulist,
                            'spectrum_id': object_name,
                            'redshift': redshift,
                            'wavelength_min_angstrom': wl_min,
                            'wavelength_max_angstrom': wl_max,
                        }
                        logger.debug(f'''NED spectrum found for "{object_name}", z={redshift}''')
                    else:
                        logger.debug(f'''No NED spectra available for object "{object_name}".''')
                else:
                    logger.debug('''NED objects found but none have a valid redshift.''')
            else:
                logger.debug('''No NED objects found at position.''')

    except Exception as err:
        logger.warning(f'''Error fetching host spectrum: {err}''')
    finally:
        logger.debug('''Releasing host spectrum query lock...''')
        TaskLock.objects.release_lock('host_spectrum_query')

    return result