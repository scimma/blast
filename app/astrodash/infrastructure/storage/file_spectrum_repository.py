from astrodash.domain.repositories.spectrum_repository import SpectrumRepository
from astrodash.domain.models.spectrum import Spectrum
from astrodash.config.settings import Settings, get_settings
from astrodash.shared.utils.validators import validate_spectrum
from astrodash.config.logging import get_logger
from astrodash.core.exceptions import FileReadException, SpectrumValidationException
from astrodash.infrastructure.ml.data_processor import DashSpectrumProcessor
import json
import os
import uuid
import urllib3
import requests
from typing import Any, Optional

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = get_logger(__name__)

class FileSpectrumRepository(SpectrumRepository):
    """
    File-based repository for spectra. Stores spectra as JSON files in a directory.
    Uses DashSpectrumProcessor to parse files.
    """
    def __init__(self, config: Settings = None):
        self.config = config or get_settings()
        self.processor = DashSpectrumProcessor(w0=4000, w1=9000, nw=1024)
        self.storage_dir = os.path.join(self.config.storage_dir, "spectra")
        os.makedirs(self.storage_dir, exist_ok=True)

    def save(self, spectrum: Spectrum) -> Spectrum:
        try:
            validate_spectrum(spectrum.x, spectrum.y, spectrum.redshift)
            if not spectrum.id:
                spectrum.id = str(uuid.uuid4())
            path = os.path.join(self.storage_dir, f"{spectrum.id}.json")
            with open(path, "w") as f:
                json.dump({
                    "id": spectrum.id,
                    "osc_ref": spectrum.osc_ref,
                    "file_name": spectrum.file_name,
                    "x": spectrum.x,
                    "y": spectrum.y,
                    "redshift": spectrum.redshift,
                    "meta": spectrum.meta
                }, f)
            logger.debug(f"Saved spectrum {spectrum.id} to {path}")
            return spectrum
        except Exception as e:
            logger.error(f"Error saving spectrum: {e}", exc_info=True)
            raise

    def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        path = os.path.join(self.storage_dir, f"{spectrum_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return Spectrum(
            id=data["id"],
            osc_ref=data.get("osc_ref"),
            file_name=data.get("file_name"),
            x=data["x"],
            y=data["y"],
            redshift=data.get("redshift"),
            meta=data.get("meta", {})
        )

    def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        # Not implemented for file-based repo
        return None

    def get_from_file(self, file: Any) -> Optional[Spectrum]:
        # Accepts UploadFile or file-like object
        filename = getattr(file, 'name', getattr(file, 'filename', 'unknown'))
        logger.debug(f"Reading spectrum file: {filename}")

        try:
            import pandas as pd
            import io
            import numpy as np

            # Handle file reading like the old backend
            file_obj = file
            if hasattr(file, 'file'):
                # This is a FastAPI UploadFile - get the underlying file object
                file_obj = file.file

            # Handle different file types like the old backend
            if filename.lower().endswith('.lnw'):
                return self._read_lnw_file(file_obj, filename)
            elif filename.lower().endswith(('.dat', '.txt')):
                return self._read_text_file(file_obj, filename)
            elif filename.lower().endswith('.fits'):
                return self._read_fits_file(file_obj, filename)
            else:
                logger.error(f"Unsupported file format: {filename}")
                return None

        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}", exc_info=True)
            return None

    def _read_lnw_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read .lnw file with specific wavelength filtering like the old backend."""
        try:
            import re

            # Read file contents
            if hasattr(file_obj, 'read'):
                file_obj.seek(0)
                content = file_obj.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
            else:
                with open(file_obj, 'r') as f:
                    content = f.read()

            # Parse like the old backend
            lines = content.splitlines()
            spectrum = []

            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Split by whitespace and filter by wavelength
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 2:
                        try:
                            wavelength = float(parts[0])
                            flux = float(parts[1])

                            # Apply wavelength filter like the old backend
                            if 4000 <= wavelength <= 9000:
                                spectrum.append((wavelength, flux))
                        except ValueError:
                            continue

            if not spectrum:
                logger.error(f"No valid spectrum data found in {filename}")
                return None

            # Sort by wavelength and separate arrays
            spectrum.sort(key=lambda x: x[0])
            wavelength = [x[0] for x in spectrum]
            flux = [x[1] for x in spectrum]

            # Create spectrum object
            spectrum_obj = Spectrum(x=list(wavelength), y=list(flux), file_name=filename)

            # Validate before saving
            try:
                validate_spectrum(spectrum_obj.x, spectrum_obj.y, spectrum_obj.redshift)
            except Exception as e:
                logger.error(f"Spectrum validation failed for .lnw file: {e}")
                return None

            saved_spectrum = self.save(spectrum_obj)
            return saved_spectrum

        except Exception as e:
            logger.error(f"Error reading .lnw file {filename}: {e}", exc_info=True)
            return None

    def _read_text_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read .dat or .txt file like the old backend."""
        try:
            import pandas as pd
            import io

            # Read file content
            if hasattr(file_obj, 'read'):
                file_obj.seek(0)
                content = file_obj.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
            else:
                with open(file_obj, 'r') as f:
                    content = f.read()

            # Try to parse with pandas first
            try:
                # Try different separators
                for sep in ['\t', ',', ' ']:
                    try:
                        df = pd.read_csv(io.StringIO(content), sep=sep, header=None, comment='#')
                        if len(df.columns) >= 2:
                            break
                    except:
                        continue
                else:
                    # If pandas fails, try manual parsing
                    lines = content.splitlines()
                    data = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    data.append([float(parts[0]), float(parts[1])])
                                except ValueError:
                                    continue
                    df = pd.DataFrame(data, columns=[0, 1])

                if len(df) == 0:
                    logger.error(f"No valid data found in {filename}")
                    return None

                # Extract wavelength and flux
                wavelength = df[0].tolist()
                flux = df[1].tolist()

                # Apply wavelength filter like the old backend
                filtered_data = [(w, f) for w, f in zip(wavelength, flux) if 4000 <= w <= 9000]

                if not filtered_data:
                    logger.error(f"No data in wavelength range 4000-9000 in {filename}")
                    return None

                wavelength = [x[0] for x in filtered_data]
                flux = [x[1] for x in filtered_data]

                # Create spectrum object
                spectrum_obj = Spectrum(x=list(wavelength), y=list(flux), file_name=filename)

                # Validate before saving
                try:
                    validate_spectrum(spectrum_obj.x, spectrum_obj.y, spectrum_obj.redshift)
                except Exception as e:
                    logger.error(f"Spectrum validation failed for text file: {e}")
                    return None

                saved_spectrum = self.save(spectrum_obj)
                return saved_spectrum

            except Exception as e:
                logger.error(f"Error parsing text file {filename}: {e}")
                return None

        except Exception as e:
            logger.error(f"Error reading text file {filename}: {e}", exc_info=True)
            return None

    def _read_fits_file(self, file_obj, filename: str) -> Optional[Spectrum]:
        """Read .fits file like the old backend."""
        try:
            from astropy.io import fits
            import numpy as np

            # Read FITS file
            if hasattr(file_obj, 'read'):
                file_obj.seek(0)
                hdul = fits.open(file_obj)
            else:
                hdul = fits.open(file_obj)

            try:
                # Try to find spectrum data in FITS
                spectrum_data = None

                # Look for common spectrum extensions
                for ext in ['SPECTRUM', 'SPECTRA', 'FLUX', 'DATA']:
                    if ext in hdul:
                        spectrum_data = hdul[ext].data
                        break

                # If not found, try first extension
                if spectrum_data is None and len(hdul) > 1:
                    spectrum_data = hdul[1].data

                if spectrum_data is None:
                    logger.error(f"No spectrum data found in FITS file {filename}")
                    return None

                # Extract wavelength and flux
                if hasattr(spectrum_data, 'wavelength') and hasattr(spectrum_data, 'flux'):
                    wavelength = spectrum_data.wavelength
                    flux = spectrum_data.flux
                elif hasattr(spectrum_data, 'wave') and hasattr(spectrum_data, 'flux'):
                    wavelength = spectrum_data.wave
                    flux = spectrum_data.flux
                elif len(spectrum_data.dtype.names) >= 2:
                    # Assume first two columns are wavelength and flux
                    wavelength = spectrum_data[spectrum_data.dtype.names[0]]
                    flux = spectrum_data[spectrum_data.dtype.names[1]]
                else:
                    logger.error(f"Cannot determine wavelength/flux columns in FITS file {filename}")
                    return None

                # Convert to lists and apply wavelength filter
                wavelength = wavelength.tolist()
                flux = flux.tolist()

                filtered_data = [(w, f) for w, f in zip(wavelength, flux) if 4000 <= w <= 9000]

                if not filtered_data:
                    logger.error(f"No data in wavelength range 4000-9000 in FITS file {filename}")
                    return None

                wavelength = [x[0] for x in filtered_data]
                flux = [x[1] for x in filtered_data]

                # Create spectrum object
                spectrum_obj = Spectrum(x=list(wavelength), y=list(flux), file_name=filename)

                # Validate before saving
                try:
                    validate_spectrum(spectrum_obj.x, spectrum_obj.y, spectrum_obj.redshift)
                except Exception as e:
                    logger.error(f"Spectrum validation failed for FITS file: {e}")
                    return None

                saved_spectrum = self.save(spectrum_obj)
                return saved_spectrum

            finally:
                hdul.close()

        except Exception as e:
            logger.error(f"Error reading FITS file {filename}: {e}", exc_info=True)
            return None


class OSCSpectrumRepository(SpectrumRepository):
    """
    Repository for retrieving spectra from the Open Supernova Catalog (OSC) API.
    """

    def __init__(self, config: Settings = None):
        self.config = config or get_settings()
        # Use configurable OSC API URL, fallback to default if not set
        self.base_url = getattr(self.config, 'osc_api_url', 'https://api.astrocats.space')
        # The working backend uses the base URL directly, not with /api suffix

    def save(self, spectrum: Spectrum) -> Spectrum:
        # OSC repository doesn't save - it only retrieves
        raise NotImplementedError("OSC repository doesn't support saving")

    def get_by_id(self, spectrum_id: str) -> Optional[Spectrum]:
        # OSC repository uses OSC references, not internal IDs
        return None

    def get_by_osc_ref(self, osc_ref: str) -> Optional[Spectrum]:
        """Get spectrum from OSC API."""
        try:
            # Extract the SN name from the OSC reference format
            # Input: "osc-sn2002er-0" -> Extract: "sn2002er" -> Convert to: "SN2002ER"
            if osc_ref.startswith('osc-'):
                # Remove "osc-" prefix and "-0" suffix
                sn_name = osc_ref[4:-2]  # "osc-sn2002er-0" -> "sn2002er"
            else:
                sn_name = osc_ref

            # The API expects the object name in uppercase
            obj_name = sn_name.upper()

            # Use the correct API structure: /{OBJECT_NAME}/spectra/time+data
            url = f"{self.base_url}/{obj_name}/spectra/time+data"
            logger.debug(f"OSC repository: Attempting to fetch spectrum from {url}")

            response = requests.get(url, verify=False, timeout=30)
            logger.debug(f"OSC repository: Received response status {response.status_code} for {osc_ref}")

            if response.status_code != 200:
                logger.error(f"OSC API returned status {response.status_code} for {osc_ref}")
                logger.error(f"Response content: {response.text[:500]}")  # Log first 500 chars of response
                return None

            data = response.json()
            logger.debug(f"OSC repository: Raw API response for {osc_ref}: {data}")

            # Parse using the actual API response structure
            try:
                # The API returns: {"SN2002ER": {"spectra": [["52512", [["wavelength", "flux"], ...]]]}}
                # We need: data[obj_name]["spectra"][0][1] to get the spectrum data array
                spectrum_data = data[obj_name]["spectra"][0][1]

                # Convert to numpy arrays and transpose to get wave, flux
                import numpy as np
                wave, flux = np.array(spectrum_data).T.astype(float)

                logger.debug(f"OSC repository: Successfully parsed spectrum data for {osc_ref}")

            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Failed to parse spectrum data structure for {osc_ref}: {e}")
                logger.error(f"Response structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                return None

            # Extract redshift if available (default to 0.0 as in reference)
            redshift = 0.0  # Redshift needs to be fetched separately as per reference

            # Extract object name for filename
            obj_name = data.get('name', obj_name)

            # Generate spectrum ID using just the SN name to avoid duplicates
            spectrum_id = f"osc_{sn_name}"

            # Create spectrum object
            spectrum = Spectrum(
                id=spectrum_id,
                x=wave.tolist(),  # Convert numpy array to list
                y=flux.tolist(),   # Convert numpy array to list
                redshift=redshift,
                osc_ref=osc_ref,
                file_name=f"{obj_name}.json",
                meta={"source": "osc", "object_name": obj_name}
            )

            logger.debug(f"OSC repository: Created spectrum object: {spectrum}")

            # Validate spectrum
            try:
                validate_spectrum(spectrum.x, spectrum.y, spectrum.redshift)
                logger.debug(f"OSC repository: Spectrum validation passed")
            except Exception as e:
                logger.error(f"OSC repository: Spectrum validation failed: {e}")
                return None

            return spectrum

        except requests.RequestException as e:
            logger.error(f"OSC API request failed for {osc_ref}: {e}")
            logger.error(f"OSC API URL being used: {self.base_url}")
            logger.error(f"Full OSC reference: {osc_ref}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving spectrum from OSC for {osc_ref}: {e}", exc_info=True)
            return None

    def get_from_file(self, file: Any) -> Optional[Spectrum]:
        # OSC repository doesn't read files
        return None
