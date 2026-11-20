"""MAST Cutouts API
This module defines a class derived from the Jupyter notebook provided by the MAST team to fetch data from the
MAST Cutouts service.
See https://cutouts.mast.stsci.edu/docs/ for documentation.
"""

import json
import os
import time
import requests
from astropy.io import fits
from dotmap import DotMap
from host.log import get_logger
logger = get_logger(__name__)


class MastCutoutAPI():
    def __init__(self, conf):
        # Define default configuration
        config = {
            'async_api_url': 'https://cutouts.mast.stsci.edu/api/v0/cutouts/async',
            'sync_api_url': 'https://cutouts.mast.stsci.edu/api/v0/cutouts/sync',
            'api_timeout': 60,
            'api_polling_interval_sec': 0.5,
            'api_polling_timeout': 30 * 60,  # Stop polling after 30 minutes
            'api_token': '',  # Credential used to authorize API access
            'positions': [],  # list[tuple[float, float]]
            'run_id': 'blast',  # Optional tracking label
            'size_px': 256,  # Square cutout size in pixels (PS1 stacks are 0.25"/pixel)
            'filters': ["g", "r", "i", "z", "y"],  # Filters to request
            'fits': True,  # Generate FITS data files
            'jpeg': False,  # Generate JPEG previews
            'results_limit_per_page': 100  # Max cutout records to fetch per results page
        }
        # Override default config with any input values, ignoring unsupported keys
        for key in config.keys():
            if key in conf:
                config[key] = conf[key]
                # No value can be empty
                assert config[key]
        self.config = DotMap(config)
        # # API token may be provided by env var
        # if not self.config.api_token:
        #     self.config.api_token = os.getenv('MAST_API_TOKEN', '')
        assert self.config.api_token
        # Configure HTTP session
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.config.api_token}"})

    def fetch_data(self):
        logger.debug(f'''Submitting async MAST cutout job (PanSTARRS {self.config.filters})...''')
        self.job_id = self.submit_async_job()
        logger.debug('''Polling async MAST cutout job (PanSTARRS {self.config.filters})...''')
        final_phase = self.wait_for_job(self.job_id)
        logger.debug(f'''MAST cutout job (PanSTARRS {self.config.filters}) finished with phase: {final_phase}''')
        if final_phase != "COMPLETED":
            error_response = self.session.get(os.path.join(self.config.async_api_url, self.job_id, 'error'),
                                              timeout=self.config.api_timeout)
            assert error_response.ok and error_response.text.strip()
            logger.error(f'''MAST cutout job error:\n{error_response.text}''')
        results_payload = self.get_cutout_results(self.job_id)
        cutouts = results_payload.get("results", [])
        fits_data = []
        for idx, cutout in enumerate(cutouts):
            try:
                fits_data.append({
                    'metadata': cutout,
                    'fits': fits.open(cutout.get("science"), fsspec_kwargs={"anon": True}),
                })
            except Exception as err:
                logger.error(f'''Error processing MAST cutout: {err}''')
        return fits_data

    def submit_async_job(self) -> str:
        """Submit a batch cutout job and return the server-assigned job ID."""
        form_data = {
            "position": [f"{ra} {dec}" for ra, dec in self.config.positions],
            "size": str(self.config.size_px),
            "generate_science": str(self.config.fits).lower(),
            "generate_preview": str(self.config.jpeg).lower(),
            "RUNID": self.config.run_id,
            "ps1": json.dumps(self.build_ps1_params(filters=self.config.filters)),
        }

        response = self.session.post(
            self.config.async_api_url,
            data=form_data,
            allow_redirects=False,
            timeout=self.config.api_timeout,
        )
        if response.status_code not in (200, 303):
            raise RuntimeError(f"Submit failed ({response.status_code}): {response.text[:500]}")

        location = response.headers.get("Location", "")
        if not location:
            raise RuntimeError("Submit succeeded but no Location header was returned")

        return location.rstrip("/").split("/")[-1]

    def build_ps1_params(self,
                         filters=["g", "r", "i", "z", "y"],  # Any of: g, r, i, z, y
                         surveys=["3pi"],  # 3pi and/or md
                         stack_files=["data"],  # stack products: data, mask, wt, exp, expwt, num
                         warp_files=[]  # warp products: data, mask, wt (leave empty for stacks only)
                         ) -> dict:
        params = {
            "filter": filters,
            "survey": surveys,
        }
        if stack_files:
            params["stack_file"] = stack_files
        if warp_files:
            params["warp_file"] = warp_files
        return params

    def get_job_phase(self, job_id: str) -> str:
        """Return the current UWS phase for a job."""
        response = self.session.get(
            os.path.join(self.config.async_api_url, self.job_id, 'phase'),
            timeout=self.config.api_timeout)
        response.raise_for_status()
        return response.text.strip().replace('"', "")

    def get_results_summary(self, job_id: str) -> dict:
        """Return JSON counters for queued, executing, completed, and failed cutouts."""
        response = self.session.get(os.path.join(self.config.async_api_url, self.job_id, 'results/summary'),
                                    timeout=self.config.api_timeout)
        response.raise_for_status()
        return response.json()

    def get_cutout_results(self, job_id: str, limit: int = 0, page: int = 0) -> dict:
        """Fetch paginated cutout metadata (JSON format)."""
        if not limit:
            limit = self.config.results_limit_per_page
        response = self.session.get(
            os.path.join(self.config.async_api_url, self.job_id, 'results/cutouts'),
            params={"output_format": "json", "limit": limit, "page": page},
            timeout=self.config.api_timeout,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_job(self, job_id: str) -> str:
        """Poll job phase until completion or timeout."""
        terminal_phases = {"COMPLETED", "ERROR", "ABORTED"}
        start = time.monotonic()

        while True:
            phase = self.get_job_phase(job_id)
            # summary = self.get_results_summary(job_id)
            # completed = summary.get("completed_jobs", 0)
            # total = summary.get("total_jobs", 0)
            # executing = summary.get("executing_jobs", 0)
            # queued = summary.get("queued_jobs", 0)

            if phase in terminal_phases:
                return phase

            if time.monotonic() - start > self.config.api_polling_timeout:
                raise TimeoutError(f"Job {job_id} did not finish within {self.config.api_polling_timeout} seconds")

            time.sleep(self.config.api_polling_interval_sec)
