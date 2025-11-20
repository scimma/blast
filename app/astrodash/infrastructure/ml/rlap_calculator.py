"""
RLAP (Relative Likelihood of Association with a Prototype) calculation functions
for the production backend.
"""

import os
import numpy as np
from scipy.signal import argrelmax
from astrodash.infrastructure.ml.dash_utils import get_training_parameters
from astrodash.infrastructure.ml.templates import create_spectrum_template_handler
from astrodash.shared.utils.helpers import get_redshift_axis, normalize_age_bin as _normalize_age_bin
from astrodash.shared.utils.redshift import get_median_redshift
from astrodash.config.settings import get_settings
from astrodash.config.logging import get_logger

logger = get_logger(__name__)

class RlapCalculator:
    def __init__(self, inputFlux, templateFluxes, templateNames, wave, inputMinMaxIndex, templateMinMaxIndexes):
        self.inputFlux = inputFlux
        self.templateFluxes = templateFluxes
        self.templateNames = templateNames
        self.wave = wave
        self.inputMinMaxIndex = inputMinMaxIndex
        self.templateMinMaxIndexes = templateMinMaxIndexes
        self.nw = len(wave)
        self.dwlog = np.log(wave[-1] / wave[0]) / self.nw

    def _cross_correlation(self, templateFlux, templateMinMaxIndex):
        templateFlux = templateFlux.astype('float')
        # Ensure template flux has the same length as input flux
        if len(templateFlux) != len(self.inputFlux):
            logger.warning(f"Template flux length ({len(templateFlux)}) doesn't match input flux length ({len(self.inputFlux)})")
            # Truncate or pad template flux to match input flux length
            if len(templateFlux) > len(self.inputFlux):
                templateFlux = templateFlux[:len(self.inputFlux)]
            else:
                # Pad with zeros
                padding = np.zeros(len(self.inputFlux) - len(templateFlux))
                templateFlux = np.concatenate([templateFlux, padding])

        xCorr = np.correlate(self.inputFlux, templateFlux, mode='full')
        rmsInput = np.sqrt(np.mean(self.inputFlux ** 2))
        rmsTemp = np.sqrt(np.mean(templateFlux ** 2))
        xCorrNorm = xCorr / (rmsInput * rmsTemp * self.nw)
        rmsXCorr = np.sqrt(np.mean(xCorrNorm ** 2))
        # Use the raw normalized cross-correlation without rolling to keep zero-lag at index (nw - 1)
        xCorrNormRearranged = xCorrNorm
        rmsA = 1
        return xCorr, rmsInput, rmsTemp, xCorrNorm, rmsXCorr, xCorrNormRearranged, rmsA

    def _get_peaks(self, crosscorr):
        peakindexes = argrelmax(crosscorr)[0]
        # Ensure peak indexes are within bounds
        peakindexes = peakindexes[peakindexes < len(crosscorr)]
        ypeaks = [abs(crosscorr[i]) for i in peakindexes]
        arr = list(zip(peakindexes, ypeaks))
        arr.sort(key=lambda x: x[1], reverse=True)
        return arr

    def _calculate_r(self, crosscorr, rmsA):
        peaks = self._get_peaks(crosscorr)
        if len(peaks) < 2:
            return 0, 0, 0
        deltapeak1, h1 = peaks[0]
        deltapeak2, h2 = peaks[1]
        r = abs((h1 - rmsA) / (np.sqrt(2) * rmsA))
        fom = (h1 - 0.05) ** 0.75 * (h1 / h2) if h2 != 0 else 0
        return r, deltapeak1, fom

    def calculate_rlap(self, crosscorr, rmsAntisymmetric, templateFlux):
        r, deltapeak, fom = self._calculate_r(crosscorr, rmsAntisymmetric)
        # Compute shift relative to the true zero-lag index of the full cross-correlation (which is at len(crosscorr)//2)
        zero_lag_index = (len(crosscorr) - 1) // 2
        shift = int(deltapeak - zero_lag_index)
        iminindex, imaxindex = self.min_max_index(self.inputFlux)
        tminindex, tmaxindex = self.min_max_index(templateFlux)
        overlapminindex = int(max(iminindex + shift, tminindex))
        overlapmaxindex = int(min(imaxindex - 1 + shift, tmaxindex - 1))

        # Clamp to valid bounds to avoid out-of-range indexing
        overlapminindex = max(0, min(self.nw - 1, overlapminindex))
        overlapmaxindex = max(0, min(self.nw - 1, overlapmaxindex))

        if overlapmaxindex <= overlapminindex:
            # No valid overlap region
            lap = 0.0
        else:
            minWaveOverlap = self.wave[overlapminindex]
            maxWaveOverlap = self.wave[overlapmaxindex]
            lap = np.log(maxWaveOverlap / minWaveOverlap) if minWaveOverlap > 0 else 0.0
        rlap = 5 * r * lap
        fom = fom * lap
        return r, lap, rlap, fom

    def min_max_index(self, flux):
        minindex, maxindex = (0, self.nw - 1)
        zeros = np.where(flux == 0)[0]
        j = 0
        for i in zeros:
            if (i != j):
                break
            j += 1
            minindex = j
        j = int(self.nw) - 1
        for i in zeros[::-1]:
            if (i != j):
                break
            j -= 1
            maxindex = j
        return minindex, maxindex

    def rlap_score(self, tempIndex):
        xcorr, rmsinput, rmstemp, xcorrnorm, rmsxcorr, xcorrnormRearranged, rmsA = self._cross_correlation(
            self.templateFluxes[tempIndex].astype('float'), self.templateMinMaxIndexes[tempIndex])
        crosscorr = xcorrnormRearranged
        r, lap, rlap, fom = self.calculate_rlap(crosscorr, rmsA, self.templateFluxes[tempIndex])
        return r, lap, rlap, fom

    def rlap_label(self):
        if not np.any(self.inputFlux):
            return "No flux", True
        self.zAxis = get_redshift_axis(self.nw, self.dwlog)
        rlapList = []
        for i in range(len(self.templateNames)):
            r, lap, rlap, fom = self.rlap_score(tempIndex=i)
            rlapList.append(rlap)
        rlapMean = round(np.mean(rlapList), 2)
        rlapLabel = str(rlapMean)
        rlapWarning = rlapMean < 6
        return rlapLabel, rlapWarning

def calculate_rlap_with_redshift(wave, flux, templateFluxes, templateNames, templateMinMaxIndexes, inputMinMaxIndex, redshift=None):
    """
    Calculate RLap score for the input spectrum using the best-match template.
    If redshift is not provided, estimate it using the templates.
    Returns (rlap_score, used_redshift, rlap_warning)
    """
    pars = get_training_parameters()
    w0, w1, nw = pars['w0'], pars['w1'], pars['nw']
    dwlog = np.log(w1 / w0) / nw

    # If redshift is not provided, estimate it using the templates
    if redshift is None:
        logger.info("Estimating redshift for RLap calculation using best-match template(s).")
        est_redshift, _, _, _ = get_median_redshift(
            flux, templateFluxes, nw, dwlog, inputMinMaxIndex, templateMinMaxIndexes, templateNames
        )
        if est_redshift is None:
            logger.error("Redshift estimation failed. RLap will be calculated in observed frame.")
            est_redshift = 0.0
        redshift = est_redshift
    else:
        logger.info(f"Using provided redshift {redshift} for RLap calculation.")

    # Shift input spectrum to rest-frame
    rest_wave = wave / (1 + redshift)
    # Interpolate flux onto the log-wavelength grid if needed (assume input is already log-binned)
    # For faithful port, we assume input is already on the correct grid

    # Calculate RLap
    rlap_calc = RlapCalculator(flux, templateFluxes, templateNames, rest_wave, inputMinMaxIndex, templateMinMaxIndexes)
    rlap_label, rlap_warning = rlap_calc.rlap_label()
    return rlap_label, redshift, rlap_warning

def compute_rlap_for_matches(matches, best_match, log_wave, input_flux_log, template_fluxes, template_names, template_minmax_indexes, known_z):
    """
    Compute RLap for the best match and attach to matches and best_match dicts.
    """
    # Find the best match index
    if not matches:
        return matches, best_match
    best_match_idx = np.argmax([m['probability'] for m in matches])
    best = matches[best_match_idx]
    sn_type = best['type']
    age = best['age']
    # Find the correct template(s) for this type/age
    input_minmax_index = get_nonzero_minmax(input_flux_log)
    rlap_label, used_redshift, rlap_warning = calculate_rlap_with_redshift(
        log_wave, input_flux_log, template_fluxes, template_names, template_minmax_indexes, input_minmax_index,
        redshift=best['redshift'] if known_z else None
    )
    # Attach RLap to all matches (or just best match)
    for m in matches:
        m['rlap'] = rlap_label
        m['rlap_warning'] = rlap_warning
    best_match['rlap'] = rlap_label
    best_match['rlap_warning'] = rlap_warning
    return matches, best_match

def prepare_log_wavelength_and_templates(spectrum, template_filename=None):
    """
    Utility to prepare log-wavelength grid, interpolate input spectrum, and load templates.
    Returns: log_wave, input_flux_log, snTemplates
    """
    pars = get_training_parameters()
    w0, w1, nw = pars['w0'], pars['w1'], pars['nw']
    dwlog = np.log(w1 / w0) / nw
    log_wave = w0 * np.exp(np.arange(nw) * dwlog)

    handler_template_path = None
    if template_filename is not None:
        settings = get_settings()
        template_dir = os.path.dirname(settings.template_path)
        handler_template_path = os.path.join(template_dir, template_filename)

    handler = create_spectrum_template_handler('dash', template_path=handler_template_path)
    snTemplates = handler.get_all_templates()
    logger.debug(f"Available sn_types: {list(snTemplates.keys())}")
    for sn_type_key in snTemplates:
        logger.debug(f"  {sn_type_key}: {list(snTemplates[sn_type_key].keys())}")
    input_flux = np.array(spectrum.y)
    input_wave = np.array(spectrum.x)
    input_flux_log = np.interp(log_wave, input_wave, input_flux, left=0, right=0)
    return log_wave, input_flux_log, snTemplates, dwlog, nw, w0, w1

def get_templates_for_type_age(snTemplates, sn_type, age_norm, log_wave):
    """
    Given snTemplates dict, SN type, normalized age bin, and log-wavelength grid,
    return template_fluxes, template_names, template_minmax_indexes for that type/age.
    """
    template_fluxes = []
    template_names = []
    template_minmax_indexes = []
    if sn_type in snTemplates:
        age_bin_keys = [str(k).strip() for k in snTemplates[sn_type].keys()]
        if age_norm.strip() in age_bin_keys:
            real_key = [k for k in snTemplates[sn_type].keys() if str(k).strip() == age_norm.strip()][0]
            snInfo = snTemplates[sn_type][real_key].get('snInfo', None)
            if isinstance(snInfo, np.ndarray) and snInfo.shape[0] > 0:
                for i in range(snInfo.shape[0]):
                    template_wave = snInfo[i][0]
                    template_flux = snInfo[i][1]
                    interp_flux = np.interp(log_wave, template_wave, template_flux, left=0, right=0)
                    nonzero = np.where(interp_flux != 0)[0]
                    if len(nonzero) > 0:
                        tmin, tmax = nonzero[0], nonzero[-1]
                    else:
                        tmin, tmax = 0, len(interp_flux) - 1
                    template_fluxes.append(interp_flux)
                    template_names.append(f"{sn_type}:{age_norm}")
                    template_minmax_indexes.append((tmin, tmax))
    return template_fluxes, template_names, template_minmax_indexes

def get_nonzero_minmax(flux):
    """Return (min_index, max_index) of nonzero flux, or (0, len(flux)-1) if all zero."""
    nonzero = np.where(flux != 0)[0]
    if len(nonzero) > 0:
        return nonzero[0], nonzero[-1]
    else:
        return 0, len(flux) - 1

def normalize_age_bin(age: str) -> str:
    """Proxy to shared helper to normalize age-bin strings to 'N to M' format."""
    return _normalize_age_bin(age)
