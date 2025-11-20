import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from astrodash.config.logging import get_logger

logger = get_logger(__name__)

def mean_zero_spectra(flux: np.ndarray, min_idx: int, max_idx: int, nw: int) -> np.ndarray:
    """Mean-zero a region of a spectrum."""
    out = np.zeros(nw)
    region = flux[min_idx:max_idx+1]
    mean = np.mean(region) if len(region) > 0 else 0
    out[min_idx:max_idx+1] = region - mean
    return out

def apodize(flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
    """Apply apodization to the spectrum edges."""
    apodized = np.copy(flux)
    edge_width = min(50, (max_idx - min_idx) // 4)
    if edge_width > 0:
        for i in range(edge_width):
            factor = 0.5 * (1 + np.cos(np.pi * i / edge_width))
            if min_idx + i < len(apodized):
                apodized[min_idx + i] *= factor
            if max_idx - i >= 0:
                apodized[max_idx - i] *= factor
    return apodized

def calc_redshift_from_crosscorr(crossCorr: np.ndarray, nw: int, dwlog: float) -> Tuple[float, np.ndarray]:
    """Calculate redshift from cross-correlation array."""
    deltaPeak = np.argmax(crossCorr[:int(nw // 2) + 1])
    zAxisIndex = np.concatenate((np.arange(-nw // 2, 0), np.arange(0, nw // 2)))
    if deltaPeak <= nw // 2:
        z = (np.exp(np.abs(zAxisIndex) * dwlog) - 1)[deltaPeak]
    else:
        z = -(np.exp(np.abs(zAxisIndex) * dwlog) - 1)[deltaPeak]
    return z, crossCorr

def cross_correlation(inputFlux: np.ndarray, tempFlux: np.ndarray, nw: int, tempMinMaxIndex: Any) -> np.ndarray:
    """Compute cross-correlation between input and template flux."""
    inputfourier = np.fft.fft(inputFlux)
    tempfourier = np.fft.fft(tempFlux)
    product = inputfourier * np.conj(tempfourier)
    xCorr = np.fft.fft(product)
    rmsInput = np.std(inputfourier)
    rmsTemp = np.std(tempfourier)
    xCorrNorm = (1. / (nw * rmsInput * rmsTemp)) * xCorr
    xCorrNormRearranged = np.concatenate((xCorrNorm[int(len(xCorrNorm) / 2):], xCorrNorm[0:int(len(xCorrNorm) / 2)]))
    crossCorr = np.correlate(inputFlux, tempFlux, mode='full')[::-1][int(nw / 2):int(nw + nw / 2)] / max(np.correlate(inputFlux, tempFlux, mode='full'))
    return crossCorr

def get_redshift(inputFlux: np.ndarray, tempFlux: np.ndarray, nw: int, dwlog: float, tempMinMaxIndex: Any) -> Tuple[float, np.ndarray]:
    """Get redshift and cross-correlation for a single template."""
    crossCorr = cross_correlation(inputFlux, tempFlux, nw, tempMinMaxIndex)
    redshift, crossCorr = calc_redshift_from_crosscorr(crossCorr, nw, dwlog)
    return redshift, crossCorr

def get_median_redshift(
    inputFlux: np.ndarray,
    tempFluxes: List[np.ndarray],
    nw: int,
    dwlog: float,
    inputMinMaxIndex: Any,
    tempMinMaxIndexes: List[Any],
    tempNames: List[str],
    outerVal: float = 0.5
) -> Tuple[Optional[float], Optional[Dict[str, Any]], Optional[str], Optional[float]]:
    """Estimate median redshift from multiple templates."""
    # Mean-zero and apodize input flux
    inputFlux = mean_zero_spectra(inputFlux, inputMinMaxIndex[0], inputMinMaxIndex[1], nw)
    inputFlux = apodize(inputFlux, inputMinMaxIndex[0], inputMinMaxIndex[1])
    redshifts = []
    crossCorrs = {}
    for i, tempFlux in enumerate(tempFluxes):
        assert tempFlux[0] == outerVal or tempFlux[0] == 0.0
        # Mean-zero and apodize template flux
        tempFlux_proc = mean_zero_spectra(tempFlux - outerVal, tempMinMaxIndexes[i][0], tempMinMaxIndexes[i][1], nw)
        tempFlux_proc = apodize(tempFlux_proc, tempMinMaxIndexes[i][0], tempMinMaxIndexes[i][1])
        redshift, crossCorr = get_redshift(inputFlux, tempFlux_proc, nw, dwlog, tempMinMaxIndexes[i])
        redshifts.append(redshift)
        crossCorrs[tempNames[i]] = crossCorr
    if redshifts:
        medianIndex = np.argsort(redshifts)[len(redshifts) // 2]
        medianRedshift = redshifts[medianIndex]
        medianName = tempNames[medianIndex]
        try:
            stdRedshift = np.std(redshifts)
        except Exception as e:
            logger.error(f"Error calculating redshift error: {e}")
            stdRedshift = None
    else:
        return None, None, None, None
    if len(redshifts) >= 10:
        redshiftError = np.std(redshifts)
    else:
        redshiftError = None
    return medianRedshift, crossCorrs, medianName, stdRedshift
