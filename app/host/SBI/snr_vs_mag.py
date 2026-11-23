#!/usr/bin/env python
# D. Jones - 5/26/23
"""
Create S/N vs. mag files for every filter
Just ASCII file output should be fine
"""
import h5py
import numpy as np
from host.models import *
from scipy.stats import binned_statistic


def main():
    for f in Filter.objects.all():
        phot = AperturePhotometry.objects.filter(filter=f, magnitude__isnull=False)
        mag = np.array(phot.values_list("magnitude", flat=True))
        if not len(mag):
            continue
        mag_error = np.array(phot.values_list("magnitude_error", flat=True))
        flux = np.array(phot.values_list("flux", flat=True))
        flux_error = np.array(phot.values_list("flux_error", flat=True))

        magbins = np.arange(np.min(mag), np.max(mag), 0.25)
        snr_binned = binned_statistic(
            mag, flux / flux_error, bins=magbins, statistic="mean"
        ).statistic
        count_binned = binned_statistic(
            mag, flux / flux_error, bins=magbins, statistic="count"
        ).statistic

        ### we need some minimum number of counts
        ### and we need to interpolate over the missing points
        mincount = 5
        magbins = (magbins[1:] + magbins[:-1]) / 2.0
        idx = np.where(count_binned > mincount)[0]
        idx_valid = np.arange(np.min(idx), np.max(idx) + 1, 1)

        snr_bins_interp = np.interp(magbins[idx_valid], magbins[idx], snr_binned[idx])
        magbins = magbins[idx_valid]

        with open(f"host/SBI/snrfiles/{f.name}_magvsnr.txt", "w") as fout:
            for m, s in zip(magbins, snr_bins_interp):
                print(f"{m:.3f} {s:.3f}", file=fout)

    return
