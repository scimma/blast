.. _data_files:

Blast Data Files
================

In addition to the data queryable via the :doc:`Blast API <web_api>`, Blast provides several downloadable data files that can be used to reproduce the SED-fitting results that Blast displays.  These are the percentiles file, the parameter chains file, and the model file.  Each file is described below.  Additionally, Blast includes an option to export all files (via the :code:`Download Data` menu), which, in addition to the previous files, includes FITS cutout images, Prospector hdf5-formatted output files, and a JSON file with schema data associated with each transient.  We hope to update/simplify a number of the use cases described below in the coming months.


Parameter Chains
----------------

Basic Example
^^^^^^^^^^^^^

::
   
    data = np.load('<transient>_global_chain.npz',allow_pickle=True)
    chains = np.atleast_1d(data['chains'])[0]
    # 16, 50, 84th percentile confidence intervals for stellar mass:
    print(np.percentile(chains['stellar_mass'],[16,50,84]))


Description
^^^^^^^^^^^

The :code:`*chain.npz` files contain posterior samples for each Prospector-:math:`\alpha` parameter, as well as the derived star-formation history (:code:`sfh`), the mass-weighted age (:code:`mwa`), the star-formation rate (:code:`sfr`), and the specific star-formation rate (:code:`ssfr`).  :code:`sfr` and :code:`ssfr` have three columns, which correspond to the SFR/sSFR averaged over the last 0 Myr, 30~Myr, and 100~Myr timescales, respectively; the 100~Myr average is the default value reported via the Blast webpages.  The :code:`age_intep` variable gives the time axis corresponding to the :code:`sfh`.  Note that the stellar mass is the *surviving* stellar mass, in contrast to the total formed mass often reported by Prospector; the ratio of surviving to total stellar mass is reported as :code:`mass_surviving_ratio` via the :ref:`sedfittingresult`.

Broadly the units should follow the descriptions in the :ref:`sed_params`, and note that :code:`age_interp` is in units of Gyr.

Percentiles
-----------

Basic Example
^^^^^^^^^^^^^
::

    data = np.load('<transient>_global_perc.npz',allow_pickle=True)
    perc = np.atleast_1d(data['percentiles'])[0]
    # 16th, 50th, and 84th percentile confidence for solar metallicity:
    print(perc['logzsol'])


Description
^^^^^^^^^^^

This file is broadly the same as the parameter chains file above, but each parameter has an array of three elements (or sometimes three columns) corresponding to the 16th, 50th, and 84th percentiles from the posterior samples.  Additionally, these files include confidence intervals for the model spectra and model photometry, in :code:`modspec` and :code:`modphot`, respectively.  The :code:`theta_lbs` key contains the full set of prospector model parameters for reference.

Note: for transients at z < 0.015, an offset is applied in the model spectra and photometry due to practical considerations.  See :ref:`low_z` below for help interpreting these data.


SED Model
---------

Basic Example
^^^^^^^^^^^^^

::

    import numpy as np
    import matplotlib.pyplot as plt

    data = np.load('<transient>_global_modeldata.npz',allow_pickle=True)

    # show the median best-fit spectrum and its uncertainties
    plt.plot(data['rest_wavelength'],data['spec'],color='k')
    plt.plot(data['rest_wavelength'],data['spec_16'],ls='--',color='0.8')
    plt.plot(data['rest_wavelength'],data['spec_84'],ls='--',color='0.8')

    # plotting model photometry is a little annoying
    # **tested only on prospector version 1.4.0, does not work in v2**
    import prospect.io.read_results as reader
    from sedpy import observate
    import pandas as pd
    result, obs, _ = reader.results_from('<transient>_global.h5', dangerous=False)

    # get wavelength from filter names
    # first download this directory: https://github.com/scimma/blast/tree/main/data/transmission
    lam_obs = []
    for f in obs['filters']:
	trans_file = f'transmission/{f}.txt'
	transmission_curve = pd.read_csv(trans_file, sep="\s+", header=None)
	wavelength = transmission_curve[0].to_numpy()
	transmission = transmission_curve[1].to_numpy()
	filt = observate.Filter(
		kname=f, nick=f, data=(wavelength, transmission)
	       )
	lam_obs += [filt.wave_effective]

    # plot model photometry -- apologies for the lack of wavelength array
    plt.errorbar(np.array(lam_obs)/(1+result['obs']['redshift']),data['phot'],yerr=[data['phot']-data['phot_16'],data['phot_84']-data['phot']],fmt='o')

    plt.show()



Description
^^^^^^^^^^^

The SED model file contains the best-fit model spectrum, photometry, and their respective confidence intervals, in units of maggies.  For a basic plotting example, follow the example code above.  To convert to a somewhat more standard unit like microJy you can use something like the following convenience function::


    def maggies_to_uJy(flux_maggies):
        """
        Converts spectral flux density from maggies to units of uJy.
        """
        return flux_maggies * 10 ** (0.4 * 23.9)


Unfortunately, for plotting the model photometry as described above, you will also need:
1) the prospector results file, in hdf5 format, which is downloadable using the "export with all files" link in a given transient results page, and
2) the filter transmission curves from Blast, available `here <https://github.com/scimma/blast/tree/main/data/transmission>`_.


Observed Photometry
-------------------

The observed photometry used in SED fitting is in units of AB magnitudes and corrected for
Milky Way reddening (Schafly & Finkbeiner 2012).  It can be derived from the :code:`transient.json` file
as follows, with fluxes in units of microJansky::


    import json
    from sedpy import observate
    import extinction

    with open('<transient>/transient.json') as f:
	d = json.load(f)

    filter_dict = {}
    for f in d['filters']:

	trans_file = f"transmission/{f['fields']['name']}.txt"
	transmission_curve = pd.read_csv(trans_file, sep="\s+", header=None)
	wavelength = transmission_curve[0].to_numpy()
	transmission = transmission_curve[1].to_numpy()
	filt = observate.Filter(
		kname=f['fields']['name'], nick=f['fields']['name'], data=(wavelength, transmission)
	       )

	filter_dict[f['pk']] = (f['fields']['name'],filt.wave_effective)

    lam_obs,flux_list,flux_error_list = [],[],[]
    mwebv = d['transient']['fields']['milkyway_dust_reddening']
    for a in d['apertures']:
       if a['aperturephotometry'][0]['fields']['is_validated'] != 'true':
	   continue

       wave_eff = filter_dict[a['aperturephotometry'][0]['fields']['filter']][1]
       ext_corr = extinction.fitzpatrick99(np.array([wave_eff]), mwebv * 3.1, r_v=3.1)[
	   0
       ]
       flux = a['aperturephotometry'][0]['fields']['flux']*10 ** (0.4 * ext_corr)
       flux_error = a['aperturephotometry'][0]['fields']['flux_error']*10 ** (0.4 * ext_corr)
       # 1% error floor is used for SED fitting:
       flux_error = np.sqrt(flux_error**2 + (0.01*flux)**2)

       # append to lists
       lam_obs += [wave_eff]
       flux_list += [flux]
       flux_error_list += [flux_error]

    lam_obs,flux_list,flux_error_list = \
      np.array(lam_obs),np.array(flux_list),np.array(flux_error_list)


Finally, to plot the observed photometry alongside the SED model above, we just need to convert the SED model to microJansky::


    redshift = d['host']['fields']['redshift']
    # alternately, d['transient']['fields']['redshift'] if host redshift is not available
    plt.errorbar(lam_obs/(1+redshift),flux_list,yerr=flux_error_list,fmt='o',color='r')
    plt.plot(data['rest_wavelength'],maggies_to_uJy(data['spec']),color='k')
    plt.show()


.. _low_z:
    
Low-Redshift Transients
-----------------------

In the local volume, small changes in redshift equate to large differences in predicted magnitude,
making it impractical to generate a sufficiently large training set for the SBI approach in this regime.  Instead,
we artificially redshift the transient photometry by an additional :math:`\Delta z = 0.015` using the WMAP 2009 cosmology.
Resulting model files are therefore at an increased distance compared to the transient and their flux must be adjusted
for plotting alongside the data.  We believe the small systematic errors caused by this subtle wavelength
shift should be negligible for most/all SED-fitting applications.  However, it is important to note
that --- for all SED-fitting approaches --- some derived SED parameters, such as stellar mass,
become increasingly unreliable when the luminosity distance cannot be reliably inferred, as is the case at :math:`z < 0.01`.

To generate plots using model files for transients at :math:`z < 0.015`, just apply a small offset to the normalization::

    from astropy.cosmology import WMAP9 as cosmo
    mag_off = cosmo.distmod(0.015+redshift).value - cosmo.distmod(redshift).value
    data = np.load('<transient>_global_modeldata.npz',allow_pickle=True)
    plt.plot(data['rest_wavelength'],maggies_to_uJy(data['spec'])* 10 ** (0.4 * mag_off))

