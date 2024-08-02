Spectral Energy Distribution Parameters
=======================================

Blast currently uses the Prospector-:math:`\alpha` model to model
galaxy SEDs.  To add your own model, please get in touch
and consult the :ref:`Developer Guide <dev_guide>`!  Via the webpages, we
report full posterior chains for the following parameters.

Prospector-:math:`\alpha` Parameters
------------------------------------
Prospector-:math:`\alpha` parameters in the output parameter files are labeled as follows:

* :code:`logmass`, :math:`\log_{10}(M_{\ast}/M_{\odot})`: Logarithm of the stellar mass.
* :code:`logzsol`, :math:`\log_{10}(Z_{\ast}/Z_{\odot})`: Logarithm of the stellar metallicity.
* :code:`logsfr_ratios`, :math:`\Delta\log_{10}(\text{SFR})_{1:6}`: Difference in star formation rate (:math:`log_{10}(SFR)`) between adjacent bins of the binned star formation history.
  This SFH model is described in `Leja et al. (2019a) <https://ui.adsabs.harvard.edu/abs/2019ApJ...876....3L/abstract>`_. The binning
  is done in lookback time from the galaxy's redshift. The two most recent bins cover [0,30] Myr and [30,100] Myr. The
  oldest bin covers :math:`[0.85, 1.0] \times t_{age}(z)`, i.e., the first 15% of the galaxy's life. The remaining bins are spaced
  logarithmically in lookback time. There are normally 7 bins in total, including the three fixed ones, giving 6 parameters.  The web pages
  give the SFH in binned star-formation rate.
* :code:`dust2`, :math:`\tau_2`: The optical depth of the diffuse dust in the galaxy. The dust treatment follows a two-component `Charlot & Fall (2000) <https://ui.adsabs.harvard.edu/abs/2000ApJ...539..718C/abstract>`_ model where most stars are affected by diffuse dust attenuation, but young stars also see birth cloud dust.
* :code:`dust_index`, :math:`\delta`: The power law index of a modified `Calzetti (2000) <https://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C/abstract>`_ dust attenuation law. The power law modification is by `Noll et al. (2009) <https://ui.adsabs.harvard.edu/abs/2009A%26A...507.1793N/abstract>`_. The UV bump strength is tied to the slope following `Kriek & Conroy (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...775L..16K/abstract>`_.
* :code:`dust1_fraction`, :math:`\tau_1/\tau_2`: The optical depth of the birth cloud dust attenuation (as a fraction of dust2). This only affects stars younger than 10 Myr. Again, this is from the Charlot & Fall (2000) model.
* :code:`log_fagn`, :math:`\log_{10}(f_\text{AGN})`: Logarithm of the fraction of bolometric luminosity that is due to AGN emission using the Nenkova et al. (`2008a <https://ui.adsabs.harvard.edu/abs/2008ApJ...685..147N/abstract>`_, `2008b <https://ui.adsabs.harvard.edu/abs/2008ApJ...685..160N/abstract>`_) CLUMPY models. This is described in detail in `Leja et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018ApJ...854...62L/abstract>`_.
* :code:`log_agn_tau`: :math:`\ln(\tau_\text{AGN})`, the optical depth (at 5500 Å) of a dust clump in the AGN torus. Again, see `Leja et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018ApJ...854...62L/abstract>`_ and the  Nenkova et al. (`2008a <https://ui.adsabs.harvard.edu/abs/2008ApJ...685..147N/abstract>`_, `2008b <https://ui.adsabs.harvard.edu/abs/2008ApJ...685..160N/abstract>`_) templates.
* :code:`gas_logz`, :math:`\log_{10}(Z_\text{gas}/Z_\odot)`: Logarithm of the gas phase metallicity.
* :code:`duste_qpah`, :math:`Q_\text{PAH}`: Fraction of dust mass in polycyclic aromatic hydrocarbons (PAHs).  This parameter,
  along with the :code:`duste_umin` and :code:`log_duste_gamma` parameters, relate to the `Draine & Li (2007) <https://ui.adsabs.harvard.edu/abs/2007ApJ...657..810D/abstract>`_
  silicate-graphite polycyclic aromatic hydrocarbon (PAH) dust emission model.
* :code:`duste_umin`, :math:`U_\text{min}`: Minimum starlight intensity that the dust grains are exposed to.
* :code:`log_duste_gamma`, :math:`\log_{10}(\gamma_e)`: Fraction of dust mass exposed to starlight of minimum intensity.
