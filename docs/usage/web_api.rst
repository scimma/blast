.. _api:

Web API
=======

Blast also provides an application programming interface, which you can use
to fetch Blast data programmatically.  The API allows queries on individual database tables (below),
as well as an API endpoint for getting all data for a given transient (:ref:`api_all`).

.. _api_individual:

Downloading Blast data for individual tables via Python
-------------------------------------------------------

The url endpoint to grab the data for a particular transient is
:code:`/api/transient/?name=transient_name&format=json`.
Here is an example Python snippet to load data as a Python dictionary for the transient
2010H

.. code:: python

    from urllib.request import urlopen
    import json

    response = urlopen('<base_blast_url>/api/transient/?name=2010H&format=json')
    data = json.loads(response.read())

Here data is a Python dictionary that contains the Blast science payload data m
model. We describe this model below.

Blast API data model
--------------------

The data model for different tables within Blast are described below.  Foreign key-linked fields
are also displayed to simplify API calls; for example, the attributes of the associated Host
are returned in addition to the Transient fields when a given transient is queried.

Transient fields
++++++++++++++++

API link: :code:`/api/transient/`

* :code:`name` - name of the transient e.g., 2022abc
* :code:`ra_deg` - transient Right Ascension in decimal degrees e.g., 132.34564
* :code:`dec_deg` - transient declination in decimal degrees e.g., 60.123424
* :code:`redshift` - transient redshift e.g., 0.01
* :code:`milkyway_dust_reddening` - transient E(B-V) e.g, 0.2
* :code:`processing_status` - Blast processing status of the transient.
    "processed" - transient has been complement processed by Blast and all data
    should be present in the science payload. "processing" - Blast is still
    processing this transient and some parts of the science payload may not
    be populated at the current time. "blocked" - this transient has not been
    successfully fully processed by Blast and some parts of the science payload
    will not be populated.
* :code:`spectroscopic_class` - spectroscopic classification, if any
* :code:`host` - foreign key link to the :code:`Host` object, described below.

Transient filtering options
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`name=` - search on transient name
* :code:`redshift_gte=` - filter on redshifts greater than or equal to the value provided
* :code:`redshift_lte=` - filter on redshifts less than or equal to the value provided
* :code:`host_redshift_gte=` - filter on host redshifts greater than or equal to the value provided
* :code:`host_redshift_lte=` - filter on host redshifts less than or equal to the value provided
* :code:`host_photometric_redshift_gte=` - filter on host photometric redshifts greater than or equal to the value provided
* :code:`host_photometric_redshift_lte=` - filter on host photometric redshifts less than or equal to the value provided

Example:
:code:`<blast_base_url>/api/transient/?redshift_gte=0.02`

Host fields
+++++++++++

API link: :code:`/api/host/`

* :code:`name` - name of the host e.g., NGC123
* :code:`ra_deg` - host Right Ascension in decimal degrees e.g., 132.34564
* :code:`dec_deg` - host declination in decimal degrees e.g., 60.123424
* :code:`redshift` - host redshift e.g., 0.01
* :code:`milkyway_dust_reddening` - host E(B-V) e.g, 0.2

Host filtering options
^^^^^^^^^^^^^^^^^^^^^^
* :code:`name=` - search on host name
* :code:`redshift_gte=` - filter on redshifts greater than or equal to the value provided
* :code:`redshift_lte=` - filter on redshifts less than or equal to the value provided
* :code:`photometric_redshift_gte=` - filter on photometric  redshifts greater than or equal to the value provided
* :code:`photometric_redshift_lte=` - filter on photometric redshifts less than or equal to the value provided

Example:
:code:`<blast_base_url>/api/host/?photometric_redshift_lte=0.02`


Aperture fields
+++++++++++++++

API link: :code:`/api/aperture/`

* :code:`ra_deg` - aperture Right Ascension in decimal degrees e.g., 132.3456
* :code:`dec_deg` - aperture declination in decimal degrees e.g., 60.123424
* :code:`orientation_deg` - orientation angle of the aperture in decimal degrees e.g., 30.123
* :code:`semi_major_axis_arcsec` - semi major axis of the aperture in arcseconds
* :code:`semi_minor_axis_arcsec` - semi minor axis of the aperture in arcseconds
* :code:`cutout` - link to the :code:`Cutout` object used to create aperture, described below
* :code:`type` - "local" or "global" aperture

Aperture filtering options
^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`transient=` - select apertures associated with a given transient name

Example:
:code:`<blast_base_url>/api/aperture/?transient=2010H`


AperturePhotometry fields
+++++++++++++++++++++++++

API link: :code:`/api/photometry/`

* :code:`flux` - Aperture photometry flux in mJy
* :code:`flux_error` - Aperture photometry flux error in mJy
* :code:`magnitude` - Aperture photometry magnitude
* :code:`magnitude_error` - Aperture photometry magnitude error
* :code:`aperture` - link to :code:`Aperture` object, described above
* :code:`filter` - link to photometric :code:`Filter` object
* :code:`transient` - link to :code:`Transient` object
* :code:`is_validated` - checks on contaminating objects in the aperture (global apertures only) or ability to resolve 2 kpc in physical scale (local apertures only)


Photometry filtering options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`transient=` - select aperture photometry associated with a given transient
* :code:`filter=` - select aperture photometry associated with a given photometric filter name

Example:
:code:`<blast_base_url>/api/aperturephotometry/?filter=H`


.. _sedfittingresult:

SEDFittingResult fit fields
+++++++++++++++++++++++++++

API link: :code:`/api/sedfittingresult/`

<aperture_type> can either be "local" or "global". <parameter> can be either,

* "log_mass" (log base 10 of the surviving host stellar mass [solar masses])
* "log_sfr" (log base 10 of the host star formation rate [solar masses / year])
* "log_ssfr" (log base 10 of the host specific star formation rate [/ year])
* "log_age" (log base 10 of the host stellar age [year])

<posterior_percentile> is the percentile value from the posterior distribution
which can either be "16", "50" ot "84"

* :code:`mass_surviving_ratio` - ratio of the surviving stellar mass to the total formed stellar mass
* :code:`<aperture_type>_aperture_host_<parameter>_<posterior_percentile>`
* :code:`transient` - link to :code:`Transient` object
* :code:`aperture` - link to :code:`Aperture` object

* :code:`chains_file` - MCMC chains for each parameter; files can be downloaded with the URL path :code:`<base_blast_url>/download_chains/<transient_name>/<aperture_type>`
* :code:`percentiles_file` - 16,50,84th percentiles for all parameters in the prospector-alpha model; files can be downloaded with the URL path :code:`<base_blast_url>/download_percentiles/<transient_name>/<aperture_type>`
* :code:`model_file` - best-fit spectrum, photometry, and uncertainties (downloaded in units of maggies); files can be downloaded with the URL path :code:`<base_blast_url>/download_modelfit/<transient_name>/<aperture_type>`


SED filtering options
^^^^^^^^^^^^^^^^^^^^^

* :code:`transient=` - select SED fitting results associated with a given transient
* :code:`aperture_type=` - select "global" or "local" SED fitting results

Example:

* :code:`<blast_base_url>/api/sedfittingresult/?transient=2010H`
* :code:`<blast_base_url>/api/sedfittingresult/?aperture_type=local`

Cutout fields
+++++++++++++

API link: :code:`/api/cutout/`

:code:`name` - the name of the cutout object
:code:`transient` - link to :code:`Transient` object
:code:`filter` - link to photometric :code:`Filter` object

Cutout filtering options
^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`transient` - select cutout images associated with a given transient
* :code:`filter` - select cutout images in a given photometric filter

Example:
:code:`<blast_base_url>/api/cutout/?transient=2010H`

Task fields
+++++++++++

API link: :code:`/api/task/`

* :code:`name` - name of each task

TaskRegister fields
+++++++++++++++++++

API link: :code:`/api/taskregister/`

* :code:`task` - link to :code:`Task` object
* :code:`status` - link to :code:`Status` object, which contains messages like "processed" or "failed"
* :code:`transient` - link to :code:`Transient` object
* :code:`user_warning` - see if user has flagged a given stage as problematic (true/false)

TaskRegister filtering options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`transient` - check the status of tasks for a given transient
* :code:`status` - search for all tasks with status "failed", for example
* :code:`task` - look for all instances of a given task

Example:
:code:`<blast_base_url>/api/taskregister/?status=failed`

.. _api_all:

Downloading *all* Blast data for a given transient
--------------------------------------------------

The url endpoint to grab the data for a particular transient is
:code:`/api/transient/get/<transient_name>`.  Here is an example Python snippet to load data as a Python dictionary for the transient 2018gv.

.. code:: python

    from urllib.request import urlopen
    import json

    response = urlopen('<base_blast_url>/api/transient/get/2018gv?format=json')
    data = json.loads(response.read())

Here data is a Python dictionary that contains the Blast science payload data
model. We describe this model below; for clarity, field names are slightly different
than in the base data model above.

Science payload data model
--------------------------

The data model for a single transient contains the following components.  Foreign key-linked fields
are also displayed to simplify API calls; for example, the attributes of the associated Host
are returned in addition to the Transient fields.

Transient fields
++++++++++++++++

* :code:`transient_name` - name of the transient e.g., 2022abc
* :code:`transient_ra_deg` - transient Right Ascension in decimal degrees e.g., 132.34564
* :code:`transient_dec_deg` - transient declination in decimal degrees e.g., 60.123424
* :code:`transient_redshift` - transient redshift e.g., 0.01
* :code:`transient_milkyway_dust_reddening` - transient E(B-V) e.g, 0.2
* :code:`transient_processing_status` - Blast processing status of the transient.
    "processed" - transient has been complement processed by Blast and all data
    should be present in the science payload. "processing" - Blast is still
    processing this transient and some parts of the science payload may not
    be populated at the current time. "blocked" - this transient has not been
    successfully fully processed by Blast and some parts of the science payload
    will not be populated.

Host fields
+++++++++++

* :code:`host_name` - name of the host e.g., NGC123
* :code:`host_ra_deg` - host Right Ascension in decimal degrees e.g., 132.34564
* :code:`host_dec_deg` - host declination in decimal degrees e.g., 60.123424
* :code:`host_redshift` - transient redshift e.g., 0.01
* :code:`host_milkyway_dust_reddening` - host E(B-V) e.g, 0.2

Aperture fields
+++++++++++++++

<aperture_type> can either be "local" or "global".

* :code:`<aperture_type>_aperture_ra_deg` - aperture Right Ascension in decimal degrees e.g., 132.3456
* :code:`<aperture_type>_aperture_dec_deg` - aperture declination in decimal degrees e.g., 60.123424
* :code:`<aperture_type>_orientation_deg` - orientation angle of the aperture in decimal degrees e.g., 30.123
* :code:`<aperture_type>_semi_major_axis_arcsec` - semi major axis of the aperture in arcseconds
* :code:`<aperture_type>_semi_minor_axis_arcsec` - semi minor axis of the aperture in arcseconds
* :code:`<aperture_type>_cutout` - name of the cutout used to create aperture e.g, 2MASS_H, None if not cutout was used


Photometry fields
+++++++++++++++++

<aperture_type> can either be "local" or "global". <filter> can be any of the
filters Blast downloads cutouts for e.g., 2MASS_H, 2MASS_J, SDSS_g ... . If the
data for a particular filter and transient does not exist the values will be None.

* :code:`<aperture_type>_aperture_<filter>_flux` - Aperture photometry flux in mJy
* :code:`<aperture_type>_aperture_<filter>_flux_error` - Aperture photometry flux error in mJy
* :code:`<aperture_type>_aperture_<filter>_magnitude` - Aperture photometry magnitude
* :code:`<aperture_type>_aperture_<filter>_magnitude_error` - Aperture photometry magnitude error


SED fit fields
++++++++++++++

<aperture_type> can either be "local" or "global". <parameter> can be either,

* "log_mass" (log base 10 of the host stellar mass [solar masses])
* "log_sfr" (log base 10 of the host star formation rate [solar masses / year])
* "log_ssfr" (log base 10 of the host specific star formation rate [/ year])
* "log_age" (log base 10 of the host stellar age [year])
* "log_tau" (log base 10 of the host star formation rate decline exponent [year])

<posterior_percentile> is the percentile value from the posterior distribution
which can either be "16", "50" ot "84"

* :code:`<aperture_type>_aperture_host_<parameter>_<posterior_percentile>`
