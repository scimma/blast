.. _api:

Web API
=======

Blast provides an HTTP application programming interface (API) for fetching data programmatically.  The API allows queries on :ref:`individual data objects associated with a transient<api_individual>`, as well as :ref:`fetching all data for a given transient<api_all>`.

.. _api_all:

Downloading *all* Blast data for a given transient
--------------------------------------------------

A **transient dataset** is the complete set of information associated with a transient: it consists of the information stored in :ref:`Blast database objects<models>` and associated data files, such as the cutout images and the SED fit files. 

A **full transient dataset** can be exported using :code:`/api/transient/export/<transient_name>/all`, which packages the data into a compressed archive file (standard ``.tar.gz`` format) that the user downloads.

There are currently two API endpoints that return **all tabular data** associated with a transient (i.e. data stored as text instead of as binary files) as a JSON-formatted document:

1. :code:`/api/transient/get/<transient_name>?format=json`

   The structure, or schema, of the document returned by this endpoint is detailed in the :ref:`science_payload_schema` section below. It is a relatively flat hierarchy.

   Here is an example Python snippet to load data as a Python dictionary for the transient 2026dix.
   
   .. code:: python
   
       from urllib.request import urlopen
       import json
   
       response = urlopen('<base_blast_url>/api/transient/get/2026dix?format=json')
       data = json.loads(response.read())

2. :code:`/api/transient/export/<transient_name>`

   The schema of the document returned by this endpoint serializes transient-associated objects more closely to the internal models (see :ref:`models`). It was designed is to capture a transient dataset in a self-contained way amenable to import into any Blast instance. The result is a more hierarchical structure, but one with static keys and less redundancy that also supports simpler parsing algorithms. The schema includes additional related objects: cutouts, surveys, and filters. Although the surveys and filters exist independently of a particular transient, they are included because transient cutout objects are associated with filters associated with surveys.

.. _api_individual:

Downloading individual Blast data objects
-------------------------------------------------------

Internally, Blast defines a set of :ref:`data models<models>` (:code:`transient`, :code:`host`, :code:`aperture`, etc.) that define the structure of objects associated with an astronomical transient.

There are API endpoints to fetch these objects from the Blast database, with URLs of the form
:code:`/api/<model_name>` (e.g. :code:`/api/host`). Each of these functions, detailed below, offers filtering options that narrow the scope of the returned results (e.g. :code:`/api/host/?redshift_gte=4&format=json`). **As the Blast database continues to grow, requesting all objects by omitting filters is discouraged.**

As a concrete example, here is a Python script that will fetch the :code:`transient` database object associated with the transient 2026dix:

.. code:: python

    from urllib.request import urlopen
    import json

    response = urlopen('<base_blast_url>/api/transient/?name=2026dix&format=json')
    data = json.loads(response.read())

This returns a JSON document like:

.. code:: javascript

   [
     {
        "id": 76390,
        "ra_deg": 177.6559502,
        "dec_deg": 55.3535889,
        "name": "2026dix",
        "display_name": null,
        "public_timestamp": "2026-02-16T09:26:15.072000Z",
        "redshift": 0.003185,
        "spectroscopic_class": "SN IIb",
        "milkyway_dust_reddening": 0.0110031844861805,
        "progress": 100,
        "software_version": "1.8.7",
        "host": {
          "id": 92200,
          "ra_deg": 177.662201,
          "dec_deg": 55.353867,
          "name": "NGC3913",
          "redshift": 0.0041580065070319,
          "redshift_err": 0.0012474019521095,
          "photometric_redshift": null,
          "photometric_redshift_err": null,
          "milkyway_dust_reddening": 0.011014966275543,
          "software_version": "1.8.7"
        }
     }
   ]

.. _api_data_schema:

Blast API data schema
---------------------

Each API endpoint for fetching the various :ref:`Blast data objects<models>`, along with its response data schema, is described below. Foreign key-linked fields are also displayed to simplify API calls; for example, the attributes of the associated Host are returned in addition to the Transient fields when a given transient is queried.

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

.. _science_payload_schema:

Science payload data schema
---------------------------

The data schema for the JSON-formatted response of the API endpoint :code:`/api/transient/get/<transient_name>?format=json` contains the components detailed in subsections below. Field names differ slightly from those in the :ref:`API data schema<api_data_schema>` above for the sake of clarity.

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
