Batch
=====

This pages explains how to use Blast in offline batch mode. This mode
processes a batch of user inputted transients with the Blast pipeline and outputs
a serialized science payload.

First, read and follow the instructions on how to run Blast locally :doc:`here </developer_guide/dev_running_blast>`
up to and including the Setup environment file section.

Input transient file
--------------------

First, create an input file with a batch of transients to be processed. This file
must be a csv following the the format with the transient name and sky position.

.. code::

    name,ra,dec
    2022example1,123.453454644,-30.34324332353
    2022example2,120.32304852353,20.03204833089302

The Right ascension and declination of the transient must be in decimal degrees.
The column names must match exactly.

Output results file
-------------------

You must also create an empty results csv file. This is where the results of
Blast will be written to as the batch of transients is processed.

Environment setup
-----------------

In addition to all the environment variables default setup you must specify the
path to you input batch file, absolute or relative to the docker-compose.yml file,

.. code::

    BATCH_CSV=<path_to_your_transient_input_file>

You must also specify the path to the target output directory, in which a timestamped
CSV file will be generated (:code:`/tmp/blast_results` by default):

.. code::

    OUTPUT_DIR=<path_to_output_results>

Running in batch mode
---------------------

With the environment variables set, you can run Blast in batch mode from the base
Blast directory,

.. code::

    bash run/blastctl batch up

This will spin up a local version of Blast that will take your batch of transients
as input and process them. Results will be periodically written to your results
file. When all transients have been processed or are blocked the containers will
all exit. The format of the results file is specified by the :doc:`Blast API <web_api>`.
