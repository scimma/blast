.. Blast documentation master file, created by
   sphinx-quickstart on Thu Dec 23 12:02:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Blast's documentation!
=================================

.. image:: https://readthedocs.org/projects/blast/badge/?version=latest
.. image:: https://github.com/scimma/blast/actions/workflows/continuous-integration.yml/badge.svg
.. image:: https://results.pre-commit.ci/badge/github/astrophpeter/blast/main.svg
.. image:: https://codecov.io/gh/astrophpeter/blast/branch/main/graph/badge.svg?token=Y9JPAF8HVD
.. image:: https://www.codefactor.io/repository/github/astrophpeter/blast/badge

Blast is developed on `GitHub <https://github.com/scimma/blast>`_.

Blast is a public web application built to find a transient's host
galaxy, identify the available archival data, and measure the resulting
host galaxy star formation rates, masses, and stellar ages - for every
new transient reported to the IAU in real-time after the transient is
announced. This information is not provided by any existing transient
broker service.


Usage
-----

.. toctree::
   :maxdepth: 2
   :caption: Usage

   usage/web_pages
   usage/web_api
   usage/sed_params
   usage/batch


Developer Guide
---------------

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/dev_getting_started
   developer_guide/dev_running_blast
   developer_guide/dev_system_pages
   developer_guide/dev_workflow
   developer_guide/dev_overview_of_repo
   developer_guide/dev_github_issues
   developer_guide/dev_documentation
   developer_guide/dev_task_runner
   developer_guide/dev_installing_packages
   developer_guide/dev_faqs


Code API
--------

.. toctree::
   :maxdepth: 2
   :caption: Code API

   API/models
   API/transient_name_server
   API/base_tasks
   API/datamodel
   API/components


Acknowledgements
----------------

.. toctree::
   :maxdepth: 2
   :caption: Acknowledgements

   acknowledgements/software
   acknowledgements/contributors
   acknowledgements/data_sources
