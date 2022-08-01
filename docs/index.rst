.. blast documentation master file, created by
   sphinx-quickstart on Thu Dec 23 12:02:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to blast's documentation!
=================================

.. image:: https://readthedocs.org/projects/blast/badge/?version=latest
.. image:: https://github.com/astrophpeter/blast/actions/workflows/continuous-integration.yml/badge.svg
.. image:: https://results.pre-commit.ci/badge/github/astrophpeter/blast/main.svg
.. image:: https://codecov.io/gh/astrophpeter/blast/branch/main/graph/badge.svg?token=Y9JPAF8HVD

Blast is developed on `github <https://github.com/astrophpeter/blast>`_.

Blast is a public web application built to find a transinet's host
galaxy, identify the available archival data, and measure the resulting
host galaxy star formation rates, masses, and stellar ages - for every
new transient reported to the IAU in real-time after the transient is
announced. This information is not provided by any existing transient
broker service.

Using the web interface
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: Web interface

   web_pages
   web_api

Developer Guide
---------------

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   dev_getting_started
   dev_running_blast
   dev_system_pages
   dev_workflow
   dev_overview_of_repo
   dev_github_issues
   dev_documentation
   dev_task_runner
   dev_faqs


Code API
--------

.. toctree::
   :maxdepth: 2
   :caption: Code API

   API/models
   API/transient_name_server
   API/base_tasks
