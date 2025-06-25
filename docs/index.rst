.. Blast documentation master file, created by
   sphinx-quickstart on Thu Dec 23 12:02:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./_static/blast_logo_transparent.png
   :alt: Blast logo

Blast is a public web application built to find a transient's host
galaxy, identify the available archival data, and measure the resulting
host galaxy star formation rates, masses, and stellar ages - for every
new transient reported to the IAU in real-time after the transient is
announced. This information is not provided by any existing transient
broker service.

Blast is developed on `GitHub <https://github.com/scimma/blast>`_.

If you have a problem or need help using the Blast service at https://blast.scimma.org and cannot find an answer here or in the `GitHub Discussions <https://github.com/scimma/blast/discussions>`_, please `📧 email our support team <mailto:support@scimma.org?subject=Blast%20support%20request&body=
Please%20provide%20a%20detailed%20description%20of%20your%20problem%20or%20question.%20Make%20sure%20to%20search%20the%20documentation%20%28https%3A%2F%2Fblast.readthedocs.io%29%20and%20our%20GitHub%20Discussions%20%28https%3A%2F%2Fgithub.com%2Fscimma%2Fblast%2Fdiscussions%29%20for%20answers%20first.%20We%20appreciate%20your%20patience%20in%20receiving%20a%20response%3B%20please%20allow%20one%20or%20two%20days%20before%20sending%20a%20follow-up%20email.%0A%0AFull%20name%3A%20%20%0AAffiliation%3A%20%20%0AMessage%3A%20%20%0A>`_.

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
   developer_guide/dev_initial_dataset
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
