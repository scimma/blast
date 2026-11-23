New dependencies
================

Blast and all its associated services are run in `Docker <https://www.docker.com/>`_
containers. This allows the application to be portable. You may be working on
some code that requires you to add a new Python package. To add this
dependency, you have to do two things.

1. Add the package and version to the :code:`app/requirements.txt`. This allows Docker
to `pip install <https://pip.pypa.io/en/stable/cli/pip_install/>`_ your new package.

2. Change the :code:`BLAST_IMAGE` variable in your :code:`env/.env.dev` to be
:code:`blast_base` instead of :code:`blast_latest`. This will force Docker to
build and install your package.

.. note::

    Setting :code:`BLAST_IMAGE=blast_latest` in your :code:`env/env.dev` means
    the latest Blast Docker image
    is downloaded and used when running Blast locally. This is faster than building
    the image from scratch every time. :code:`BLAST_IMAGE=blast_base` forces the
    image to be built from scratch which is required when you have added a new
    dependency. Eventually, when a new dependency is accepted into the main branch it
    will become part of the Blast latest image.
