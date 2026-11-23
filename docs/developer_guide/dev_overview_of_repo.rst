Repository overview
===================

This section gives a high-level overview of the Blast repository. It is
designed to help new developers navigate the code.

run
---

The :code:`run/` directory contains the scripts to start or test the Blast
application either locally or on GitHub actions.

nginx
-----

The :code:`run/` directory contains the nginx config file which controls how the
application is served.

docs
----

The :code:`docs/` directory contains all the code and text used to make the
documentation (which you are reading right now). If you are making a
documentation addition of change only, you will only edit code is this
directory.

docker
------

The :code:`docker/` directory contains the Docker Compose files used to start
all the services Blast uses (e.g., database, web app, web server). It contains
Docker Compose files for running and testing Blast.

env
---

The :code:`env/` directory contains all the .env files which have all the Blast
settings which we don't want to make visible in source control (e.g., database
login details).

data
----

The :code:`data/` directory contains all the data Blast needs to operate. When
the application is running this is the default location where all data is stored
(e.g., database files and cutouts images). In source control, there is critical
data (transmission curves) and some example data (example cutout images) used
to populate offline local versions of Blast during development.

.github
-------

The :code:`.github/` directory contains the files and templates used for feature
request and bug reports. It also contains workflows for continuous integration
testing.

app
---

The :code:`app/` directory contains all the Blast Django app source code.
:code:`app/Dockerfile` tells Docker how to build the Blast container.
:code:`app/requirements.txt` contains all the Blast package dependencies. We
provide a more detailed breakdown below of the sub directories below.

entrypoints
+++++++++++

The :code:`app/entrypoints` directory contains the entrypoint scripts used to
start the Blast application and related services. These scripts are used by the
Docker Compose files. This directory also contains utility scripts to clean data
directories when restarting Blast.

app
+++

The :code:`app/app/` directory contains the Django level application settings.

host
++++

The :code:`app/host/` directory contains all the Blast source code.

templates
^^^^^^^^^

The :code:`app/host/templates/` directory contains all the html temples used to
render the Blast web pages.

tests
^^^^^

The :code:`app/host/tests/` directory contains all code used to test Blast.

migrations
^^^^^^^^^^

The :code:`app/host/migrations/` directory contains all the Django database
migrations which are applied upon startup of Blast.
