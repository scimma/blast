FAQs
====


Docker
------

1. I'm getting the error :code:`unknown flag: --profile` when I try and start
the docker containers.

You need to update your docker version to a more recent one which supports
--profiles flag. You need Docker Compose version 1.28.0 or later. If you are
having trouble installing the required version of docker, there is still support
for the old version.

For the slim stack:

.. code:: none

    bash run/blast.run.sh old_slim_dev

or if you want the full stack:

.. code:: none

    bash run/blast.run.sh old_full_dev

To run tests,

.. code:: none

    bash run/blast.run.sh old_test

And finally, to build the docs locally,

.. code:: none

    bash run/blast.run.sh old_docs

But be warned, I will remove this at some point. So getting the most recent
version of docker installed is the best solution.
