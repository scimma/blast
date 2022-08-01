Documentation
=============

A great place to start for a new developer on blast is to write or edit
documentation (pages like the one you are reading right now!). The process for
editing or adding documentation relatively straight forward and does not require
you to run the main blast app. All you need is a text editor and git installed.

The blast documentation is written in `sphinx <https://www.sphinx-doc.org/en/master/#user-guides>`_
and is built and hosted automatically using `read the docs <https://readthedocs.org/>`_.

All the documentation code and text is contained within :code:`docs/`. Once you
have made changes to the documentation you can preview those changes by running,

.. code:: none

    bash run/blast.run.sh docs

Then open :code:`blast/docs/build/index.html` in your web browser to see the
changes. Everytime you make changes to the documentation code you have to re-run
the above command.

As well as viewing changes locally, once you have added or made changes using
the :doc:`developer workflow <dev_workflow>`
and have a draft pull request open, everytime you push changes a preview of
the documentation is available. You can view this preview by clicking here in the
pull request:

.. image:: _static/auto_read_the_docs_build.png
