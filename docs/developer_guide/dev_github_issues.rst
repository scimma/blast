What to work on?
================

So you've read the Blast developer guide, now you want to know what exactly to
work on. The best place to start is the
`issues <https://github.com/scimma/blast/issues>`_ tab on the Blast GitHub
page. Here you will find a list of develop tasks that need to be worked on for
Blast. They are tagged by level of difficulty:

Entry level issues
------------------

These issues are tagged with :code:`Difficulty: entry level`. These issues are
self-contained short tasks that often require only editing html or documentation
rst files. These issues will often have specific instructions on how to complete
them. These issues are a great place to start for new developers.

Intermediate issues
-------------------

These issues are tagged with :code:`Difficulty: intermediate`. These issues are
slightly longer tasks which require editing Python code with Blast. You may have
to write tests to get work on these issues through code review. The issue will
often have some direction on how to complete the feature request, but you will
have to figure out specific implementation details. These issues are great for
developers who have some experience with Python and / or have completed
a few entry level issues.

Advanced issues
---------------

These issues are tagged with :code:`Difficulty: advanced`. These issues are
open-ended tasks which often require knowledge of all aspects of the application,
or involve project level structural changes. These issues are great for
developers who are confident with Python and Django and have completed
many intermediate level issues.

Documentation
-------------

Any issues tagged with :code:`Documentation` only requires editing
documentation, you don't even need to run the Blast app locally to work on these.
See the :doc:`Documentation <dev_documentation>` developer guide docs to get started.

Slim stack
----------

An issues tagged with :code:`Slim stack` only requires the Blast web app and
database, so you can work on these issues by running :code:`bash run/blastctl slim_dev up`
See :doc:`Running Blast <dev_running_blast>` documentation for more details.

Full stack
----------

An issue tagged with :code:`Full stack` requires all the Blast services to
be worked on which means you will need to run :code:`bash run/blastctl full_dev up`
to work on them. See :doc:`Running Blast <dev_running_blast>` documentation
for more details.





You are now ready to get stuck in, pick an issue and good luck! :-)
