Blast system pages
==================

In addition to the public html pages Blast serves it maybe useful in the
development cycle to access some of Blast's system / administrator pages.

Django admin dashboard
----------------------

The Blast Django admin site (see the docs
`here <https://docs.djangoproject.com/en/4.0/ref/contrib/admin/>`_) allows you
to view, edit, and launch periodic tasks, and add data to the database through
a web interface.

Once Blast is running locally to see the Django admin dashboard go to `http://localhost:8000/ <http://localhost:8000/>`_ where you will be prompted for a login. The login user and
password are set by :code:`DJANGO_SUPERUSER_PASSWORD`
:code:`DJANGO_SUPERUSER_USERNAME` defined in your :code:`env/.env.dev` file.

Once logged in, you should see a page like this:

.. image::../_static/django_admin_screenshot.png

.. note::

    To see the following system pages you will need to be running the full
    Blast stack with :code:`bash run/blastctl full_dev up`, and not the slim version.

Flower
------

The Flower dashboard (see the docs `here <https://flower.readthedocs.io/en/latest/>`_)
allows you to monitor the backend computation tasks being run in Blast. This allows
you to see which tasks are being run and which tasks are failing.

Once Blast is running locally to see the Flower dashboard go to `<localhost:8888>`_.

.. image::../_static/flower_dashboard.png


RabbitMQ
--------

The RabbitMQ management dashboard (see the docs `here <https://www.rabbitmq.com/documentation.html>`_)
allows you to see the message broker traffic where Blast computation tasks are
sent to workers.

Once Blast is running locally to see the RabbitMQ management dashboard go
to `<localhost:15672>`_. where you will be prompted for a login. The login user and
password are set by :code:`RABBITMQ_USERNAME`
:code:`RABBITMQ_PASSWORD` defined in your :code:`env/.env.dev` file.

Once logged in, you see a page like this:

.. image::../_static/rabbitmq_screenshot.png
