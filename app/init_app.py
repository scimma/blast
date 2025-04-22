#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    # The database migrations must be executed first to establish the db schema
    execute_from_command_line(["__main__.py", "migrate"])

    # Create the Django admin account if it does not exist
    with open("entrypoints/setup_superuser.py") as script:
        script_text = script.read()
    execute_from_command_line(["__main__.py", "shell", f"--command={script_text}"])

    # Collect the assets for the static file server
    execute_from_command_line(["__main__.py", "collectstatic", "--noinput"])

    # Create the periodic tasks if they are not already registered
    with open("entrypoints/setup_initial_periodic_tasks.py") as script:
        script_text = script.read()
    execute_from_command_line(["__main__.py", "shell", f"--command={script_text}"])


if __name__ == "__main__":
    main()
