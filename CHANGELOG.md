# Changelog

All notable changes to the Blast application will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project (mostly) adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2025-06-13

### Added

- Improved workflow execution speed by using ThreadPoolExecutor to download cutouts in parallel.
- Added a graphical SVG-based view of the workflow and task status to the transient result pages.

### Changed

- The Python package "GHOST" that was used for associating transients with a host galaxy was replaced with its successor "Prost", a library by the same author with improved accuracy.
- When workflows are manually retriggered via the `retrigger_transient/[transient]` API endpoint, failed tasks are retried. The automated retriggering periodic task does not retry failed tasks.
- Consolidated object storage interface into a unified class used for data initialization and generated data storage.
- Improved the workflow initialization algorithm to prune redundant TaskRegister objects and handle the addition of new tasks to the transient workflow.
- Improved the accuracy of the workflow progress calculation with a new algorithm. This change necessitated a revision of the task prerequisites to more accurately reflect the actual workflow definition used by the Celery Canvas to execute tasks.
- Consolidated, minimized, and updated the versions of the set of Python package dependencies in `requirements.txt`.
- Update the MariaDB version for the Django database used in the Docker Compose deployment from v10 to v11.1.

### Fixed

- Secured two API endpoints that should require authentication to run:  `reprocess_transient/[transient]`, `retrigger_transient/[transient]`.
- Fixed a bug in the aperture construction unit test to support downloading cutouts for a subset of filters.
- Implemented a file locking mechanism to prevent concurrent processes from deleting data files prematurely in the `GlobalApertureConstruction`, `GlobalAperturePhotometry`, and `LocalAperturePhotometry` workflow tasks.
- Updated the NGINX proxy configuration in the Docker Compose deployment to fix problems with localhost routing in the development environment.

## [1.3.0] - 2025-05-09

### Added

- Implemented an automated retriggering mechanism executed periodically by Celery Beat to resume aborted workflows.

### Fixed

- There was an error related to the angle value in `/app/host/plotting_utils.py`, where the reference to `theta_rad` in `plot_aperture()` was preventing transient result pages from rendering due to an invalid value type. It was fixed by referencing `theta_rad.value` instead.
