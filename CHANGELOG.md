# Changelog

All notable changes to the Blast application will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project (mostly) adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.3]

### Changed

- Refactored and revised logic in HostInformation task.
- Use TaskLock for global rate limiting of NED and SDSS queries.
- Fix minor bug in TransientTaskRunner task

## [1.6.2]

### Changed

- Update Prost host galaxy matcher to v1.2.13.

## [1.6.1]

### Changed

- Added plaintext email for support requests or contacts.
- Disabled celery-beat when running in batch mode.

## [1.6.0]

### Changed

- Revamped the "Add Transient" web form:
  - Improved CSV handling to handle whitespace surrounding values.
  - Changed the wording from "upload" to "add", "define", and "import" to reflect the two distinct ways users can add transients to the database.
  - Redesigned the form structure to make the two addition methods mutually exclusive and not simultaneously visible.
  - Improved algorithm for reporting when added transients are ignored because they already exist in the database.

## [1.5.7]

### Changed

- Upgraded Django to v5.1.12

## [1.5.6]

### Added

- Added a privacy policy about the usage data collection.
- Ensure usage metric object character fields do not exceed the length limits.

### Fixed

- Fixed two API endpoints that should require authentication: "report" and "resolve" on transient result pages.

## [1.5.5]

### Added

- Added a decorator function to capture HTTP requests to specific endpoints as `UsageMetricsLog` objects.
- Added a periodic task `UsageLogRoller` to roll the usage logs by exporting logs to archive files uploaded to the object store and pruning the Django database table.

## [1.5.4]

### Added

- Added a new "garbage collector" periodic task that purges the `cutout_cdn` and `sed_output` scratch space of orphaned data.

## [1.5.3] - 2025-07-07

### Fixed

- Fixed a bug introduced in v1.5.2 when loading some transient result pages.

## [1.5.2] - 2025-07-03

### Fixed

- Fixed a bug when loading result pages, where the page would fail with a "Server Error (500)" message if there were missing data files expected by the page renderer. These errors are now handled more gracefully by rendering empty plots and a message sayingt o reprocess the transient.

### Changed

- Restructured the navigation bar to separate the user account interface and ancillary information from application tools. Links to documentation, source code repo, and acknowledgements were moved to a dropdown menu.
- Improved the responsiveness of the web pages and support for browser scaling.
- Retrigger and reprocess buttons are now displayed on the result pages based on user permissions.

## [1.5.1] - 2025-07-02

### Fixed

- The static landing page was not populating the support email address in the email template defined by the `mailto` link in the footer. Instead of supplying the value via context processor, it was supplied via template tag such that the information is available to the periodic rendering function `update_home_page_statistics()`.

## [1.5.0] - 2025-07-01

### Added

- Added support for OIDC-based authentication and role-based access control (RBAC). Anyone can authenticate but only authorized users are allowed to use protected API functions such as transient upload, reprocessing, or retriggering. Django admins can authorize users by granting them the custom permission "Can launch a new transient workflow". The recommended approach is to grant the permission to a group created for the purpose (e.g. "users-default"), and then add users to that group.
- There is a new registration process for new Blast users. Upon login users are redirected to the uploads page, where they encounter a message informing them that they need to request authorization to access this feature by clicking a link that will prepopulate an email using a template. The template instructs the new user to provide their name, and email address (these are not supplied by all identity providers) as well as an explanation for the request.

### Fixed

- Transient result pages indicate when SED data is loading or if that data is unavailable.

## [1.4.1] - 2025-06-13

### Fixed

- Fixed a minor bug in the Blast documentation `developer_guide/dev_system_pages.md` preventing images from being rendered.

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
