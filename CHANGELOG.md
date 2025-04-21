# Changelog

All notable changes to the Blast application will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project (mostly) adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2025-04-21

### Changed

- The Python package "GHOST" that was used for associating transients with a host galaxy was replaced with its successor "Prost", a library by the same author with improved accuracy.

### Fixed

- There was an error related to the angle value in `/app/host/plotting_utils.py`, where the reference to `theta_rad` in `plot_aperture()` was preventing transient result pages from rendering due to an invalid value type. It was fixed by referencing `theta_rad.value` instead.
