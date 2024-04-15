:html_theme.sidebar_secondary.remove:

.. _changes:

Release Notes
=============

Version 1.3.5 (released Apr 15, 2024)
-------------------------------------

- Fixes fractional `max_samples` (#47)
- Adds support for `monotonic_cst` (#49)

Version 1.3.4 (released Feb 21, 2024)
-------------------------------------

- Reorder multi-target outputs (#35)
- Add tests for model serialization (#36)
- Update and fix documentation and examples

Version 1.3.3 (released Feb 16, 2024)
-------------------------------------

- Set default value of `weighted_leaves` at prediction time to False (#34)
- Update and fix documentation and examples

Version 1.3.2 (released Feb 15, 2024)
-------------------------------------

- Fix bug in multi-target output when `max_samples_leaf` > 1 (#30)
- Update quantile forest examples (#31)
- Update and fix documentation (#33)

Version 1.3.1 (released Feb 12, 2024)
-------------------------------------

- Fix single-output performance regression (#29)

Version 1.3.0 (released Feb 11, 2024)
-------------------------------------

- Support for multiple-output quantile regression (#26)
- Update conformalized quantile regression example (#28)

Version 1.2.5 (released Feb 10, 2024)
-------------------------------------

- Fix weighted leaf and quantile bug (#27)

Version 1.2.4 (released Jan 16, 2024)
-------------------------------------

- Use base model parameter validation when available
- Resolve Cython 3 deprecation warnings

Version 1.2.3 (released Oct 09, 2023)
-------------------------------------

- Fix bug that could prevent interpolation from being correctly applied (#15)
- Update documentation and docstrings

Version 1.2.2 (released Oct 08, 2023)
-------------------------------------

- Optimize performance for predictions when `max_samples_leaf` = 1 (#13)
- Update documentation and examples (#14)

Version 1.2.1 (released Oct 04, 2023)
-------------------------------------

- More efficient calculation of weighted quantiles (#11)
- Add support for Python version 3.12

Version 1.2.0 (released Aug 01, 2023)
-------------------------------------

- Add optional default_quantiles parameter to the model initialization
- Update documentation

Version 1.1.3 (released Jul 08, 2023)
-------------------------------------

- Fix building from the source distribution
- Minor update to documentation

Version 1.1.2 (released Mar 22, 2023)
-------------------------------------

- Fix for compatibility with development version of scikit-learn
- Update documentation and examples

Version 1.1.1 (released Dec 19, 2022)
-------------------------------------

- Fix for compatibility with scikit-learn 1.2.0
- Fix to documentation
- Update version requirements

Version 1.1.0 (released Nov 07, 2022)
-------------------------------------

- Update default `max_samples_leaf` to 1 (previously None)
- Update documentation and unit tests
- Miscellaneous update for compatibility with scikit-learn >= 1.1.0

This version supports Python versions 3.8 to 3.11. Note that support for 32-bit Python on Windows has been dropped in this release.

Version 1.0.2 (released Mar 28, 2022)
-------------------------------------

- Add sample weighting by leaf size

Version 1.0.1 (released Mar 23, 2022)
-------------------------------------

- Suppresses UserWarning

Version 1.0.0 (released Mar 23, 2022)
-------------------------------------

Initial release.
