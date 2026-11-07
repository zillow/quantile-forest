:html_theme.sidebar_secondary.remove:

.. _changes:

Release Notes
=============

Version 1.5.0 (released Nov 06, 2026)
-------------------------------------

This version adds support for Python version 3.15 and for missing values, and adds wheels for arm64 Linux. It drops support for Python 3.9 and 3.10 and now requires scikit-learn 1.9 or later. The leaf samples are now weighted by their sample weights, and the default `interpolation` is now `"averaged_inverted_cdf"`, which matches scikit-learn's quantile definition, so predictions can change. The training samples of each leaf are now stored without padding, which can greatly reduce the memory taken by a model and its pickle. The `sparse_pickle` parameter of `fit` is removed. Models pickled by earlier versions still load, but earlier versions cannot load models pickled by this version.

**Breaking changes**

* Weight leaf samples by their sample weights, and default to scikit-learn's quantile definition by `@reidjohnson <https://github.com/reidjohnson>`_ in `#190 <https://github.com/zillow/quantile-forest/pull/190>`_
* Store the training samples of each leaf without padding by `@reidjohnson <https://github.com/reidjohnson>`_ in `#201 <https://github.com/zillow/quantile-forest/pull/201>`_
* Prepare 1.5.0: release notes, nightly scikit-learn tests, cleanups by `@reidjohnson <https://github.com/reidjohnson>`_ in `#202 <https://github.com/zillow/quantile-forest/pull/202>`_

**New features**

* Add arm64 Linux wheel builds to CI/CD by `@bezineb5 <https://github.com/bezineb5>`_ in `#158 <https://github.com/zillow/quantile-forest/pull/158>`_
* Python 3.15 Support by `@reidjohnson <https://github.com/reidjohnson>`_ in `#197 <https://github.com/zillow/quantile-forest/pull/197>`_
* Support missing values as scikit-learn's forests do by `@reidjohnson <https://github.com/reidjohnson>`_ in `#198 <https://github.com/zillow/quantile-forest/pull/198>`_

**Performance**

* Make weighted predictions and proximity counts independent of training size by `@reidjohnson <https://github.com/reidjohnson>`_ in `#179 <https://github.com/zillow/quantile-forest/pull/179>`_
* Speed up out-of-bag sample lookups by `@reidjohnson <https://github.com/reidjohnson>`_ in `#178 <https://github.com/zillow/quantile-forest/pull/178>`_
* Store training targets as an array, so predict doesn't copy them by `@reidjohnson <https://github.com/reidjohnson>`_ in `#184 <https://github.com/zillow/quantile-forest/pull/184>`_
* Compute OOB quantiles without a Python loop over rows by `@reidjohnson <https://github.com/reidjohnson>`_ in `#185 <https://github.com/zillow/quantile-forest/pull/185>`_
* Compute monotonic bounds one tree level at a time by `@reidjohnson <https://github.com/reidjohnson>`_ in `#186 <https://github.com/zillow/quantile-forest/pull/186>`_

**Bug fixes**

* Avoid changing global random state during leaf subsampling by `@Ra1nhy <https://github.com/Ra1nhy>`_ in `#167 <https://github.com/zillow/quantile-forest/pull/167>`_
* Fix sample weight indexing when collecting leaf samples by `@AryanRG <https://github.com/AryanRG>`_ in `#168 <https://github.com/zillow/quantile-forest/pull/168>`_
* Fix float max_samples_leaf in predict and leaf padding by `@reidjohnson <https://github.com/reidjohnson>`_ in `#170 <https://github.com/zillow/quantile-forest/pull/170>`_
* Stop quantile_ranks from resetting fitted feature attributes by `@reidjohnson <https://github.com/reidjohnson>`_ in `#172 <https://github.com/zillow/quantile-forest/pull/172>`_
* Validate sample weights before storing them in fit by `@reidjohnson <https://github.com/reidjohnson>`_ in `#174 <https://github.com/zillow/quantile-forest/pull/174>`_
* Fix proximity_counts argument checks and return types, and docstrings by `@reidjohnson <https://github.com/reidjohnson>`_ in `#177 <https://github.com/zillow/quantile-forest/pull/177>`_
* Fix models loaded from sparse pickles or read-only memory maps by `@reidjohnson <https://github.com/reidjohnson>`_ in `#180 <https://github.com/zillow/quantile-forest/pull/180>`_
* Fix sparse inputs in predict by `@reidjohnson <https://github.com/reidjohnson>`_ in `#173 <https://github.com/zillow/quantile-forest/pull/173>`_
* Validate quantiles as array-likes and reject NaN or empty quantiles by `@reidjohnson <https://github.com/reidjohnson>`_ in `#175 <https://github.com/zillow/quantile-forest/pull/175>`_
* Declare that missing values are not supported, and reject them up front by `@reidjohnson <https://github.com/reidjohnson>`_ in `#176 <https://github.com/zillow/quantile-forest/pull/176>`_
* Fix weighted quantiles that use zero-weight inputs or miss exact ranks by `@reidjohnson <https://github.com/reidjohnson>`_ in `#171 <https://github.com/zillow/quantile-forest/pull/171>`_
* Group leaf samples with a stable sort, so results don't depend on the CPU by `@reidjohnson <https://github.com/reidjohnson>`_ in `#181 <https://github.com/zillow/quantile-forest/pull/181>`_
* Validate inputs once for OOB samples, without a warning per tree by `@reidjohnson <https://github.com/reidjohnson>`_ in `#182 <https://github.com/zillow/quantile-forest/pull/182>`_
* Use the fitted trees, not n_estimators, in predict by `@reidjohnson <https://github.com/reidjohnson>`_ in `#183 <https://github.com/zillow/quantile-forest/pull/183>`_
* Rank among the clipped leaf values in quantile_ranks with monotonic constraints by `@reidjohnson <https://github.com/reidjohnson>`_ in `#188 <https://github.com/zillow/quantile-forest/pull/188>`_
* Fix small issues found in a review of main by `@reidjohnson <https://github.com/reidjohnson>`_ in `#199 <https://github.com/zillow/quantile-forest/pull/199>`_
* Redraw OOB samples as the forest drew them, for older models by `@reidjohnson <https://github.com/reidjohnson>`_ in `#200 <https://github.com/zillow/quantile-forest/pull/200>`_

**Documentation**

* Document that monotonic constraints narrow prediction intervals by `@reidjohnson <https://github.com/reidjohnson>`_ in `#187 <https://github.com/zillow/quantile-forest/pull/187>`_
* Document what proximity counts count by `@reidjohnson <https://github.com/reidjohnson>`_ in `#189 <https://github.com/zillow/quantile-forest/pull/189>`_
* Fix the gallery examples, and add an example of monotonic constraints by `@reidjohnson <https://github.com/reidjohnson>`_ in `#191 <https://github.com/zillow/quantile-forest/pull/191>`_
* Add a QRF vs. gradient boosting example and tidy up the gallery by `@reidjohnson <https://github.com/reidjohnson>`_ in `#194 <https://github.com/zillow/quantile-forest/pull/194>`_
* Add gallery sliders by `@reidjohnson <https://github.com/reidjohnson>`_ in `#195 <https://github.com/zillow/quantile-forest/pull/195>`_
* Add docs navigation by `@reidjohnson <https://github.com/reidjohnson>`_ in `#196 <https://github.com/zillow/quantile-forest/pull/196>`_

**Maintenance**

* Bump actions/checkout from 6 to 7 by `@dependabot <https://github.com/dependabot>`_ in `#157 <https://github.com/zillow/quantile-forest/pull/157>`_
* Bump docker/setup-qemu-action from 3 to 4 by `@dependabot <https://github.com/dependabot>`_ in `#161 <https://github.com/zillow/quantile-forest/pull/161>`_
* Bump actions/setup-python from 6 to 7 by `@dependabot <https://github.com/dependabot>`_ in `#159 <https://github.com/zillow/quantile-forest/pull/159>`_
* Bump pypa/gh-action-pypi-publish from 1.14.0 to 1.14.1 by `@dependabot <https://github.com/dependabot>`_ in `#160 <https://github.com/zillow/quantile-forest/pull/160>`_
* Bump pypa/gh-action-pypi-publish from 1.14.1 to 1.14.2 by `@dependabot <https://github.com/dependabot>`_ in `#163 <https://github.com/zillow/quantile-forest/pull/163>`_
* Remove inline from pyx by `@reidjohnson <https://github.com/reidjohnson>`_ in `#169 <https://github.com/zillow/quantile-forest/pull/169>`_
* Bump pypa/cibuildwheel from 4.1.0 to 4.3.4 by `@dependabot <https://github.com/dependabot>`_ in `#166 <https://github.com/zillow/quantile-forest/pull/166>`_
* Bump pypa/cibuildwheel from 4.3.4 to 4.3.5 by `@dependabot <https://github.com/dependabot>`_ in `#192 <https://github.com/zillow/quantile-forest/pull/192>`_
* Bump pypa/gh-action-pypi-publish from 1.14.2 to 1.14.3 by `@dependabot <https://github.com/dependabot>`_ in `#193 <https://github.com/zillow/quantile-forest/pull/193>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.4.2...v1.5.0

Version 1.4.2 (released Jun 21, 2026)
-------------------------------------

* Update Precommit Hooks by `@reidjohnson <https://github.com/reidjohnson>`_ in `#140 <https://github.com/zillow/quantile-forest/pull/140>`_
* Bump actions/upload-artifact from 4 to 5 by `@dependabot <https://github.com/dependabot>`_ in `#136 <https://github.com/zillow/quantile-forest/pull/136>`_
* Bump actions/download-artifact from 5 to 6 by `@dependabot <https://github.com/dependabot>`_ in `#137 <https://github.com/zillow/quantile-forest/pull/137>`_
* Bump pypa/cibuildwheel from 3.1.4 to 3.3.0 by `@dependabot <https://github.com/dependabot>`_ in `#139 <https://github.com/zillow/quantile-forest/pull/139>`_
* Bump actions/checkout from 5 to 6 by `@dependabot <https://github.com/dependabot>`_ in `#141 <https://github.com/zillow/quantile-forest/pull/141>`_
* Python 3.14 Support by `@reidjohnson <https://github.com/reidjohnson>`_ in `#133 <https://github.com/zillow/quantile-forest/pull/133>`_
* Bump actions/upload-artifact from 5 to 6 by `@dependabot <https://github.com/dependabot>`_ in `#143 <https://github.com/zillow/quantile-forest/pull/143>`_
* Bump actions/download-artifact from 6 to 7 by `@dependabot <https://github.com/dependabot>`_ in `#142 <https://github.com/zillow/quantile-forest/pull/142>`_
* Bump pypa/cibuildwheel from 3.3.0 to 3.3.1 by `@dependabot <https://github.com/dependabot>`_ in `#144 <https://github.com/zillow/quantile-forest/pull/144>`_
* Bump actions/upload-artifact from 6 to 7 by `@dependabot <https://github.com/dependabot>`_ in `#145 <https://github.com/zillow/quantile-forest/pull/145>`_
* Bump actions/download-artifact from 7 to 8 by `@dependabot <https://github.com/dependabot>`_ in `#146 <https://github.com/zillow/quantile-forest/pull/146>`_
* Bump codecov/codecov-action from 5 to 6 by `@dependabot <https://github.com/dependabot>`_ in `#148 <https://github.com/zillow/quantile-forest/pull/148>`_
* Bump pypa/cibuildwheel from 3.3.1 to 3.4.1 by `@dependabot <https://github.com/dependabot>`_ in `#149 <https://github.com/zillow/quantile-forest/pull/149>`_
* Bump pypa/gh-action-pypi-publish from 1.13.0 to 1.14.0 by `@dependabot <https://github.com/dependabot>`_ in `#150 <https://github.com/zillow/quantile-forest/pull/150>`_
* Support scikit-learn 1.9 (DTYPE removed from sklearn.tree._tree) by `@MaxGhenis <https://github.com/MaxGhenis>`_ in `#152 <https://github.com/zillow/quantile-forest/pull/152>`_
* Fix sample weight with sklearn 1.9 by `@reidjohnson <https://github.com/reidjohnson>`_ in `#153 <https://github.com/zillow/quantile-forest/pull/153>`_
* Bump pypa/cibuildwheel from 3.4.1 to 4.1.0 by `@dependabot <https://github.com/dependabot>`_ in `#155 <https://github.com/zillow/quantile-forest/pull/155>`_
* Bump codecov/codecov-action from 6 to 7 by `@dependabot <https://github.com/dependabot>`_ in `#154 <https://github.com/zillow/quantile-forest/pull/154>`_
* Resolve Altair build warnings by `@reidjohnson <https://github.com/reidjohnson>`_ in `#156 <https://github.com/zillow/quantile-forest/pull/156>`_

This version adds support for Python version 3.14 and scikit-learn 1.9.

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.4.1...v1.4.2

Version 1.4.1 (released Sep 10, 2025)
-------------------------------------

* Use uv for build by `@reidjohnson <https://github.com/reidjohnson>`_ in `#113 <https://github.com/zillow/quantile-forest/pull/113>`_
* Update docs by `@reidjohnson <https://github.com/reidjohnson>`_ in `#114 <https://github.com/zillow/quantile-forest/pull/114>`_
* Bump actions/checkout from 4 to 5 by `@dependabot <https://github.com/dependabot>`_ in `#121 <https://github.com/zillow/quantile-forest/pull/121>`_
* Bump actions/download-artifact from 4 to 5 by `@dependabot <https://github.com/dependabot>`_ in `#122 <https://github.com/zillow/quantile-forest/pull/122>`_
* Bump pypa/cibuildwheel from 3.1.3 to 3.1.4 by `@dependabot <https://github.com/dependabot>`_ in `#123 <https://github.com/zillow/quantile-forest/pull/123>`_
* Remove California dataset in tests by `@reidjohnson <https://github.com/reidjohnson>`_ in `#127 <https://github.com/zillow/quantile-forest/pull/127>`_
* Bump pypa/gh-action-pypi-publish from 1.12.4 to 1.13.0 by `@dependabot <https://github.com/dependabot>`_ in `#125 <https://github.com/zillow/quantile-forest/pull/125>`_
* Bump actions/setup-python from 5 to 6 by `@dependabot <https://github.com/dependabot>`_ in `#126 <https://github.com/zillow/quantile-forest/pull/126>`_
* Add NaN check and test by `@reidjohnson <https://github.com/reidjohnson>`_ in `#129 <https://github.com/zillow/quantile-forest/pull/129>`_
* Add `sample_weight` to tests by `@reidjohnson <https://github.com/reidjohnson>`_ in `#130 <https://github.com/zillow/quantile-forest/pull/130>`_
* Update build files for warnings by `@reidjohnson <https://github.com/reidjohnson>`_ in `#131 <https://github.com/zillow/quantile-forest/pull/131>`_
* Update docs for package compatibility by `@reidjohnson <https://github.com/reidjohnson>`_ in `#132 <https://github.com/zillow/quantile-forest/pull/132>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.4.0...v1.4.1

Version 1.4.0 (released Jan 21, 2025)
-------------------------------------

* Upgrade minimum sklearn by `@brentonmallen1 <https://github.com/brentonmallen1>`_ in `#92 <https://github.com/zillow/quantile-forest/pull/92>`_
* Refactor usage of scikit-learn utilities by `@reidjohnson <https://github.com/reidjohnson>`_ in `#95 <https://github.com/zillow/quantile-forest/pull/95>`_
* Update contributor documentation by `@reidjohnson <https://github.com/reidjohnson>`_ in `#96 <https://github.com/zillow/quantile-forest/pull/96>`_
* Bump pypa/gh-action-pypi-publish from 1.10.1 to 1.10.2 by `@dependabot <https://github.com/dependabot>`_ in `#93 <https://github.com/zillow/quantile-forest/pull/93>`_
* Bump pypa/cibuildwheel from 2.20.0 to 2.21.1 by `@dependabot <https://github.com/dependabot>`_ in `#94 <https://github.com/zillow/quantile-forest/pull/94>`_
* Update typing for Python 3.9+ by `@reidjohnson <https://github.com/reidjohnson>`_ in `#97 <https://github.com/zillow/quantile-forest/pull/97>`_
* Bump pypa/cibuildwheel from 2.21.1 to 2.21.2 by `@dependabot <https://github.com/dependabot>`_ in `#98 <https://github.com/zillow/quantile-forest/pull/98>`_
* Bump pypa/gh-action-pypi-publish from 1.10.2 to 1.10.3 by `@dependabot <https://github.com/dependabot>`_ in `#99 <https://github.com/zillow/quantile-forest/pull/99>`_
* Bump pypa/cibuildwheel from 2.21.2 to 2.21.3 by `@dependabot <https://github.com/dependabot>`_ in `#100 <https://github.com/zillow/quantile-forest/pull/100>`_
* Add Conda instructions to docs by `@reidjohnson <https://github.com/reidjohnson>`_ in `#101 <https://github.com/zillow/quantile-forest/pull/101>`_
* Bump pypa/gh-action-pypi-publish from 1.10.3 to 1.11.0 by `@dependabot <https://github.com/dependabot>`_ in `#102 <https://github.com/zillow/quantile-forest/pull/102>`_
* Bump pypa/gh-action-pypi-publish from 1.11.0 to 1.12.2 by `@dependabot <https://github.com/dependabot>`_ in `#104 <https://github.com/zillow/quantile-forest/pull/104>`_
* scikit-learn 1.6 compatibility by `@reidjohnson <https://github.com/reidjohnson>`_ in `#105 <https://github.com/zillow/quantile-forest/pull/105>`_
* Bump codecov/codecov-action from 4 to 5 by `@dependabot <https://github.com/dependabot>`_ in `#106 <https://github.com/zillow/quantile-forest/pull/106>`_
* Bump pypa/cibuildwheel from 2.21.3 to 2.22.0 by `@dependabot <https://github.com/dependabot>`_ in `#107 <https://github.com/zillow/quantile-forest/pull/107>`_
* Bump pypa/gh-action-pypi-publish from 1.12.2 to 1.12.3 by `@dependabot <https://github.com/dependabot>`_ in `#108 <https://github.com/zillow/quantile-forest/pull/108>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.3.11...v1.4.0

Version 1.3.11 (released Sep 10, 2024)
--------------------------------------

* Refactor leaf mapping by `@reidjohnson <https://github.com/reidjohnson>`_ in `#80 <https://github.com/zillow/quantile-forest/pull/80>`_
* Update build linting by `@reidjohnson <https://github.com/reidjohnson>`_ in `#81 <https://github.com/zillow/quantile-forest/pull/81>`_
* Check `monotonic_cst` before fit by `@reidjohnson <https://github.com/reidjohnson>`_ in `#82 <https://github.com/zillow/quantile-forest/pull/82>`_
* Bump pypa/gh-action-pypi-publish from 1.9.0 to 1.10.0 by `@dependabot <https://github.com/dependabot>`_ in `#83 <https://github.com/zillow/quantile-forest/pull/83>`_
* Fix unaggregated quantile ranks by `@reidjohnson <https://github.com/reidjohnson>`_ in `#84 <https://github.com/zillow/quantile-forest/pull/84>`_
* Fix Cython build warnings by `@reidjohnson <https://github.com/reidjohnson>`_ in `#86 <https://github.com/zillow/quantile-forest/pull/86>`_
* Bump pypa/gh-action-pypi-publish from 1.10.0 to 1.10.1 by `@dependabot <https://github.com/dependabot>`_ in `#85 <https://github.com/zillow/quantile-forest/pull/85>`_
* Fix multi-target serialization by `@reidjohnson <https://github.com/reidjohnson>`_ in `#88 <https://github.com/zillow/quantile-forest/pull/88>`_
* Fix doc tests by `@reidjohnson <https://github.com/reidjohnson>`_ in `#89 <https://github.com/zillow/quantile-forest/pull/89>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.3.10...v1.3.11

Version 1.3.10 (released Aug 31, 2024)
--------------------------------------

* Optimize leaf node mapping by `@reidjohnson <https://github.com/reidjohnson>`_ in `#78 <https://github.com/zillow/quantile-forest/pull/78>`_
* Refactor helper functions by `@reidjohnson <https://github.com/reidjohnson>`_ in `#79 <https://github.com/zillow/quantile-forest/pull/79>`_
* Update examples, documentation, and build

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.3.9...v1.3.10

Version 1.3.9 (released Aug 23, 2024)
-------------------------------------

* Fix reproduction of scikit-learn trees with MAE criterion by `@reidjohnson <https://github.com/reidjohnson>`_ in `#77 <https://github.com/zillow/quantile-forest/pull/77>`_
* Update documentation and examples

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.3.8...v1.3.9

Version 1.3.8 (released Aug 15, 2024)
-------------------------------------

* Bump pypa/cibuildwheel from 2.19.1 to 2.19.2 by `@dependabot <https://github.com/dependabot>`_ in `#63 <https://github.com/zillow/quantile-forest/pull/63>`_
* Update interpolation docstring by `@reidjohnson <https://github.com/reidjohnson>`_ in `#64 <https://github.com/zillow/quantile-forest/pull/64>`_
* Add Hugging Face Hub example by `@reidjohnson <https://github.com/reidjohnson>`_ in `#65 <https://github.com/zillow/quantile-forest/pull/65>`_
* Fix doc warning by `@reidjohnson <https://github.com/reidjohnson>`_ in `#66 <https://github.com/zillow/quantile-forest/pull/66>`_
* Add Tree SHAP example by `@reidjohnson <https://github.com/reidjohnson>`_ in `#67 <https://github.com/zillow/quantile-forest/pull/67>`_
* Update example plots by `@reidjohnson <https://github.com/reidjohnson>`_ in `#68 <https://github.com/zillow/quantile-forest/pull/68>`_
* Add proximity counts example by `@reidjohnson <https://github.com/reidjohnson>`_ in `#70 <https://github.com/zillow/quantile-forest/pull/70>`_
* Add quantile ranks example by `@reidjohnson <https://github.com/reidjohnson>`_ in `#69 <https://github.com/zillow/quantile-forest/pull/69>`_
* Fix feature name warning by `@reidjohnson <https://github.com/reidjohnson>`_ in `#71 <https://github.com/zillow/quantile-forest/pull/71>`_
* Bump pypa/cibuildwheel from 2.19.2 to 2.20.0 by `@dependabot <https://github.com/dependabot>`_ in `#72 <https://github.com/zillow/quantile-forest/pull/72>`_
* Fix for `monotonic_cst`  by `@reidjohnson <https://github.com/reidjohnson>`_ in `#74 <https://github.com/zillow/quantile-forest/pull/74>`_
* Fix monotonic constraint performance by `@reidjohnson <https://github.com/reidjohnson>`_ in `#75 <https://github.com/zillow/quantile-forest/pull/75>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.3.7...v1.3.8

Version 1.3.7 (released Jun 19, 2024)
-------------------------------------

* Compatibility for NumPy 2.0 by `@reidjohnson <https://github.com/reidjohnson>`_ in `#59 <https://github.com/zillow/quantile-forest/pull/59>`_
* Bump pypa/gh-action-pypi-publish from 1.8.14 to 1.9.0 by `@dependabot <https://github.com/dependabot>`_ in `#56 <https://github.com/zillow/quantile-forest/pull/56>`_
* Bump pypa/cibuildwheel from 2.18.1 to 2.19.1 by `@dependabot <https://github.com/dependabot>`_ in `#57 <https://github.com/zillow/quantile-forest/pull/57>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.3.6...v1.3.7

Version 1.3.6 (released May 22, 2024)
-------------------------------------

* Fix import of scikit-learn 1.5 `parse_version` by `@reidjohnson <https://github.com/reidjohnson>`_ in `#52 <https://github.com/zillow/quantile-forest/pull/52>`_
* Update pre-commit hooks by `@reidjohnson <https://github.com/reidjohnson>`_ in `#53 <https://github.com/zillow/quantile-forest/pull/53>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.3.5...v1.3.6

Version 1.3.5 (released Apr 15, 2024)
-------------------------------------

* Fix fractional `max_samples` by `@reidjohnson <https://github.com/reidjohnson>`_ in `#47 <https://github.com/zillow/quantile-forest/pull/47>`_
* Add support for `monotonic_cst` by `@reidjohnson <https://github.com/reidjohnson>`_ in `#49 <https://github.com/zillow/quantile-forest/pull/49>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.3.4...v1.3.5

Version 1.3.4 (released Feb 21, 2024)
-------------------------------------

* Reorder multi-target outputs by `@reidjohnson <https://github.com/reidjohnson>`_ in `#35 <https://github.com/zillow/quantile-forest/pull/35>`_
* Add tests for model serialization by `@reidjohnson <https://github.com/reidjohnson>`_ in `#36 <https://github.com/zillow/quantile-forest/pull/36>`_
* Update and fix documentation and examples

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.3.3...v1.3.4

Version 1.3.3 (released Feb 16, 2024)
-------------------------------------

* Set default value of `weighted_leaves` at prediction time to False by `@reidjohnson <https://github.com/reidjohnson>`_ in `#34 <https://github.com/zillow/quantile-forest/pull/34>`_
* Update and fix documentation and examples

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.3.2...v1.3.3

Version 1.3.2 (released Feb 15, 2024)
-------------------------------------

* Fix bug in multi-target output when `max_samples_leaf` > 1 by `@reidjohnson <https://github.com/reidjohnson>`_ in `#30 <https://github.com/zillow/quantile-forest/pull/30>`_
* Update quantile forest examples by `@reidjohnson <https://github.com/reidjohnson>`_ in `#31 <https://github.com/zillow/quantile-forest/pull/31>`_
* Update and fix documentation by `@reidjohnson <https://github.com/reidjohnson>`_ in `#33 <https://github.com/zillow/quantile-forest/pull/33>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.3.1...v1.3.2

Version 1.3.1 (released Feb 12, 2024)
-------------------------------------

* Fix single-output performance regression by `@reidjohnson <https://github.com/reidjohnson>`_ in `#29 <https://github.com/zillow/quantile-forest/pull/29>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.3.0...v1.3.1

Version 1.3.0 (released Feb 11, 2024)
-------------------------------------

* Support for multiple-output quantile regression by `@reidjohnson <https://github.com/reidjohnson>`_ in `#26 <https://github.com/zillow/quantile-forest/pull/26>`_
* Update conformalized quantile regression example by `@reidjohnson <https://github.com/reidjohnson>`_ in `#28 <https://github.com/zillow/quantile-forest/pull/28>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.2.5...v1.3.0

Version 1.2.5 (released Feb 10, 2024)
-------------------------------------

* Fix weighted leaf and quantile bug by `@reidjohnson <https://github.com/reidjohnson>`_ in `#27 <https://github.com/zillow/quantile-forest/pull/27>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.2.4...v1.2.5

Version 1.2.4 (released Jan 16, 2024)
-------------------------------------

* Use base model parameter validation when available
* Resolve Cython 3 deprecation warnings

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.2.3...v1.2.4

Version 1.2.3 (released Oct 09, 2023)
-------------------------------------

* Fix bug that could prevent interpolation from being correctly applied by `@reidjohnson <https://github.com/reidjohnson>`_ in `#15 <https://github.com/zillow/quantile-forest/pull/15>`_
* Update documentation and docstrings

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.2.2...v1.2.3

Version 1.2.2 (released Oct 08, 2023)
-------------------------------------

* Optimize performance for predictions when `max_samples_leaf` = 1 by `@reidjohnson <https://github.com/reidjohnson>`_ in `#13 <https://github.com/zillow/quantile-forest/pull/13>`_
* Update documentation and examples by `@reidjohnson <https://github.com/reidjohnson>`_ in `#14 <https://github.com/zillow/quantile-forest/pull/14>`_

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.2.1...v1.2.2

Version 1.2.1 (released Oct 04, 2023)
-------------------------------------

* More efficient calculation of weighted quantiles by `@reidjohnson <https://github.com/reidjohnson>`_ in `#11 <https://github.com/zillow/quantile-forest/pull/11>`_
* Add support for Python version 3.12

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.2.0...v1.2.1

Version 1.2.0 (released Aug 01, 2023)
-------------------------------------

* Add optional `default_quantiles` parameter to the model initialization
* Update documentation

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.1.3...v1.2.0

Version 1.1.3 (released Jul 08, 2023)
-------------------------------------

* Fix building from the source distribution
* Minor update to documentation

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.1.2...v1.1.3

Version 1.1.2 (released Mar 22, 2023)
-------------------------------------

* Fix for compatibility with development version of scikit-learn
* Update documentation and examples

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.1.1...v1.1.2

Version 1.1.1 (released Dec 19, 2022)
-------------------------------------

* Fix for compatibility with scikit-learn 1.2.0
* Fix to documentation
* Update version requirements

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.1.0...v1.1.1

Version 1.1.0 (released Nov 07, 2022)
-------------------------------------

* Update default `max_samples_leaf` to 1 (previously None)
* Update documentation and unit tests
* Miscellaneous update for compatibility with scikit-learn >= 1.1.0

This version supports Python versions 3.8 to 3.11. Note that support for 32-bit Python on Windows has been dropped in this release.

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.0.2...v1.1.0

Version 1.0.2 (released Mar 28, 2022)
-------------------------------------

* Add sample weighting by leaf size

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.0.1...v1.0.2

Version 1.0.1 (released Mar 23, 2022)
-------------------------------------

* Suppress UserWarning

**Full Changelog**: https://github.com/zillow/quantile-forest/compare/v1.0.0...v1.0.1

Version 1.0.0 (released Mar 23, 2022)
-------------------------------------

Initial release.
