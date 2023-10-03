---
title: 'quantile-forest: A Python Package for Quantile Regression Forests'
tags:
  - Python
  - Machine Learning
  - Quantile Regression
authors:
  - name: Reid A. Johnson
    orcid: 0009-0001-1449-4940
    affiliation: 1
affiliations:
 - name: Zillow Group, USA
   index: 1
date: 1 October 2023
bibliography: paper.bib
---

# Summary

Quantile regression forests (QRF) is a non-parametric, tree-based ensemble method for estimating conditional quantiles [@meinshausen2006quantile]. It is a generalization of the random forests algorithm, a versatile ensemble learning algorithm, originally proposed in [@breiman2001random], that has proven extremely popular and useful as a general-purpose machine learning method [@biau2016random;@hengl2018random;@wager2018estimation;@athey2019generalized]. Instead of outputting the weighted mean value of training labels for regression like random forests, QRF employ the weighted empirical distribution of training labels to obtain the predictive distribution. This enables QRF to output probabilistic predictions for regression problems, which are widely useful for constructing estimates of uncertainty [@petropoulos2022forecasting].

`quantile-forest` provides a fast, feature-rich implementation of the QRF algorithm. The estimators provided in this package are optimized using Cython [@behnel2010cython] for training and inference speed, and can estimate arbitrary quantiles at prediction time without retraining. They provide methods for out-of-bag estimation, calculating quantile ranks, and computing proximity counts. The provided QRF estimators are also compatible with and can serve as drop-in replacements for the widely used forest regressors available in scikit-learn [@kramer2016scikit]. The package was designed to be used by a broad array of researchers and in production business settings. It has already been cited in scholarly work [@althoff2023conform;@saporta2023statistical;@Prinzhorn2023] and used in a production setting at the real estate technology company Zillow. The combination of speed, design, and functionality in `quantile-forest` enables exciting scientific explorations in academia and industry alike.

# Statement of Need

Quantile regression is useful for understanding relationships between variables outside of the mean of the data, which can be particularly useful for understanding outcomes that are non-normally distributed or that have nonlinear relationships with predictor variables. It can be used to understand an outcome at its various quantiles and to compare groups or levels of an exposure on those quantiles [@koenker2005quantile].

QRF, an extension of the random forest algorithm, provides a flexible, nonlinear and nonparametric way of performing quantile regression on the predictive distributions for high-dimensional data. Unlike traditional machine learning algorithms that focus solely on point estimation, QRF enables researchers to obtain a more comprehensive understanding of the underlying data distribution. It provides predictions not only for the expected outcome but also for various quantiles, allowing researchers to quantify uncertainties and capture the full spectrum of potential outcomes. QRF has become a standard method for probabilistic prediction in machine learning and applied to many areas requiring reliable probabilistic predictions.

Traditional prediction intervals often rely on assumptions such as normality, which may not hold in many real-world scenarios [@gyamerah2020long]. QRF, on the other hand, allows researchers to generate prediction intervals that are non-parametric, flexible, and adaptive to different data distributions. This capability is invaluable for quantifying uncertainties in a wide range of research areas, including finance [@cordoba2021spatially], environmental sciences [@francke2008estimation;@fang2018quantile;@zhang2018parallel], healthcare [@molinder2020probabilistic;@dean2022quantile], and more. A crucial difference between QRF and many other quantile regression approaches is that after training a QRF once, one has access to all the quantiles at inference time, whereas most approaches require retraining separately for each quantile.

As a cutting-edge statistical modeling technique, the QRF algorithm holds immense potential for researchers across many domains, providing them with a powerful tool to address complex problems involving quantile regression and uncertainty estimation. The QRF algorithm is broadly available in R, which is host to the canonical QRF implementation [@quantregforest2017] as well as established alternative implementations [@athey2019generalized]. However Python has emerged as a prevailing standard programming language within the scientific community, making it a popular option for researchers and practitioners. The absence of a comprehensive Python implementation of the QRF algorithm severely hampers researchers' ability to utilize and benefit from its wide-ranging applications.

We seek to fill this need by providing a comprehensive Python-based implementation of the QRF algorithm. The QRF implementation provided in this package has been optimized for training and inference speed. It allows specifying prediction quantiles after training, permitting a trained model to be reused to estimate conditional quantiles as needed. In addition to this base prediction functionality, the package also includes utilities that enhance the algorithm's applicability and usefulness for researchers and practitioners. These utilities include:

* Out-of-bag scoring: The QRF algorithm can utilize the out-of-bag (OOB) samples, which are the data points that are not used during the construction of a specific decision tree in the forest. OOB scoring can be used to obtain unbiased estimates of prediction errors and quantile-specific metrics, enabling researchers to assess the performance and reliability of the QRF model without the need for additional validation datasets.
* Quantile rank calculation: The QRF algorithm can be leveraged to facilitate the calculation of quantile ranks. Quantile ranks provide a measure of relative standing for each data point in the distribution. This capability allows researchers to compare and rank observations based on their position within the quantile distribution, providing valuable insights for various applications, such as risk assessment and anomaly detection.
* Proximity and similarity estimation: The QRF algorithm can compute proximity measures that quantify the similarity between pairs of observations based on their paths through the forest. These proximity measures capture the notion of closeness or similarity between data points and enable researchers to perform tasks such as clustering, anomaly detection, and identifying influential observations.

By incorporating these utilities into a Python-based implementation of the QRF algorithm, researchers gain a comprehensive and versatile toolkit for quantile regression and uncertainty estimation. Researchers can now harness the power of Python's ecosystem to seamlessly integrate the QRF algorithm into their existing workflows, perform thorough model evaluation through OOB scoring, assess quantile ranks, and leverage proximity and similarity measures for a wide range of data analysis tasks. Accordingly, this package empowers researchers to estimate conditional quantiles accurately and gain deeper insights into complex data.

# Acknowledgements

We thank Andy Krause and Andrew Martin for support throughout the development and release of this work, and Ravi Ashar-Bryant for help during its early development. We also thank the many other members of Zillow Group who have shown support for this project.

# Examples

## Training and Predicting

```python
from quantile_forest import RandomForestQuantileRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

qrf = RandomForestQuantileRegressor().fit(X, y)

y_pred = qrf.predict(X_test, quantiles=[0.025, 0.5, 0.975])
y_pred_oob = qrf.predict(X_train, quantiles=[0.025, 0.5, 0.975], oob_score=True)
```

## Estimating Quantile Ranks

```python
from quantile_forest import RandomForestQuantileRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

qrf = RandomForestQuantileRegressor().fit(X_train, y_train)
y_ranks = qrf.quantile_ranks(X_test, y_test)
```

## Computing Proximities

```python
from quantile_forest import RandomForestQuantileRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

qrf = RandomForestQuantileRegressor().fit(X_train, y_train)
proximities = qrf.proximity_counts(X_test)
```

# References
