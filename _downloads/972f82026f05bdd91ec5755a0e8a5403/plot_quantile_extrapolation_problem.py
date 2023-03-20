"""
================================================
Quantile regression forest extrapolation problem
================================================

An example on a toy dataset that demonstrates that the prediction intervals
produced by a quantile regression forest do not extrapolate outside of the
bounds of the data in the training set, an important limitation of the
approach.

"""
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from quantile_forest import RandomForestQuantileRegressor

np.random.seed(0)

n_samples = 1000
bounds = [-1, 21]
extrap_pct = 0.2

x = np.linspace(bounds[0], bounds[1], n_samples)

f = np.sin(x)
std = 0.01 + np.abs(x - 5.0) / 5.0
noise = np.random.normal(scale=std)
y = f + noise

extrap_min_idx = int(n_samples * (extrap_pct / 2))
extrap_max_idx = int(n_samples - (n_samples * (extrap_pct / 2)))

x_train = x[extrap_min_idx:extrap_max_idx]
y_train = y[extrap_min_idx:extrap_max_idx]

x_mid = x[extrap_min_idx:extrap_max_idx]
x_left = x[:extrap_min_idx]
x_right = x[extrap_max_idx:]

y_mid = y[extrap_min_idx:extrap_max_idx]
y_left = y[:extrap_min_idx]
y_right = y[extrap_max_idx:]

xx = np.atleast_2d(np.linspace(-1, 21, n_samples)).T
xx_mid = xx[extrap_min_idx:extrap_max_idx]
xx_left = xx[:extrap_min_idx]
xx_right = xx[extrap_max_idx:]

qrf = RandomForestQuantileRegressor(
    max_samples_leaf=None,
    min_samples_leaf=10,
    random_state=0,
)
qrf.fit(np.expand_dims(x_train, axis=-1), y_train)

y_pred = qrf.predict(xx, quantiles=[0.025, 0.5, 0.975])  # extrapolate
y_pred_mid = y_pred[extrap_min_idx:extrap_max_idx]
y_pred_left = y_pred[:extrap_min_idx]
y_pred_right = y_pred[extrap_max_idx:]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax1.plot(x_train, y_train, c="#f2a619", lw=0, marker=".", ms=4)
ax1.plot(x_train, f[extrap_min_idx:extrap_max_idx], c="black")
ax1.fill_between(x_mid.ravel(), y_pred_mid[:, 0], y_pred_mid[:, 2], color="#e0f2ff")
ax1.plot(x_mid, y_pred_mid[:, 1], c="#006aff", lw=2)
ax1.set_xlim(bounds)
ax1.set_ylim([-8, 8])
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Prediction Intervals on Training Data")

ax2.plot(x, y, c="#f2a619", lw=0, marker=".", ms=4)
ax2.plot(x, f, c="black")
ax2.plot(xx_mid, y_pred_mid[:, 1], c="#006aff", lw=2)
ax2.fill_between(xx_mid.ravel(), y_pred_mid[:, 0], y_pred_mid[:, 2], color="#e0f2ff")
ax2.fill_between(xx_left.ravel(), y_pred_left[:, 0], y_pred_left[:, 2], alpha=0.2, color="r")
ax2.fill_between(xx_right.ravel(), y_pred_right[:, 0], y_pred_right[:, 2], alpha=0.2, color="r")
ax2.plot(xx_left, y_pred_left[:, 1], alpha=0.8, c="r", lw=3)
ax2.plot(xx_right, y_pred_right[:, 1], alpha=0.8, c="r", lw=3)
ax2.set_xlim(bounds)
ax2.set_ylim([-8, 8])
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("Prediction Intervals with Extrapolated Values")

plt.subplots_adjust(top=0.15)
fig.tight_layout(pad=3)

plt.show()
