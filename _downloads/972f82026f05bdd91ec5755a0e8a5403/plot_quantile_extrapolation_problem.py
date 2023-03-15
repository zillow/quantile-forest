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

import numpy as np
import matplotlib.pyplot as plt
from quantile_forest import RandomForestQuantileRegressor

np.random.seed(0)

n_points = 1000
bounds = [-1, 21]
extrap_pct = 0.2

x = np.linspace(bounds[0], bounds[1], n_points)

f = np.sin(x)
std = 0.01 + np.abs(x - 5.0) / 5.0
noise = np.random.normal(scale=std)
y = f + noise

extrap_min_idx = int(n_points * (extrap_pct / 2))
extrap_max_idx = int(n_points - (n_points * (extrap_pct / 2)))

x_train = x[extrap_min_idx:extrap_max_idx]
y_train = y[extrap_min_idx:extrap_max_idx]

x_mid = x[extrap_min_idx:extrap_max_idx]
x_left = x[:extrap_min_idx]
x_right = x[extrap_max_idx:]

y_mid = y[extrap_min_idx:extrap_max_idx]
y_left = y[:extrap_min_idx]
y_right = y[extrap_max_idx:]

qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=0)
qrf.fit(np.expand_dims(x_train, axis=-1), y_train)

y_pred1 = qrf.predict(np.expand_dims(x_train, axis=-1), quantiles=[0.025, 0.5, 0.975])

y_pred2 = qrf.predict(np.expand_dims(x, axis=-1), quantiles=[0.025, 0.5, 0.975])  # extrapolate
y_pred2_mid = y_pred2[extrap_min_idx:extrap_max_idx]
y_pred2_left = y_pred2[:extrap_min_idx]
y_pred2_right = y_pred2[extrap_max_idx:]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax1.plot([x_train, x_train], [y_pred1[:, 0], y_pred1[:, 2]], alpha=0.2, c="#e0f2ff", lw=3)
ax1.plot(x_train, y_train, c="#f2a619", lw=0, marker="o", mfc="none", ms=1.5)
ax1.plot(x_train, f[extrap_min_idx:extrap_max_idx], c="#006aff")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Prediction Intervals on Training Data")

ax2.plot([x_mid, x_mid], [y_pred2_mid[:, 0], y_pred2_mid[:, 2]], alpha=0.2, c="#e0f2ff", lw=3)
ax2.plot([x_left, x_left], [y_pred2_left[:, 0], y_pred2_left[:, 2]], alpha=0.02, c="r", lw=3)
ax2.plot([x_right, x_right], [y_pred2_right[:, 0], y_pred2_right[:, 2]], alpha=0.02, lw=3, c="r")
ax2.plot(x_left, y_pred2_left[:, 1], alpha=0.5, c="r", lw=0, marker="o", mfc="none", ms=1.5)
ax2.plot(x_right, y_pred2_right[:, 1], alpha=0.5, c="r", lw=0, marker="o", mfc="none", ms=1.5)
ax2.plot(x, y, c="#f2a619", lw=0, marker="o", mfc="none", ms=1.5)
ax2.plot(x, f, c="#006aff")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("Prediction Intervals with Extrapolated Values")

plt.subplots_adjust(top=0.15)
fig.tight_layout(pad=3)

plt.show()
