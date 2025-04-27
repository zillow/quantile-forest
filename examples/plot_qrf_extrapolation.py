"""
Extrapolation with Quantile Regression Forests
==============================================

This example uses a toy dataset to illustrate the prediction intervals
produced by a quantile regression forest (QRF) on extrapolated data. QRFs do
not intrinsically extrapolate outside the bounds of the training data, which
is an important limitation of the approach. Notice that the extrapolated
interval with a standard QRF fails to reliably cover values outside those
observed in the training set. To overcome this limitation, we can use a
procedure known as Xtrapolation, which can estimate the extrapolation bounds
for samples that fall outside the range of the training data. This example is
adapted from `"Extrapolation-Aware Nonparametric Statistical Inference"
<https://arxiv.org/abs/2402.09758>`_ by Niklas Pfister and Peter Bühlmann.
"""

import math

import altair as alt
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
n_samples = 500
extrap_frac = 0.25
bounds = [0, 15]
func = lambda x: x * np.sin(x)
func_str = "f(x) = x sin(x)"
quantiles = [0.025, 0.975, 0.5]
qrf_params = {"min_samples_leaf": 4, "max_samples_leaf": None, "random_state": random_state}


def make_func_Xy(func, n_samples, bounds, add_noise=True, random_state=None):
    """Make a dataset from a specified function."""
    random_state = check_random_state(random_state)

    x = np.linspace(bounds[0], bounds[1], n_samples)
    f = func(x)

    std = 0.01 + np.abs(x - 5.0) / 5.0
    noise = random_state.normal(scale=std) if add_noise else np.zeros_like(f)
    y = f + noise

    return np.atleast_2d(x).T, y


class Xtrapolation:
    """Xtrapolation procedure.

    Performs extrapolation-aware nonparametric statistical inference based on
    an existing nonparametric estimate. Adapted from the Python code [1] for
    the Xtrapolation procedure introduced in [2].

    The procedure specifically applies a QRF for generating local polynomials
    to estimate derivatives in a single dimension. For multi-dimensional
    problems, using the original implementation is strongly encouraged.

    References
    ----------
    .. [1] https://github.com/NiklasPfister/ExtrapolationAware-Inference
    .. [2] N. Pfister and P. Bühlmann, "Extrapolation-Aware Nonparametric
           Statistical Inference", arXiv preprint, 2024.
           https://arxiv.org/abs/2402.09758
    """

    def __init__(self, orders=np.array([1])):
        self.orders_ = orders
        self.max_order_ = np.max(orders)

    @staticmethod
    def _penalized_locpol(fval, v, X, weights, degree, pen=0, penalize_intercept=False):
        v = v.reshape(-1, 1)
        n = X.shape[0]
        dd = degree + 1
        if penalize_intercept:
            pen_list = list(range(0, dd))
        else:
            pen_list = list(range(1, dd))

        # Construct design matrices.
        DDmat = np.zeros((n * dd, n * dd))
        DYmat = np.zeros((n * dd, 1))
        for i in range(n):
            Wi = np.sqrt(weights[i, :].reshape(-1, 1))

            # Construct DDmat (block-diagonal).
            x0v = X[i, :].dot(v)
            Di = np.tile((X.dot(v) - x0v).reshape(-1, 1), dd) ** np.arange(dd) * Wi
            DDmat[(i * dd) : ((i + 1) * dd), (i * dd) : ((i + 1) * dd)] = (Di.T).dot(Di)

            # Construct DYmat.
            DYmat[(i * dd) : ((i + 1) * dd), :] = (Di.T).dot((fval.reshape(-1, 1)) * Wi)

        Z = np.zeros((dd, dd))
        for kk in pen_list:
            Z[kk, kk] = math.factorial(kk)
        PP = np.kron(np.diag(np.sum(weights, axis=1)) - weights, Z)
        penmat = pen * (PP.T).dot(PP)
        B = np.linalg.solve(DDmat + penmat, DYmat)
        coefs = B.reshape(n, -1)

        # Extract derivatives from coefficients.
        deriv_mat = coefs * np.array([math.factorial(k) for k in range(degree + 1)])
        return deriv_mat

    @staticmethod
    def _get_tree_weight_matrix(X, Y, X_eval=None, n_trees=100, rng=None, **kwargs):
        """Fit forest and extract weights.

        This implementation extracts the weight matrix from a list of quantile
        random forests, each with a single tree fitted on non-bootstrapped
        samples. This allows for controlling the bootstrap selection for each
        tree and summing the weight matrices across all of the trees.
        """
        if "n_estimators" in kwargs:
            n_trees = kwargs["n_estimators"]
        kwargs["n_estimators"] = 1
        if "random_state" in kwargs:
            del kwargs["random_state"]
        kwargs["bootstrap"] = False

        rng = np.random.RandomState(0) if rng is None else rng

        trees = [RandomForestQuantileRegressor(random_state=i, **kwargs) for i in range(n_trees)]

        n = X.shape[0]
        nn = 0
        if X_eval is not None:
            nn = X_eval.shape[0]
            X = np.r_[X, X_eval]
        weight_mat = np.zeros((n + nn, n + nn))

        s = 0.5
        bn = int(n * s)

        for tree in trees:
            # Draw bootstrap sample.
            boot_sample = rng.choice(np.arange(n), bn, replace=False)
            split1 = boot_sample[: int(bn / 2)]
            split2 = np.concatenate([boot_sample[int(bn / 2) :], np.arange(nn) + n])

            # Fit tree.
            tree.fit(X[split1, :], Y[split1].flatten())

            # Extract tree weight matrix.
            y_train_leaves = tree._get_y_train_leaves(X[split2, :], Y.reshape(-1, 1))
            nrows = X[split2, :].shape[0]
            matrix = np.zeros((nrows, nrows))
            for leaf in y_train_leaves[0]:
                indices = leaf[0]
                indices = indices[indices != 0] - 1
                if len(indices) > 0:
                    matrix[np.ix_(indices, indices)] = 1
            weight_mat[np.ix_(split2, split2)] += matrix

        # Normalize weights (rows correspond to weights - non-symmetric).
        weight_mat /= weight_mat.sum(axis=1)[:, None]

        return weight_mat

    def fit_weights(self, X, fval, x0=None, train=False, rng=None, **kwargs):
        """Compute random forest weights for derivative estimation."""
        n, d = X.shape
        fval = fval.flatten()

        if train:
            d_xtra = d
            xtra_features = list(range(d))
            weights = [None] * d_xtra
            for jj, var in enumerate(xtra_features):
                var_order = list(range(d))
                var_order = np.array([var] + var_order[:var] + var_order[var + 1 :])
                weights[jj] = self._get_tree_weight_matrix(
                    X[:, var_order], fval, x0, rng=rng, **kwargs
                )
        else:
            weights = self._get_tree_weight_matrix(X, fval, x0, rng=rng, **kwargs)[n:, :n]

        return weights

    def fit_derivatives(self, X, fval, pen=0.1, rng=None, **kwargs):
        """Estimate derivatives."""
        n, d = X.shape
        fval = fval.flatten()

        # Fit weights for local polynomial.
        weights = self.fit_weights(X, fval, train=True, rng=rng, **kwargs)

        # Estimate derivatives with local polynomial.
        derivatives = np.zeros((self.max_order_ + 1, n, d))
        Xtilde = X[:, list(range(d))]

        # Fit local polynomial.
        for jj in range(d):
            vv = np.zeros((d, 1))
            vv[jj] = 1
            tmp = self._penalized_locpol(
                fval,
                vv,
                Xtilde,
                weights[jj],
                degree=self.max_order_ + 1,
                pen=pen,
                penalize_intercept=False,
            )
            for kk in range(self.max_order_ + 1):
                derivatives[kk, :, jj] = fval if kk == 0 else tmp[:, kk]

        return derivatives

    def prediction_bounds(self, X, fval, x0, nn=50, rng=None, **kwargs):
        """Compute extrapolation bounds."""
        n, d = X.shape
        fval = fval.flatten()
        if len(x0.shape) == 1:
            x0 = x0.reshape(-1, 1)
        n0 = x0.shape[0]
        xtra_features = list(range(d))

        # Fit derivatives.
        derivatives = self.fit_derivatives(X, fval, rng=rng, **kwargs)

        # Determine weighting for extrapolation points (using rotation).
        mu = derivatives[1].mean(axis=0)
        _, D, Vt = np.linalg.svd(derivatives[1] - mu[None, :])
        TT = (Vt.T) * D[None, :]
        Xtilde = X[:, xtra_features].dot(TT)
        x0tilde = x0[:, xtra_features].dot(TT)

        # Find closest points between rotated points (Euclidean).
        weight_x0 = np.zeros((n0, n))
        for ii in range(n0):
            xinds = np.argsort(np.sum((x0tilde[None, ii, :] - Xtilde) ** 2, axis=1))[:nn]
            weight_x0[ii, xinds] = 1 / nn

        # Precompute factorials.
        order_factorials = np.empty(self.max_order_ + 1)
        for oo in range(self.max_order_ + 1):
            order_factorials[oo] = math.factorial(oo)

        # Iterate over all extrapolation points and average/intersect.
        bounds = np.zeros((n0, len(self.orders_), 3))
        for ll, xpt in enumerate(x0):
            xinds = np.where(weight_x0[ll, :] != 0)[0]

            # Number of anchor points to check.
            f_lower = np.zeros((len(xinds), len(self.orders_)))
            f_upper = np.zeros((len(xinds), len(self.orders_)))
            f_median = np.zeros((len(xinds), len(self.orders_)))
            for ii, xind in enumerate(xinds):
                xx = X[xind, :].reshape(1, -1)
                vv = (xpt - xx)[:, xtra_features]
                vv_norm = np.sqrt(np.sum(vv**2))

                # Compute directional derivatives.
                deriv_mat = np.zeros((n, self.max_order_ + 1))
                deriv_mat[:, 0] = derivatives[0, :, :].mean(axis=1)
                if vv_norm > np.finfo(float).eps:
                    vv_direction = np.array(vv / vv_norm).reshape(-1, 1)
                    for kk in range(1, self.max_order_ + 1):
                        deriv_mat[:, kk] = derivatives[kk, :, :].dot(vv_direction**kk).flatten()

                # Select bounds.
                deriv_min = np.quantile(deriv_mat, 0, axis=0)
                deriv_max = np.quantile(deriv_mat, 1, axis=0)
                deriv_median = np.quantile(deriv_mat, 0.5, axis=0)

                # Estimate extrapolation bounds.
                mterm = 0
                kk = 0
                for oo in range(self.max_order_ + 1):
                    if oo in self.orders_:
                        lo_bdd = deriv_min[oo] * (vv_norm**oo) / order_factorials[oo]
                        up_bdd = deriv_max[oo] * (vv_norm**oo) / order_factorials[oo]
                        median_deriv = deriv_median[oo] * (vv_norm**oo) / order_factorials[oo]
                        f_lower[ii, kk] = mterm + lo_bdd
                        f_upper[ii, kk] = mterm + up_bdd
                        f_median[ii, kk] = mterm + median_deriv
                        kk += 1
                    mterm += deriv_mat[xind, oo] * (vv_norm**oo) / order_factorials[oo]

            # Combine bounds over x-indices.
            ww = (weight_x0[ll, xinds] / np.sum(weight_x0[ll, :]))[:, None]
            f_median = np.sum(f_median * ww, axis=0)

            # Aggregate by optimal-average.
            f_lower = np.max(f_lower, axis=0)
            f_upper = np.min(f_upper, axis=0)
            ind = f_upper < f_lower
            average = (f_upper + f_lower) / 2
            f_lower[ind] = average[ind]
            f_upper[ind] = average[ind]

            bounds[ll, :, 0] = f_lower
            bounds[ll, :, 1] = f_upper
            bounds[ll, :, 2] = f_median

        return bounds


def train_test_split(train_indices, rng=None, **kwargs):
    """Fit model on training samples and extrapolate on test samples."""
    X_train = X[train_indices, :]
    y_train = y[train_indices]

    # Run quantile regression (with forests).
    qrf = RandomForestQuantileRegressor(**kwargs)
    qrf.fit(X_train, y_train)
    qmat = qrf.predict(X, quantiles=quantiles)

    # Xtrapolation.
    bounds_list = [None] * len(quantiles)
    for i in range(len(quantiles)):
        # Run Xtrapolation on quantile.
        xtra = Xtrapolation()
        bounds_list[i] = xtra.prediction_bounds(
            X_train, qmat[train_indices, i], X, rng=rng, **kwargs
        )

    return {
        "train_indices": train_indices,
        "quantiles": quantiles,
        "qmat": qmat,
        "bounds_list": bounds_list,
    }


def prob_randomized_pi(qmat, y, coverage):
    """Calculate calibration probability."""
    alpha_included = np.mean((qmat[:, 0] <= y) & (y <= qmat[:, 1]))
    alpha_excluded = np.mean((qmat[:, 0] < y) & (y < qmat[:, 1]))
    if coverage <= alpha_excluded:
        prob_si = 1
    elif coverage >= alpha_included:
        prob_si = 0
    else:
        prob_si = (coverage - alpha_included) / (alpha_excluded - alpha_included)
    return prob_si


def randomized_pi(qmat, prob_si, y, random_state=None):
    """Calculate coverage."""
    rng = np.random.RandomState(0) if random_state is None else random_state
    si_index = rng.choice([False, True], len(y), replace=True, p=[prob_si, 1 - prob_si])
    included = (qmat[:, 0] < y) & (y < qmat[:, 1])
    boundary = (qmat[:, 0] == y) | (qmat[:, 1] == y)
    return included | (boundary & si_index)


def get_coverage_qrf(qmat, train_indices, test_indices, y_train, level, *args):
    """Calculate extrapolation coverage for regular quantile forest."""
    prob_si = prob_randomized_pi(qmat[train_indices, :], y_train, level)
    qrf = randomized_pi(qmat, prob_si, y, *args)
    return np.mean(qrf[test_indices])


def get_coverage_xtr(bounds_list, train_indices, test_indices, y_train, level, *args):
    """Calculate extrapolation coverage for Xtrapolation."""
    bb_low = np.max(bounds_list[0][:, :, 0], axis=1)
    bb_upp = np.min(bounds_list[1][:, :, 1], axis=1)
    bb_low_train, bb_upp_train = bb_low[train_indices], bb_upp[train_indices]
    prob_si = prob_randomized_pi(np.c_[bb_low_train, bb_upp_train], y_train, level)
    xtra = randomized_pi(np.c_[bb_low, bb_upp], prob_si, y, *args)
    return np.mean(xtra[test_indices])


# Create a dataset that requires extrapolation.
X, y = make_func_Xy(func, n_samples, bounds, add_noise=True, random_state=0)

# Fit and extrapolate based on train-test split (depending on X).
extrap_min_idx = int(n_samples * (extrap_frac / 2))
extrap_max_idx = int(n_samples - (n_samples * (extrap_frac / 2)))
sort_X = np.argsort(X.squeeze())
train_indices = np.repeat(False, len(y))
train_indices[sort_X[extrap_min_idx] : sort_X[extrap_max_idx]] = True
res = train_test_split(train_indices, rng=random_state, **qrf_params)

# Get coverages for extrapolated samples.
args = (train_indices, ~train_indices, y[train_indices], quantiles[1] - quantiles[0], random_state)
cov_qrf = get_coverage_qrf(res["qmat"], *args)
cov_xtr = get_coverage_xtr(res["bounds_list"], *args)

df = pd.DataFrame(
    {
        "X_true": X.squeeze(),
        "y_func": func(X.squeeze()),
        "y_true": y,
        "y_pred": res["qmat"][:, 2],
        "y_pred_low": res["qmat"][:, 0],
        "y_pred_upp": res["qmat"][:, 1],
        "bb_low": np.max(res["bounds_list"][0][:, :, 0], axis=1),
        "bb_upp": np.min(res["bounds_list"][1][:, :, 1], axis=1),
        "bb_mid": np.median(res["bounds_list"][2][:, :, :2], axis=(1, 2)),
        "train": res["train_indices"],
        "test_left": [True] * extrap_min_idx + [False] * (len(y) - extrap_min_idx),
        "test_right": [False] * extrap_max_idx + [True] * (len(y) - extrap_max_idx),
        "cov_qrf": cov_qrf,
        "cov_xtr": cov_xtr,
    }
)


def plot_qrf_vs_xtrapolation_comparison(df, func_str):
    """Plot comparison of QRF vs. Xtrapolation on extrapolated data."""

    def _plot_extrapolations(
        df,
        title="",
        legend=False,
        func_str="",
        x_domain=None,
        y_domain=None,
    ):
        x_scale = None
        if x_domain is not None:
            x_scale = alt.Scale(domain=x_domain, nice=False, padding=0)
        y_scale = None
        if y_domain is not None:
            y_scale = alt.Scale(domain=y_domain, nice=True)

        points_color = alt.value("#f2a619")
        line_true_color = alt.value("gray")
        if legend:
            points_color = alt.Color(
                "point_label:N", scale=alt.Scale(range=["#f2a619"]), title=None
            )
            line_true_color = alt.Color(
                "line_label:N", scale=alt.Scale(range=["gray"]), title=None
            )

        tooltip_true = [
            alt.Tooltip("X_true:Q", format=",.3f", title="X"),
            alt.Tooltip("y_true:Q", format=",.3f", title="Y"),
        ]

        tooltip_pred = tooltip_true + [
            alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
            alt.Tooltip("y_pred_low:Q", format=",.3f", title="Predicted Lower Y"),
            alt.Tooltip("y_pred_upp:Q", format=",.3f", title="Predicted Upper Y"),
        ]

        base = alt.Chart(df.assign(**{"point_label": "Observations", "line_label": func_str}))

        bar_pred = base.mark_bar(clip=True, width=2).encode(
            x=alt.X("X_true"),
            y=alt.Y("y_pred_low"),
            y2=alt.Y2("y_pred_upp"),
            color=alt.condition(alt.datum["extrapolate"], alt.value("red"), alt.value("#e0f2ff")),
            opacity=alt.condition(alt.datum["extrapolate"], alt.value(0.05), alt.value(0.8)),
            tooltip=tooltip_pred,
        )

        circle_true = base.mark_circle(size=20).encode(
            x=alt.X("X_true:Q", scale=x_scale, title="X"),
            y=alt.Y("y_true:Q", scale=y_scale, title="Y"),
            color=points_color,
            tooltip=tooltip_true,
        )

        line_true = base.mark_line().encode(
            x=alt.X("X_true:Q", scale=x_scale, title=""),
            y=alt.Y("y_func:Q", scale=y_scale, title=""),
            color=line_true_color,
            tooltip=tooltip_true,
        )

        line_pred = base.mark_line(clip=True).encode(
            x=alt.X("X_true:Q", title="", scale=x_scale),
            y=alt.Y("y_pred:Q", scale=y_scale),
            color=alt.condition(alt.datum["extrapolate"], alt.value("red"), alt.value("#006aff")),
            tooltip=tooltip_pred,
        )

        chart = bar_pred + circle_true + line_true + line_pred

        if "coverage" in df.columns:
            text_coverage = (
                base.transform_aggregate(coverage="mean(coverage)")
                .transform_calculate(
                    coverage_text=(
                        "'Extrapolated Coverage: '"
                        f" + format({alt.datum['coverage'] * 100}, '.1f') + '%'"
                        f" + ' (target = {(quantiles[1] - quantiles[0]) * 100}%)'"
                    )
                )
                .mark_text(align="left", baseline="top", color="gray")
                .encode(
                    x=alt.value(5),
                    y=alt.value(5),
                    text=alt.Text("coverage_text:N"),
                )
            )
            chart += text_coverage

        if legend:
            # For desired legend ordering.
            data = {
                "y_pred_line": {"type": "line", "color": "#006aff", "name": "Predicted Median"},
                "y_pred_area": {
                    "type": "area",
                    "color": "#e0f2ff",
                    "name": "Predicted 95% Interval",
                },
                "y_extrp_line": {"type": "line", "color": "red", "name": "Extrapolated Median"},
                "y_extrp_area": {
                    "type": "area",
                    "color": "red",
                    "name": "Extrapolated 95% Interval",
                },
            }
            for k, v in data.items():
                blank = alt.Chart(pd.DataFrame({k: [v["name"]]}))
                if v["type"] == "line":
                    blank = blank.mark_line(color=k)
                elif v["type"] == "area":
                    blank = blank.mark_area(color=k)
                blank = blank.encode(
                    color=alt.Color(f"{k}:N", scale=alt.Scale(range=[v["color"]]), title=None)
                )
                chart += blank
            chart = chart.resolve_scale(color="independent")

        chart = chart.properties(title=title, height=200, width=300)
        return chart

    kwargs = {"func_str": func_str, "x_domain": [0, 15], "y_domain": [-15, 20]}
    xtra_mapper = {"bb_mid": "y_pred", "bb_low": "y_pred_low", "bb_upp": "y_pred_upp"}

    chart1 = alt.layer(
        _plot_extrapolations(
            df.query("~(test_left | test_right)").assign(**{"coverage": lambda x: x["cov_qrf"]}),
            title="Extrapolation with Standard QRF",
            **kwargs,
        ).resolve_scale(color="independent"),
        _plot_extrapolations(df.query("test_left").assign(extrapolate=True), **kwargs),
        _plot_extrapolations(df.query("test_right").assign(extrapolate=True), **kwargs),
    )
    chart2 = alt.layer(
        _plot_extrapolations(
            df.query("~(test_left | test_right)").assign(**{"coverage": lambda x: x["cov_xtr"]}),
            title="Extrapolation with Xtrapolation Procedure",
            legend=True,
            **kwargs,
        ).resolve_scale(color="independent"),
        _plot_extrapolations(
            df.query("test_left")
            .assign(extrapolate=True)
            .drop(columns=["y_pred", "y_pred_low", "y_pred_upp"])
            .rename(xtra_mapper, axis="columns"),
            **kwargs,
        ),
        _plot_extrapolations(
            df.query("test_right")
            .assign(extrapolate=True)
            .drop(columns=["y_pred", "y_pred_low", "y_pred_upp"])
            .rename(xtra_mapper, axis="columns"),
            **kwargs,
        ),
    )
    chart = chart1 | chart2
    return chart


chart = plot_qrf_vs_xtrapolation_comparison(df, func_str)
chart
