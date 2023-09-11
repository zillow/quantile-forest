Search.setIndex({"docnames": ["api", "auto_examples/index", "auto_examples/plot_quantile_extrapolation_problem", "auto_examples/plot_quantile_interpolation", "auto_examples/plot_quantile_regression_intervals", "auto_examples/plot_quantile_toy_example", "auto_examples/plot_quantile_vs_standard_forest", "auto_examples/plot_quantile_weighted_vs_unweighted", "auto_examples/sg_execution_times", "generated/quantile_forest.ExtraTreesQuantileRegressor", "generated/quantile_forest.RandomForestQuantileRegressor", "index", "install", "user_guide"], "filenames": ["api.rst", "auto_examples/index.rst", "auto_examples/plot_quantile_extrapolation_problem.rst", "auto_examples/plot_quantile_interpolation.rst", "auto_examples/plot_quantile_regression_intervals.rst", "auto_examples/plot_quantile_toy_example.rst", "auto_examples/plot_quantile_vs_standard_forest.rst", "auto_examples/plot_quantile_weighted_vs_unweighted.rst", "auto_examples/sg_execution_times.rst", "generated/quantile_forest.ExtraTreesQuantileRegressor.rst", "generated/quantile_forest.RandomForestQuantileRegressor.rst", "index.rst", "install.rst", "user_guide.rst"], "titles": ["API Reference", "General Examples", "Quantile regression forest extrapolation problem", "Predicting with different quantile interpolation methods", "Quantile regression forest prediction intervals", "Quantile regression forest predictions compared to ground truth function", "Quantile regression forest vs. standard regression forest", "Predicting with weighted and unweighted quantiles", "Computation times", "ExtraTreesQuantileRegressor", "RandomForestQuantileRegressor", "quantile-forest", "Getting Started", "User Guide"], "terms": {"thi": [0, 9, 10, 11, 13], "i": [0, 3, 6, 7, 9, 10, 11, 12, 13], "full": [0, 2, 3, 4, 5, 6, 7, 13], "document": [0, 11], "packag": [0, 11, 12, 13], "purpos": [1, 11], "introductori": [1, 11], "quantil": [1, 8, 9, 10, 12], "regress": [1, 7, 8, 9, 10, 11], "forest": [1, 7, 8, 9, 10, 12], "v": [1, 8, 9, 10, 12], "standard": [1, 7, 8, 13], "predict": [1, 2, 6, 8, 9, 10], "compar": [1, 8], "ground": [1, 8], "truth": [1, 8], "function": [1, 8, 9, 10, 11, 13], "differ": [1, 8], "interpol": [1, 8, 9, 10], "method": [1, 8, 9, 10, 13], "extrapol": [1, 8], "problem": [1, 8], "weight": [1, 8, 9, 10, 13], "unweight": [1, 8, 9, 10, 13], "interv": [1, 2, 5, 8, 9, 10, 13], "download": [1, 2, 3, 4, 5, 6, 7], "all": [1, 9, 10, 11, 13], "python": [1, 2, 3, 4, 5, 6, 7, 12], "sourc": [1, 2, 3, 4, 5, 6, 7, 9], "code": [1, 2, 3, 4, 5, 6, 7, 12], "auto_examples_python": 1, "zip": [1, 3, 4, 7], "jupyt": [1, 2, 3, 4, 5, 6, 7], "notebook": [1, 2, 3, 4, 5, 6, 7], "auto_examples_jupyt": 1, "galleri": [1, 2, 3, 4, 5, 6, 7], "sphinx": [1, 2, 3, 4, 5, 6, 7, 12], "go": [2, 3, 4, 5, 6, 7], "end": [2, 3, 4, 5, 6, 7, 9, 10], "exampl": [2, 3, 4, 5, 6, 7, 9, 10, 13], "an": [2, 3, 4, 5, 6, 7, 9, 10, 11, 13], "toi": [2, 3], "dataset": [2, 3, 4, 6, 7, 9, 10, 13], "demonstr": [2, 5], "produc": 2, "do": [2, 13], "outsid": 2, "bound": 2, "data": [2, 3, 4, 9, 10, 13], "train": [2, 9, 10, 13], "set": [2, 9, 10, 13], "import": [2, 3, 4, 5, 6, 7, 9, 10, 13], "limit": [2, 13], "approach": [2, 13], "print": [2, 3, 4, 5, 6, 7], "__doc__": [2, 3, 4, 5, 6, 7], "matplotlib": [2, 3, 4, 5, 6, 7], "pyplot": [2, 3, 4, 5, 6, 7], "plt": [2, 3, 4, 5, 6, 7], "numpi": [2, 3, 4, 5, 7, 13], "np": [2, 3, 4, 5, 7, 9, 10, 13], "from": [2, 3, 4, 5, 6, 7, 9, 10, 13], "quantile_forest": [2, 3, 4, 5, 6, 7, 9, 10, 12, 13], "randomforestquantileregressor": [2, 3, 4, 5, 6, 7, 9, 13], "random": [2, 5, 7, 9, 10, 13], "seed": [2, 5], "0": [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13], "n_sampl": [2, 4, 5, 6, 7, 9, 10], "1000": [2, 4, 5, 7, 9, 10], "1": [2, 3, 4, 5, 6, 7, 9, 10, 12, 13], "21": 2, "extrap_pct": 2, "2": [2, 3, 4, 5, 6, 7, 9, 10], "x": [2, 3, 4, 5, 6, 7, 9, 10, 13], "linspac": [2, 5], "f": [2, 5, 9, 10, 13], "sin": [2, 5], "std": [2, 7], "01": 2, "ab": 2, "5": [2, 3, 4, 5, 6, 7, 9, 10, 13], "nois": [2, 5], "normal": [2, 9, 10], "scale": [2, 4, 6], "y": [2, 3, 4, 5, 6, 7, 9, 10, 13], "extrap_min_idx": 2, "int": [2, 4, 9, 10, 13], "extrap_max_idx": 2, "x_train": [2, 4, 5, 6, 7, 13], "y_train": [2, 4, 5, 6, 7, 13], "x_mid": 2, "x_left": 2, "x_right": 2, "y_mid": 2, "y_left": 2, "y_right": 2, "xx": [2, 5], "atleast_2d": [2, 5], "t": [2, 5, 13], "xx_mid": 2, "xx_left": 2, "xx_right": 2, "qrf": [2, 4, 5, 6, 7, 9, 10, 13], "max_samples_leaf": [2, 3, 7, 9, 10, 13], "none": [2, 3, 7, 9, 10, 13], "min_samples_leaf": [2, 5, 9, 10], "10": [2, 3, 4, 5, 6, 7, 13], "random_st": [2, 3, 4, 5, 6, 7, 9, 10, 13], "fit": [2, 3, 4, 5, 6, 7, 9, 10], "expand_dim": 2, "axi": [2, 3, 4, 7], "y_pred": [2, 3, 4, 5, 9, 10, 13], "025": [2, 3, 4, 5], "975": [2, 3, 4, 5], "y_pred_mid": 2, "y_pred_left": 2, "y_pred_right": 2, "fig": [2, 4], "ax1": [2, 4], "ax2": [2, 4], "subplot": [2, 4], "nrow": [2, 4], "ncol": [2, 4], "figsiz": [2, 4], "4": [2, 4, 7], "plot": [2, 4, 5, 7], "c": [2, 4, 5, 7, 13], "f2a619": [2, 3, 4, 5, 6, 7], "lw": [2, 4, 5], "marker": [2, 4], "m": [2, 4, 5], "black": [2, 5], "fill_between": [2, 4, 5, 7], "ravel": [2, 5], "color": [2, 3, 4, 5, 6, 7], "e0f2ff": [2, 4, 5], "006aff": [2, 3, 4, 5, 7], "set_xlim": [2, 4], "set_ylim": [2, 4], "8": [2, 4, 9, 10, 12], "set_xlabel": [2, 4], "set_ylabel": [2, 4], "set_titl": 2, "alpha": [2, 4, 7], "r": [2, 9, 10, 12], "3": [2, 4, 5, 7, 9, 10, 12, 13], "valu": [2, 3, 4, 6, 7, 9, 10, 13], "subplots_adjust": [2, 4], "top": [2, 4], "15": [2, 3, 4], "tight_layout": [2, 4], "pad": [2, 4], "show": [2, 3, 4, 5, 6, 7], "total": [2, 3, 4, 5, 6, 7, 8, 9, 10, 13], "run": [2, 3, 4, 5, 6, 7, 9, 10, 12], "time": [2, 3, 4, 5, 6, 7, 13], "script": [2, 3, 4, 5, 6, 7], "minut": [2, 3, 4, 5, 6, 7], "439": [2, 8], "second": [2, 3, 4, 5, 6, 7], "plot_quantile_extrapolation_problem": [2, 8], "py": [2, 3, 4, 5, 6, 7, 8], "ipynb": [2, 3, 4, 5, 6, 7], "gener": [2, 3, 4, 5, 6, 7, 9, 10, 11, 13], "comparison": [3, 6, 7], "can": [3, 9, 10, 12, 13], "appli": [3, 9, 10], "dure": [3, 9, 10, 13], "when": [3, 7, 9, 10, 13], "desir": [3, 9, 10, 13], "li": [3, 9, 10], "between": [3, 6, 9, 10], "two": [3, 9, 10, 13], "point": [3, 9, 10], "creat": [3, 6, 7, 9, 10], "arrai": [3, 7, 9, 10, 13], "est": 3, "n_estim": [3, 4, 6, 7, 9, 10], "bootstrap": [3, 9, 10, 13], "fals": [3, 7, 9, 10, 13], "linear": [3, 9, 10], "lower": [3, 9, 10, 13], "higher": [3, 9, 10], "midpoint": [3, 9, 10], "nearest": [3, 9, 10], "ffd237": 3, "0d4599": 3, "a6e5ff": [3, 6], "y_median": 3, "y_err": 3, "append": 3, "concaten": [3, 4, 13], "sc": 3, "scatter": 3, "arang": [3, 4, 13], "len": [3, 4, 7, 13], "35": 3, "k": [3, 9], "zorder": 3, "eb": 3, "median": [3, 4, 5, 6, 13], "enumer": [3, 7], "errorbar": 3, "yerr": 3, "ecolor": 3, "fmt": 3, "o": 3, "xlim": [3, 7], "75": [3, 13], "25": [3, 5, 7, 13], "xtick": 3, "tolist": 3, "xlabel": [3, 5, 6, 7], "sampl": [3, 4, 5, 9, 10, 13], "featur": [3, 9, 10], "ylabel": [3, 5, 6, 7], "actual": [3, 6], "legend": [3, 5, 6, 7], "loc": [3, 5, 6], "110": [3, 8], "plot_quantile_interpol": [3, 8], "how": [4, 9, 10], "us": [4, 5, 7, 9, 10, 12, 13], "california": 4, "hous": 4, "ticker": 4, "funcformatt": 4, "sklearn": [4, 5, 6, 7, 9, 10, 13], "model_select": [4, 5, 6, 7, 13], "kfold": 4, "util": [4, 6, 9, 10], "valid": [4, 6, 9, 10], "check_random_st": [4, 6], "rng": [4, 5, 6], "dollar_formatt": 4, "lambda": [4, 7], "p": 4, "format": [4, 9, 10], "load": 4, "price": 4, "fetch_california_h": [4, 9, 10], "min": 4, "target": [4, 6, 9, 10], "size": [4, 5, 9, 10, 13], "perm": 4, "permut": 4, "100": [4, 7, 9, 10], "kf": 4, "n_split": 4, "get_n_split": 4, "y_true": [4, 9, 10], "y_pred_low": [4, 5], "y_pred_upp": [4, 5], "train_index": 4, "test_index": 4, "split": [4, 9, 10], "x_test": [4, 5, 6, 7, 13], "y_test": [4, 5, 6, 7, 13], "set_param": [4, 9, 10], "max_featur": [4, 9, 10], "shape": [4, 9, 10], "get": [4, 9, 10], "95": [4, 5], "y_pred_i": 4, "dollar": 4, "1e5": 4, "y_pred_interv": 4, "sort_idx": 4, "argsort": 4, "y_min": 4, "minimum": [4, 9, 10], "y_max": 4, "max": 4, "maximum": [4, 9, 10, 13], "float": [4, 9, 10, 13], "round": [4, 9, 10], "10000": 4, "low": 4, "mid": 4, "upp": 4, "_": [4, 7], "l": [4, 13], "grei": 4, "grid": 4, "xaxi": 4, "set_major_formatt": 4, "yaxi": 4, "condit": [4, 5, 13], "observ": [4, 5], "center": 4, "mean": [4, 6, 7, 9, 10, 13], "order": [4, 9, 10, 13], "224": [4, 8], "plot_quantile_regression_interv": [4, 8], "The": [5, 9, 10, 11, 12, 13], "noisi": 5, "train_test_split": [5, 6, 7, 13], "def": [5, 7], "make_toy_dataset": 5, "randomst": [5, 9, 10], "uniform": 5, "sigma": 5, "lognorm": 5, "exp": 5, "return": [5, 9, 10, 13], "max_depth": [5, 9, 10], "y_pred_m": 5, "label": [5, 6], "test": [5, 9, 10, 11, 13], "upper": 5, "left": [5, 9, 10], "293": [5, 8], "plot_quantile_toy_exampl": [5, 8], "synthet": [6, 7], "right": [6, 9, 10], "skew": 6, "In": [6, 9, 10, 13], "distribut": [6, 9, 10, 13], "scipi": [6, 9, 10, 12], "sp": 6, "ensembl": [6, 7, 9, 10, 13], "randomforestregressor": [6, 7, 13], "5000": 6, "skewnorm_rv": 6, "stat": [6, 9, 10], "skewnorm": 6, "rv": 6, "randn": 6, "reshap": 6, "regr_rf": 6, "regr_qrf": 6, "test_siz": [6, 7, 13], "y_pred_rf": [6, 13], "y_pred_qrf": [6, 13], "c0c0c0": 6, "name": [6, 9, 10], "rf": [6, 7, 13], "hist": 6, "bin": 6, "50": [6, 7], "count": [6, 9, 10], "499": [6, 8], "plot_quantile_vs_standard_forest": [6, 8], "runtim": 7, "comput": [7, 9, 10, 13], "output": [7, 9, 10, 13], "A": [7, 9, 10, 13], "regressor": [7, 9, 10, 13], "includ": [7, 13], "contextlib": 7, "contextmanag": 7, "t0": 7, "yield": 7, "t1": 7, "make_regress": 7, "n_featur": [7, 9, 10], "estimator_s": 7, "n_repeat": 7, "n_size": 7, "empti": 7, "j": [7, 9, 10, 13], "rang": [7, 9, 10, 13], "rf_time": 7, "qrf_weighted_tim": 7, "weighted_quantil": [7, 9, 10, 13], "true": [7, 9, 10, 13], "qrf_unweighted_tim": 7, "rf_time_avg": 7, "qrf_weighted_time_avg": 7, "qrf_unweighted_time_avg": 7, "list": [7, 9, 10, 13], "rf_time_std": 7, "qrf_weighted_time_std": 7, "qrf_unweighted_time_std": 7, "001751": 7, "96": 7, "number": [7, 9, 10, 13], "estim": [7, 9, 10, 13], "11": [7, 8], "831": [7, 8], "plot_quantile_weighted_vs_unweight": [7, 8], "00": 8, "17": 8, "394": 8, "execut": 8, "auto_exampl": 8, "file": 8, "mb": 8, "04": 8, "class": [9, 10, 11], "default_quantil": [9, 10, 13], "criterion": [9, 10], "squared_error": [9, 10], "min_samples_split": [9, 10], "min_weight_fraction_leaf": [9, 10], "max_leaf_nod": [9, 10], "min_impurity_decreas": [9, 10], "oob_scor": [9, 10, 13], "n_job": [9, 10], "verbos": [9, 10, 12], "warm_start": [9, 10], "ccp_alpha": [9, 10], "max_sampl": [9, 10], "extra": 9, "tree": [9, 10, 13], "provid": [9, 10, 11, 13], "implement": [9, 11], "meta": [9, 10], "decis": [9, 10, 13], "variou": [9, 10], "sub": [9, 10], "averag": [9, 10, 13], "improv": [9, 10], "accuraci": [9, 10], "control": [9, 10], "over": [9, 10], "paramet": [9, 10, 13], "default": [9, 10, 13], "model": [9, 10, 13], "tri": [9, 10], "each": [9, 10, 13], "must": [9, 10], "strictli": [9, 10], "If": [9, 10, 12, 13], "absolute_error": [9, 10], "measur": [9, 10], "qualiti": [9, 10], "support": [9, 10], "criteria": [9, 10], "ar": [9, 10, 12, 13], "squar": [9, 10], "error": [9, 10], "which": [9, 10, 13], "equal": [9, 10, 13], "varianc": [9, 10, 13], "reduct": [9, 10], "select": [9, 10, 13], "absolut": [9, 10], "depth": [9, 10], "node": [9, 10, 13], "expand": [9, 10], "until": [9, 10], "leav": [9, 10, 13], "pure": [9, 10], "contain": [9, 10], "less": [9, 10], "than": [9, 10, 13], "requir": [9, 10, 12], "intern": [9, 10], "consid": [9, 10], "fraction": [9, 10, 13], "ceil": [9, 10], "leaf": [9, 10, 13], "ani": [9, 10, 13], "onli": [9, 10, 13], "least": [9, 10], "branch": [9, 10], "mai": [9, 10], "have": [9, 10, 13], "effect": [9, 10, 13], "smooth": [9, 10], "especi": [9, 10], "permit": [9, 10], "unlimit": [9, 10], "sum": [9, 10], "input": [9, 10, 13], "sample_weight": [9, 10], "sqrt": [9, 10], "log2": [9, 10], "look": [9, 10], "best": [9, 10], "auto": [9, 10], "equival": [9, 10], "bag": [9, 10, 13], "more": [9, 10, 13], "achiev": [9, 10], "smaller": [9, 10, 13], "e": [9, 10, 13], "g": [9, 10], "note": [9, 10, 13], "search": [9, 10], "doe": [9, 10], "stop": [9, 10], "one": [9, 10, 13], "partit": [9, 10], "found": [9, 10], "even": [9, 10], "inspect": [9, 10], "grow": [9, 10], "first": [9, 10, 13], "fashion": [9, 10], "defin": [9, 10], "rel": [9, 10], "impur": [9, 10], "induc": [9, 10], "decreas": [9, 10], "greater": [9, 10], "equat": [9, 10], "follow": [9, 10, 12], "n_t": [9, 10], "n": [9, 10, 13], "n_t_r": [9, 10], "right_impur": [9, 10], "n_t_l": [9, 10], "left_impur": [9, 10], "where": [9, 10, 13], "current": [9, 10], "child": [9, 10], "refer": [9, 10], "pass": [9, 10, 13], "bool": [9, 10], "whether": [9, 10], "build": [9, 10, 12, 13], "whole": [9, 10], "out": [9, 10, 13], "score": [9, 10, 13], "avail": [9, 10, 12], "job": [9, 10], "parallel": [9, 10], "decision_path": [9, 10], "unless": [9, 10], "joblib": [9, 10], "parallel_backend": [9, 10], "context": [9, 10], "processor": [9, 10], "instanc": [9, 10], "draw": [9, 10], "reus": [9, 10], "solut": [9, 10], "previou": [9, 10], "call": [9, 10, 13], "add": [9, 10], "otherwis": [9, 10], "just": [9, 10], "new": [9, 10, 13], "non": [9, 10, 13], "neg": [9, 10], "complex": [9, 10], "minim": [9, 10], "cost": [9, 10], "prune": [9, 10], "subtre": [9, 10], "largest": [9, 10], "chosen": [9, 10], "By": [9, 10, 13], "perform": [9, 10, 13], "base": [9, 10, 13], "thu": [9, 10], "should": [9, 10], "attribut": [9, 10], "base_estimator_": [9, 10], "extratreequantileregressor": 9, "estimators_": [9, 10], "forestregressor": 9, "collect": [9, 10], "feature_importances_": [9, 10], "ndarrai": [9, 10], "n_features_in_": [9, 10], "seen": [9, 10], "feature_names_in_": [9, 10], "ha": [9, 10], "string": [9, 10], "n_outputs_": [9, 10], "oob_score_": [9, 10], "obtain": [9, 10], "exist": [9, 10], "oob_prediction_": [9, 10], "n_output": [9, 10], "optim": 9, "meinshausen": [9, 10, 13], "journal": [9, 10, 13], "machin": [9, 10, 13], "learn": [9, 10, 11, 12, 13], "research": [9, 10, 13], "7": [9, 10, 13], "jun": [9, 10], "983": [9, 10, 13], "999": [9, 10, 13], "2006": [9, 10, 13], "http": [9, 10, 13], "www": [9, 10, 13], "jmlr": [9, 10, 13], "org": [9, 10, 13], "paper": [9, 10, 13], "volume7": [9, 10, 13], "meinshausen06a": [9, 10, 13], "pdf": [9, 10, 13], "return_x_i": [9, 10, 13], "3352": 9, "indic": [9, 10, 13], "like": [9, 10, 13], "spars": [9, 10], "matrix": [9, 10], "its": [9, 10, 13], "dtype": [9, 10, 13], "convert": [9, 10], "float32": [9, 10], "csr_matrix": [9, 10], "x_leav": [9, 10], "For": [9, 10, 13], "datapoint": [9, 10], "index": [9, 10, 13], "up": [9, 10], "properti": [9, 10], "path": [9, 10], "version": [9, 10], "18": [9, 10], "n_node": [9, 10], "zero": [9, 10, 13], "element": [9, 10], "goe": [9, 10], "through": [9, 10], "csr": [9, 10], "n_nodes_ptr": [9, 10], "column": [9, 10, 13], "give": [9, 10, 13], "th": [9, 10, 13], "brought": [9, 10], "It": [9, 10, 13], "also": [9, 10, 13], "known": [9, 10], "gini": [9, 10], "warn": [9, 10], "mislead": [9, 10], "high": [9, 10, 13], "cardin": [9, 10], "mani": [9, 10], "uniqu": [9, 10], "see": [9, 10], "permutation_import": [9, 10], "altern": [9, 10], "singl": [9, 10, 13], "consist": [9, 10], "root": [9, 10], "case": [9, 10], "sparse_pickl": [9, 10], "csc_matrix": [9, 10], "real": [9, 10, 13], "would": [9, 10], "net": [9, 10], "ignor": [9, 10], "while": [9, 10, 13], "classif": [9, 10, 13], "thei": [9, 10, 13], "result": [9, 10], "carri": [9, 10], "either": [9, 10], "pickl": [9, 10], "underli": [9, 10], "structur": [9, 10], "self": [9, 10], "object": [9, 10, 13], "get_metadata_rout": [9, 10], "metadata": [9, 10], "rout": [9, 10], "pleas": [9, 10], "check": [9, 10], "user": [9, 10], "guid": [9, 10], "mechan": [9, 10], "work": [9, 10], "metadatarequest": [9, 10], "encapsul": [9, 10], "inform": [9, 10, 11, 13], "get_param": [9, 10], "deep": [9, 10], "subobject": [9, 10], "param": [9, 10], "dict": [9, 10], "map": [9, 10], "weighted_leav": [9, 10, 13], "aggregate_leaves_first": [9, 10, 13], "duplic": [9, 10], "specifi": [9, 10, 13], "part": [9, 10], "surround": [9, 10], "whichev": [9, 10], "calcul": [9, 10, 13], "assign": [9, 10, 13], "aggreg": [9, 10, 13], "sibl": [9, 10], "small": [9, 10], "effici": [9, 10, 13], "ones": [9, 10], "invers": [9, 10, 13], "oob": [9, 10, 13], "n_quantil": [9, 10], "els": [9, 10], "q": [9, 10, 13], "other": [9, 10], "correspond": [9, 10, 13], "row": [9, 10, 13], "omit": [9, 10], "assum": [9, 10, 13], "ident": [9, 10], "proximity_count": [9, 10, 13], "max_proxim": [9, 10, 13], "return_sort": [9, 10], "proxim": [9, 10], "priorit": [9, 10], "sort": [9, 10], "descend": [9, 10, 13], "arbitrari": [9, 10, 13], "tupl": [9, 10, 13], "detail": [9, 10], "berkelei": [9, 10], "edu": [9, 10], "breiman": [9, 10], "randomforest": [9, 10], "cc_home": [9, 10], "htm": [9, 10], "prox": [9, 10], "quantile_rank": [9, 10, 13], "kind": [9, 10], "rank": [9, 10], "80": [9, 10], "below": [9, 10], "given": [9, 10, 11, 13], "gap": [9, 10], "ti": [9, 10], "exact": [9, 10], "definit": [9, 10], "depend": [9, 10, 12], "option": [9, 10, 13], "keyword": [9, 10], "weak": [9, 10], "strict": [9, 10], "interpret": [9, 10, 13], "percentag": [9, 10], "multipl": [9, 10], "match": [9, 10], "cumul": [9, 10], "similar": [9, 10], "except": [9, 10], "y_rank": [9, 10, 13], "n_train": [9, 10], "coeffici": [9, 10], "determin": [9, 10, 13], "u": [9, 10], "residu": [9, 10], "possibl": [9, 10], "becaus": [9, 10, 12], "arbitrarili": [9, 10], "wors": [9, 10], "constant": [9, 10, 13], "alwai": [9, 10], "expect": [9, 10, 13], "disregard": [9, 10], "some": [9, 10], "precomput": [9, 10], "kernel": [9, 10], "instead": [9, 10, 13], "n_samples_fit": [9, 10], "wrt": [9, 10], "set_fit_request": [9, 10], "str": [9, 10], "unchang": [9, 10], "request": [9, 10], "relev": [9, 10], "enable_metadata_rout": [9, 10], "set_config": [9, 10], "rais": [9, 10], "alia": [9, 10], "origin": [9, 10], "metadata_rout": [9, 10], "retain": [9, 10, 13], "allow": [9, 10, 13], "you": [9, 10], "chang": [9, 10], "insid": [9, 10], "pipelin": [9, 10], "updat": [9, 10], "simpl": [9, 10, 13], "well": [9, 10], "nest": [9, 10], "latter": [9, 10, 13], "form": [9, 10], "compon": [9, 10], "__": [9, 10], "so": [9, 10], "": [9, 10, 13], "set_predict_request": [9, 10], "set_score_request": [9, 10], "poisson": 10, "devianc": 10, "find": [10, 13], "significantli": 10, "slower": 10, "both": [10, 13], "decisiontreeregressor": 10, "extratreesquantileregressor": 10, "extrem": 10, "3592": 10, "scikit": [11, 12, 13], "compat": 11, "instal": 11, "main": 11, "background": 11, "kei": 11, "concept": 11, "doctr": 11, "cython": 12, "0a4": 12, "pip": 12, "To": [12, 13], "manual": 12, "edit": 12, "fail": 12, "ensur": 12, "openbla": 12, "lapack": 12, "access": 12, "On": 12, "maco": 12, "brew": 12, "export": 12, "system_version_compat": 12, "pytest": 12, "doc": 12, "sphinx_requir": 12, "txt": 12, "b": 12, "html": 12, "_build": 12, "proven": 13, "veri": 13, "popular": 13, "power": 13, "accur": 13, "approxim": 13, "respons": 13, "variabl": 13, "That": 13, "we": 13, "let": 13, "covari": 13, "predictor": 13, "howev": 13, "about": 13, "infer": 13, "complet": 13, "alon": 13, "outlier": 13, "detect": 13, "dimension": 13, "practic": 13, "empir": 13, "sever": 13, "wai": 13, "2c": 13, "degre": 13, "freedom": 13, "extend": 13, "straightforward": 13, "rather": 13, "store": 13, "suffici": 13, "statist": 13, "At": 13, "frequenc": 13, "formal": 13, "y_j": 13, "frac": 13, "sum_": 13, "mathbb": 13, "y_i": 13, "denot": 13, "fall": 13, "unknown": 13, "Then": 13, "pair": 13, "same": 13, "wa": 13, "propos": 13, "mei06": 13, "load_diabet": 13, "reg": 13, "initi": 13, "per": 13, "subset": 13, "randomli": 13, "materi": 13, "impact": 13, "notabl": 13, "advantag": 13, "onc": 13, "accordingli": 13, "sinc": 13, "accept": 13, "three": 13, "without": 13, "explicitli": 13, "overwritten": 13, "overrid": 13, "need": 13, "monoton": 13, "accord": 13, "repeat": 13, "co": 13, "occur": 13, "larger": 13, "n_": 13, "gg": 13, "cdot": 13, "leafsampl": 13, "belong": 13, "across": 13, "y_pred_weight": 13, "y_pred_unweight": 13, "allclos": 13, "y_pred_oob": 13, "flag": 13, "arrang": 13, "ib": 13, "x_mix": 13, "y_pred_mix": 13, "y_pred_train_oob": 13, "y_pred_test": 13, "wherebi": 13, "recov": 13, "retrain": 13, "boolean": 13, "befor": 13, "configur": 13, "essenti": 13, "replic": 13, "process": 13, "share": 13, "y_ranks_oob": 13, "present": 13, "were": 13, "respect": 13, "nicolai": 13, "6": 13, "url": 13}, "objects": {"quantile_forest": [[9, 0, 1, "", "ExtraTreesQuantileRegressor"], [10, 0, 1, "", "RandomForestQuantileRegressor"]], "quantile_forest.ExtraTreesQuantileRegressor": [[9, 1, 1, "", "apply"], [9, 2, 1, "", "base_estimator_"], [9, 1, 1, "", "decision_path"], [9, 2, 1, "", "feature_importances_"], [9, 1, 1, "", "fit"], [9, 1, 1, "", "get_metadata_routing"], [9, 1, 1, "", "get_params"], [9, 1, 1, "", "predict"], [9, 1, 1, "", "proximity_counts"], [9, 1, 1, "", "quantile_ranks"], [9, 1, 1, "", "score"], [9, 1, 1, "", "set_fit_request"], [9, 1, 1, "", "set_params"], [9, 1, 1, "", "set_predict_request"], [9, 1, 1, "", "set_score_request"]], "quantile_forest.RandomForestQuantileRegressor": [[10, 1, 1, "", "apply"], [10, 2, 1, "", "base_estimator_"], [10, 1, 1, "", "decision_path"], [10, 2, 1, "", "feature_importances_"], [10, 1, 1, "", "fit"], [10, 1, 1, "", "get_metadata_routing"], [10, 1, 1, "", "get_params"], [10, 1, 1, "", "predict"], [10, 1, 1, "", "proximity_counts"], [10, 1, 1, "", "quantile_ranks"], [10, 1, 1, "", "score"], [10, 1, 1, "", "set_fit_request"], [10, 1, 1, "", "set_params"], [10, 1, 1, "", "set_predict_request"], [10, 1, 1, "", "set_score_request"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:property"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "property", "Python property"]}, "titleterms": {"api": [0, 11], "refer": [0, 11, 13], "quantil": [0, 2, 3, 4, 5, 6, 7, 11, 13], "forest": [0, 2, 4, 5, 6, 11, 13], "gener": 1, "exampl": [1, 11], "regress": [2, 4, 5, 6, 13], "extrapol": 2, "problem": 2, "predict": [3, 4, 5, 7, 13], "differ": 3, "interpol": 3, "method": 3, "interv": 4, "compar": 5, "ground": 5, "truth": 5, "function": 5, "v": 6, "standard": 6, "weight": 7, "unweight": 7, "comput": 8, "time": 8, "extratreesquantileregressor": 9, "randomforestquantileregressor": 10, "get": [11, 12], "start": [11, 12], "user": [11, 13], "guid": [11, 13], "prerequisit": 12, "instal": 12, "develop": 12, "troubleshoot": 12, "test": 12, "coverag": 12, "document": 12, "introduct": 13, "fit": 13, "rank": 13, "proxim": 13, "count": 13}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 60}, "alltitles": {"API Reference": [[0, "api-reference"], [11, "id3"]], "Quantile Forests": [[0, "quantile-forests"]], "General Examples": [[1, "general-examples"]], "Quantile regression forest extrapolation problem": [[2, "quantile-regression-forest-extrapolation-problem"]], "Predicting with different quantile interpolation methods": [[3, "predicting-with-different-quantile-interpolation-methods"]], "Quantile regression forest prediction intervals": [[4, "quantile-regression-forest-prediction-intervals"]], "Quantile regression forest predictions compared to ground truth function": [[5, "quantile-regression-forest-predictions-compared-to-ground-truth-function"]], "Quantile regression forest vs. standard regression forest": [[6, "quantile-regression-forest-vs-standard-regression-forest"]], "Predicting with weighted and unweighted quantiles": [[7, "predicting-with-weighted-and-unweighted-quantiles"]], "Computation times": [[8, "computation-times"]], "ExtraTreesQuantileRegressor": [[9, "extratreesquantileregressor"]], "RandomForestQuantileRegressor": [[10, "randomforestquantileregressor"]], "quantile-forest": [[11, "quantile-forest"]], "Getting Started": [[11, "id1"], [12, "getting-started"]], "User Guide": [[11, "id2"], [13, "user-guide"]], "Examples": [[11, "id4"]], "Prerequisites": [[12, "prerequisites"]], "Install": [[12, "install"]], "Developer Install": [[12, "developer-install"]], "Troubleshooting": [[12, "troubleshooting"]], "Test and Coverage": [[12, "test-and-coverage"]], "Documentation": [[12, "documentation"]], "Introduction": [[13, "introduction"]], "Quantile Regression Forests": [[13, "quantile-regression-forests"]], "Fitting and Predicting": [[13, "fitting-and-predicting"]], "Quantile Ranks": [[13, "quantile-ranks"]], "Proximity Counts": [[13, "proximity-counts"]], "References": [[13, "references"]]}, "indexentries": {"extratreesquantileregressor (class in quantile_forest)": [[9, "quantile_forest.ExtraTreesQuantileRegressor"]], "apply() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.apply"]], "base_estimator_ (quantile_forest.extratreesquantileregressor property)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.base_estimator_"]], "decision_path() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.decision_path"]], "feature_importances_ (quantile_forest.extratreesquantileregressor property)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.feature_importances_"]], "fit() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.fit"]], "get_metadata_routing() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.get_metadata_routing"]], "get_params() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.get_params"]], "predict() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.predict"]], "proximity_counts() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.proximity_counts"]], "quantile_ranks() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.quantile_ranks"]], "score() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.score"]], "set_fit_request() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.set_fit_request"]], "set_params() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.set_params"]], "set_predict_request() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.set_predict_request"]], "set_score_request() (quantile_forest.extratreesquantileregressor method)": [[9, "quantile_forest.ExtraTreesQuantileRegressor.set_score_request"]], "randomforestquantileregressor (class in quantile_forest)": [[10, "quantile_forest.RandomForestQuantileRegressor"]], "apply() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.apply"]], "base_estimator_ (quantile_forest.randomforestquantileregressor property)": [[10, "quantile_forest.RandomForestQuantileRegressor.base_estimator_"]], "decision_path() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.decision_path"]], "feature_importances_ (quantile_forest.randomforestquantileregressor property)": [[10, "quantile_forest.RandomForestQuantileRegressor.feature_importances_"]], "fit() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.fit"]], "get_metadata_routing() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.get_metadata_routing"]], "get_params() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.get_params"]], "predict() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.predict"]], "proximity_counts() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.proximity_counts"]], "quantile_ranks() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.quantile_ranks"]], "score() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.score"]], "set_fit_request() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.set_fit_request"]], "set_params() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.set_params"]], "set_predict_request() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.set_predict_request"]], "set_score_request() (quantile_forest.randomforestquantileregressor method)": [[10, "quantile_forest.RandomForestQuantileRegressor.set_score_request"]]}})