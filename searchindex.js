Search.setIndex({"docnames": ["api", "gallery/index", "gallery/plot_quantile_conformalized", "gallery/plot_quantile_example", "gallery/plot_quantile_extrapolation", "gallery/plot_quantile_interpolation", "gallery/plot_quantile_intervals", "gallery/plot_quantile_multioutput", "gallery/plot_quantile_vs_standard", "gallery/plot_quantile_weighting", "generated/quantile_forest.ExtraTreesQuantileRegressor", "generated/quantile_forest.RandomForestQuantileRegressor", "index", "install", "references", "user_guide"], "filenames": ["api.rst", "gallery/index.rst", "gallery/plot_quantile_conformalized.rst", "gallery/plot_quantile_example.rst", "gallery/plot_quantile_extrapolation.rst", "gallery/plot_quantile_interpolation.rst", "gallery/plot_quantile_intervals.rst", "gallery/plot_quantile_multioutput.rst", "gallery/plot_quantile_vs_standard.rst", "gallery/plot_quantile_weighting.rst", "generated/quantile_forest.ExtraTreesQuantileRegressor.rst", "generated/quantile_forest.RandomForestQuantileRegressor.rst", "index.rst", "install.rst", "references.rst", "user_guide.rst"], "titles": ["API Reference", "General Examples", "QRFs for Conformalized Quantile Regression", "Predicting with Quantile Regression Forests", "Extrapolation with Quantile Regression Forests", "Comparing Quantile Interpolation Methods", "Quantile Regression Forests Prediction Intervals", "Multiple-Output Quantile Regression", "Quantile Regression Forests vs. Random Forests", "Weighted vs. Unweighted Quantile Estimates", "ExtraTreesQuantileRegressor", "RandomForestQuantileRegressor", "quantile-forest", "Getting Started", "References", "User Guide"], "terms": {"thi": [0, 2, 5, 7, 10, 11, 12, 15], "i": [0, 2, 4, 7, 8, 9, 10, 11, 12, 13, 15], "full": [0, 4, 15], "document": 0, "packag": [0, 12, 13, 15], "purpos": [1, 12], "introductori": [1, 12], "illustr": [1, 5, 8, 12], "compar": [1, 3], "quantil": [1, 10, 11, 13, 14], "interpol": [1, 10, 11], "method": [1, 9, 10, 11, 12, 15], "extrapol": 1, "regress": [1, 9, 10, 11, 12, 14], "forest": [1, 2, 5, 9, 10, 11, 13, 14], "multipl": [1, 10, 11], "output": [1, 2, 9, 10, 11, 15], "predict": [1, 2, 4, 5, 7, 8, 9, 10, 11, 12], "qrf": [1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 15], "conform": 1, "interv": [1, 2, 3, 4, 7, 10, 11, 15], "v": [1, 4, 10, 11, 13], "random": [1, 3, 4, 7, 9, 10, 11], "weight": [1, 10, 11], "unweight": [1, 10, 11, 15], "estim": [1, 2, 5, 7, 8, 10, 11, 12], "an": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15], "exampl": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15], "demonstr": [2, 3, 4, 7], "us": [2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15], "construct": [2, 15], "reliabl": [2, 8], "cqr": 2, "offer": [2, 15], "attain": 2, "valid": [2, 6, 8, 10, 11], "coverag": 2, "while": [2, 9, 10, 11, 15], "mai": [2, 10, 11, 15], "requir": [2, 10, 11, 13], "addit": [2, 15], "calibr": 2, "notic": [2, 4], "we": [2, 15], "obtain": [2, 10, 11], "level": 2, "e": [2, 10, 11, 15], "percentag": [2, 8, 10, 11], "sampl": [2, 3, 5, 6, 7, 9, 10, 11, 15], "actaulli": 2, "fall": [2, 15], "within": 2, "closer": 2, "target": [2, 6, 7, 10, 11, 15], "adapt": 2, "from": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15], "carl": 2, "mcbride": 2, "elli": 2, "http": [2, 6, 10, 11, 14], "www": [2, 10, 11, 14], "kaggl": 2, "com": 2, "code": [2, 13], "carlmcbrideelli": 2, "import": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15], "altair": [2, 3, 4, 5, 6, 7, 8, 9], "alt": [2, 3, 4, 5, 6, 7, 8, 9], "numpi": [2, 3, 4, 5, 6, 7, 9, 13, 15], "np": [2, 3, 4, 5, 6, 7, 9, 10, 11, 15], "panda": [2, 3, 4, 5, 6, 7, 8, 9], "pd": [2, 3, 4, 5, 6, 7, 8, 9], "sklearn": [2, 3, 6, 7, 8, 9, 10, 11, 15], "dataset": [2, 4, 5, 6, 7, 8, 9, 10, 11, 15], "model_select": [2, 3, 6, 7, 8, 9, 15], "train_test_split": [2, 3, 7, 8, 9, 15], "util": [2, 6, 8, 10, 11], "check_random_st": [2, 6, 8], "quantile_forest": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15], "randomforestquantileregressor": [2, 3, 4, 5, 6, 7, 8, 9, 10, 15], "strategi": 2, "random_st": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15], "0": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15], "rng": [2, 3, 6, 8], "cov_pct": 2, "95": [2, 3, 4, 6, 7], "alpha": 2, "100": [2, 6, 7, 9, 10, 11, 15], "load": [2, 6], "california": [2, 6], "hous": [2, 6], "price": [2, 6], "fetch_california_h": [2, 6, 10, 11], "n_sampl": [2, 3, 4, 6, 7, 8, 9, 10, 11, 15], "min": [2, 4, 6], "size": [2, 3, 4, 5, 6, 7, 10, 11, 15], "1000": [2, 3, 4, 6, 9, 10, 11], "perm": [2, 6], "permut": [2, 6], "x": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15], "data": [2, 4, 5, 6, 9, 10, 11, 12, 15], "y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15], "x_train": [2, 3, 4, 6, 7, 8, 9, 15], "x_test": [2, 3, 4, 6, 7, 8, 9, 15], "y_train": [2, 3, 4, 6, 7, 8, 9, 15], "y_test": [2, 3, 6, 7, 8, 9, 15], "def": [2, 3, 4, 5, 6, 7, 8, 9], "sort_y_valu": 2, "y_pred": [2, 3, 4, 5, 6, 7, 10, 11, 15], "y_pi": 2, "sort": [2, 5, 7, 8, 9, 10, 11], "valu": [2, 4, 5, 6, 7, 8, 9, 10, 11, 15], "indic": [2, 10, 11, 15], "argsort": [2, 6], "return": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15], "y_pred_low": [2, 3, 4, 6, 7], "y_pred_upp": [2, 3, 4, 6, 7], "1": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15], "coverage_scor": 2, "y_true": [2, 4, 6, 7, 10, 11], "effect": [2, 10, 11, 15], "score": [2, 10, 11, 15], "mean": [2, 6, 8, 9, 10, 11, 15], "float": [2, 10, 11, 15], "mean_width_scor": 2, "width": [2, 3, 4, 5, 6, 7, 8, 9], "mean_width": 2, "ab": [2, 4], "qrf_strategi": 2, "2": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15], "fit": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "calcul": [2, 4, 5, 8, 9, 10, 11, 12, 15], "lower": [2, 3, 4, 5, 6, 7, 10, 11], "upper": [2, 3, 4, 5, 6, 7], "test": [2, 3, 4, 10, 11, 12, 15], "y_pred_interv": [2, 6], "stack": 2, "axi": [2, 4, 5, 6, 8, 9], "point": [2, 3, 5, 7, 10, 11], "aggregate_leaves_first": [2, 10, 11, 15], "fals": [2, 3, 4, 5, 6, 7, 9, 10, 11, 15], "cqr_strategi": 2, "creat": [2, 4, 5, 7, 8, 10, 11], "set": [2, 4, 10, 11, 15], "x_calib": 2, "y_calib": 2, "test_siz": [2, 7, 9, 15], "5": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15], "y_pred_interval_calib": 2, "y_pred_low_calib": 2, "y_pred_upp_calib": 2, "b": [2, 13], "conf_scor": 2, "vstack": 2, "t": [2, 3, 4, 7, 15], "max": [2, 4, 6, 11], "get": [2, 4, 6, 7, 10, 11, 12], "": [2, 10, 11, 15], "distribut": [2, 8, 10, 11, 15], "len": [2, 4, 5, 6, 7, 15], "subtract": 2, "add": [2, 10, 11], "y_conf_low": 2, "y_conf_upp": 2, "frame": 2, "arg": 2, "df": [2, 4, 5, 6, 7, 8, 9], "concat": 2, "datafram": [2, 3, 4, 5, 6, 7, 8, 9], "pipe": [2, 6, 9], "lambda": [2, 4, 6, 7, 9], "100_000": [2, 6], "assign": [2, 3, 4, 9, 10, 11, 15], "metric": [2, 10, 11], "merg": 2, "groupbi": [2, 8, 9], "appli": [2, 5, 10, 11], "seri": 2, "reset_index": [2, 9], "plot_prediction_interv": 2, "domain": [2, 4, 6], "click": [2, 5, 7, 8, 9], "selection_point": [2, 5, 7, 8, 9], "field": [2, 5, 7, 8, 9], "y_label": 2, "bind": [2, 5, 7, 8, 9], "legend": [2, 3, 4, 5, 7, 8, 9], "color_circl": 2, "color": [2, 3, 4, 5, 6, 7, 8, 9], "n": [2, 3, 4, 5, 7, 8, 9, 10, 11, 15], "scale": [2, 3, 4, 6, 7, 8], "ye": 2, "No": 2, "rang": [2, 3, 4, 7, 9, 10, 11, 15], "f2a619": [2, 3, 4, 5, 6, 7, 8, 9], "red": [2, 4], "titl": [2, 3, 4, 5, 6, 7, 8, 9], "color_bar": 2, "e0f2ff": [2, 3, 4, 6], "tooltip": [2, 3, 4, 5, 6, 7, 8, 9], "q": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15], "format": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "d": [2, 6, 9], "true": [2, 4, 9, 10, 11, 15], "base": [2, 4, 5, 6, 10, 11, 12, 15], "chart": [2, 3, 4, 5, 6, 7, 8, 9], "transform_calcul": [2, 6, 8], "datum": [2, 4, 6, 8], "circl": [2, 6], "mark_circl": [2, 3, 4, 5, 6, 7], "30": [2, 6], "encod": [2, 3, 4, 5, 6, 7, 8, 9], "nice": [2, 3, 4, 6, 7], "condit": [2, 3, 4, 5, 6, 7, 8, 9, 12, 15], "lightgrai": [2, 5, 7, 8, 9], "opac": [2, 3, 4, 5, 6, 7, 9], "add_param": [2, 5, 7, 8, 9], "bar": [2, 6], "mark_bar": [2, 4, 5, 6, 8], "pad": [2, 4, 6], "clamp": 2, "y2": [2, 3, 4, 5, 6, 7, 9], "none": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15], "8": [2, 3, 4, 6, 10, 11, 13], "tick": [2, 6], "mark_tick": [2, 6], "orient": [2, 6], "horizont": [2, 6], "thick": [2, 6], "006aff": [2, 3, 4, 5, 6, 7, 8, 9], "4": [2, 5, 6, 7, 9, 13], "tick_low": [2, 6], "tick_upp": [2, 6], "diagon": [2, 6], "var1": [2, 6], "var2": [2, 6], "mark_lin": [2, 3, 4, 6, 7, 9], "black": [2, 3, 4, 6, 7], "strokedash": [2, 6], "text_coverag": 2, "transform_aggreg": 2, "coverage_text": 2, "f": [2, 3, 4, 5, 7, 8, 10, 11, 15], "1f": 2, "mark_text": 2, "align": 2, "left": [2, 10, 11], "baselin": 2, "top": 2, "text": 2, "text_with": 2, "width_text": 2, "20": [2, 4, 5], "hconcat": 2, "kei": [2, 5, 7, 8, 9, 12], "int": [2, 4, 6, 10, 11], "all": [2, 6, 7, 10, 11, 12, 15], "ax": [2, 6], "df_i": 2, "queri": [2, 4], "drop": [2, 12], "properti": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "height": [2, 3, 4, 5, 6, 7, 8, 9], "225": 2, "300": [2, 4], "median": [3, 4, 5, 6, 7, 8, 10, 11, 15], "The": [3, 5, 10, 11, 12, 13, 15], "ground": 3, "truth": 3, "function": [3, 7, 10, 11, 15], "gener": [3, 6, 7, 8, 10, 11, 12, 15], "noisi": 3, "bound": [3, 4, 7], "10": [3, 4, 8, 9, 15], "make_toy_dataset": 3, "random_se": 3, "randomst": [3, 10, 11], "uniform": [3, 7], "sin": [3, 4], "sigma": 3, "25": [3, 4, 7, 9, 15], "nois": [3, 4, 7], "lognorm": 3, "exp": 3, "atleast_2d": [3, 4, 7], "x_sampl": 3, "linspac": [3, 4, 7], "y_sampl": 3, "reshap": [3, 7, 8], "max_depth": [3, 7, 10, 11], "3": [3, 5, 6, 7, 10, 11, 12, 13, 15], "min_samples_leaf": [3, 4, 10, 11], "025": [3, 4, 5, 6, 7], "975": [3, 4, 5, 6, 7], "df_train": 3, "y_pred_m": 3, "df_test": 3, "plot_fit_and_interv": 3, "copi": [3, 4, 6], "y_true_label": 3, "y_pred_label": 3, "y_area_label": 3, "point_label": [3, 4], "observ": [3, 4, 6], "3f": [3, 4, 5, 7, 9], "line_tru": [3, 4], "line_pr": [3, 4], "area_pr": 3, "mark_area": [3, 4, 7, 9], "For": [3, 4, 7, 10, 11, 15], "desir": [3, 4, 5, 10, 11, 15], "order": [3, 4, 6, 7, 10, 11, 15], "blank": [3, 4], "resolve_scal": [3, 4], "independ": [3, 4], "400": [3, 5, 7, 8, 9], "650": [3, 7, 8, 9], "toi": [4, 5, 7], "produc": [4, 9], "do": [4, 15], "outsid": 4, "train": [4, 9, 10, 11, 15], "limit": [4, 15], "approach": [4, 15], "fail": [4, 13], "accur": [4, 15], "those": 4, "seed": [4, 7], "15": [4, 5], "extrap_frac": 4, "func": [4, 7], "func_str": 4, "make_func_xi": [4, 7], "std": [4, 9], "01": 4, "normal": [4, 9, 10, 11], "get_train_xi": 4, "min_idx": 4, "max_idx": 4, "get_test_x": 4, "extrap_min_idx": 4, "extrap_max_idx": 4, "exclud": 4, "region": 4, "includ": [4, 9, 15], "them": 4, "max_samples_leaf": [4, 5, 7, 9, 10, 11, 15], "expand_dim": 4, "x_true": 4, "y_func": 4, "concaten": [4, 6, 7, 15], "zero": [4, 10, 11, 15], "ones": [4, 10, 11], "test_left": 4, "test_right": 4, "plot_extrapol": 4, "x_domain": [4, 6], "y_domain": [4, 6], "line_label": 4, "x_scale": 4, "y_scale": 4, "points_color": 4, "line_true_color": 4, "tooltip_tru": 4, "tooltip_pr": 4, "points_tru": 4, "bar_pr": 4, "05": 4, "y_pred_lin": 4, "type": 4, "line": [4, 7, 9], "name": [4, 9, 10, 11], "y_pred_area": 4, "area": [4, 5, 7, 9], "y_extrp_lin": 4, "y_extrp_area": 4, "k": [4, 6, 10], "item": 4, "elif": 4, "200": 4, "kwarg": [4, 15], "chart1": [4, 6], "chart2": [4, 6], "layer": 4, "can": [5, 7, 8, 10, 11, 12, 13, 15], "dure": [5, 10, 11, 15], "when": [5, 9, 10, 11, 12, 15], "li": [5, 10, 11], "between": [5, 8, 10, 11], "two": [5, 7, 10, 11, 15], "In": [5, 7, 8, 10, 11, 15], "singl": [5, 7, 10, 11, 15], "split": [5, 6, 10, 11], "separ": 5, "group": 5, "actual": [5, 6, 8], "ar": [5, 10, 11, 12, 13, 15], "doe": [5, 10, 11, 15], "precis": 5, "correspond": [5, 10, 11, 15], "one": [5, 7, 10, 11, 15], "linear": [5, 10, 11], "ffd237": 5, "higher": [5, 10, 11], "0d4599": 5, "midpoint": [5, 10, 11], "nearest": [5, 10, 11], "a6e5ff": 5, "000000": 5, "updat": [5, 10, 11], "arrai": [5, 10, 11, 15], "est": 5, "n_estim": [5, 6, 8, 9, 10, 11], "bootstrap": [5, 10, 11, 15], "initi": [5, 15], "idx": [5, 6], "enumer": [5, 7, 9], "tolist": 5, "y_med": 5, "y_low": 5, "y_upp": 5, "popul": [5, 9], "result": [5, 9, 10, 11], "differ": 5, "extend": [5, 9, 12, 15], "plot_interpol": 5, "list": [5, 7, 8, 9, 10, 11, 15], "step": 5, "75": [5, 9, 15], "label": [5, 8], "ticksiz": 5, "facet": 5, "column": [5, 9, 10, 11, 15], "header": 5, "labelori": 5, "bottom": 5, "titleori": 5, "featur": [5, 10, 11], "configure_facet": 5, "space": 5, "configure_rang": [5, 7, 8, 9], "categori": [5, 7, 8, 9], "rangeschem": [5, 7, 8, 9], "configure_scal": 5, "bandpaddinginn": 5, "9": 5, "configure_view": 5, "stroke": 5, "how": [6, 10, 11], "inspir": 6, "figur": 6, "meinshausen": [6, 10, 11, 12, 14, 15], "jmlr": [6, 10, 11, 14], "org": [6, 10, 11, 14], "paper": [6, 10, 11, 14], "v7": 6, "meinshausen06a": [6, 10, 11, 14], "html": [6, 13], "kfold": 6, "kf": 6, "n_split": 6, "get_n_split": 6, "fold": 6, "cross": 6, "train_index": 6, "test_index": 6, "set_param": [6, 10, 11], "max_featur": [6, 10, 11], "shape": [6, 10, 11, 15], "y_pred_i": 6, "append": 6, "convert": [6, 9, 10, 11], "dollar": 6, "plot_calibration_and_interv": 6, "plot_calibr": 6, "minimum": [6, 9, 10, 11], "both": [6, 11, 15], "maximum": [6, 9, 10, 11, 15], "plot_interv": 6, "sort_idx": 6, "iloc": 6, "arang": [6, 15], "center": 6, "index": [6, 10, 11, 15], "y_pred_width": 6, "250": 6, "325": 6, "regressor": [7, 8, 9, 10, 11, 15], "variabl": [7, 15], "each": [7, 10, 11, 15], "simultan": 7, "ha": [7, 10, 11], "three": 7, "2500": 7, "defin": [7, 10, 11], "map": [7, 9, 10, 11], "signal": 7, "log1p": 7, "sqrt": [7, 10, 11], "empti": 7, "tile": 7, "squeez": 7, "plot_multioutput": 7, "symbolopac": [7, 9], "multi": [7, 15], "comparison": [8, 9], "standard": [8, 9, 15], "synthet": 8, "right": [8, 10, 11], "skew": 8, "As": 8, "greater": [8, 10, 11], "overlap": 8, "frequenc": [8, 15], "more": [8, 10, 11, 15], "than": [8, 10, 11, 15], "scipi": [8, 10, 11, 13], "sp": 8, "ensembl": [8, 9, 10, 11, 12, 15], "randomforestregressor": [8, 9, 15], "5000": 8, "loc": 8, "skewnorm_rv": 8, "stat": [8, 10, 11], "skewnorm": 8, "rv": 8, "randn": 8, "regr_rf": 8, "regr_qrf": 8, "y_pred_rf": [8, 15], "rf": [8, 9, 15], "y_pred_qrf": [8, 15], "c0c0c0": 8, "plot_prediction_histogram": 8, "round": [8, 10, 11], "as_": 8, "transform_fold": 8, "transform_joinaggreg": 8, "total": [8, 9, 10, 11, 15], "count": [8, 10, 11, 12], "pct": 8, "o": 8, "labelangl": 8, "labelexpr": 8, "null": 8, "sum": [8, 10, 11, 15], "xoffset": 8, "bin": 8, "runtim": 9, "comput": [9, 10, 11, 12, 15], "ident": [9, 10, 11], "rel": [9, 10, 11], "depend": [9, 10, 11, 13], "number": [9, 10, 11, 15], "leaf": [9, 10, 11, 15], "A": [9, 10, 11, 12, 15], "time": [9, 12, 15], "contextlib": 9, "contextmanag": 9, "t0": 9, "yield": 9, "t1": 9, "make_regress": [9, 15], "500": 9, "n_featur": [9, 10, 11], "n_target": [9, 15], "001751": 9, "est_siz": 9, "50": 9, "n_repeat": 9, "over": [9, 10, 11], "iter": 9, "j": [9, 10, 11, 15], "rf_time": 9, "_": 9, "qrf_weighted_tim": 9, "weighted_quantil": [9, 10, 11, 15], "qrf_unweighted_tim": 9, "zip": 9, "millisecond": 9, "second": 9, "agg": 9, "set_axi": 9, "join": 9, "str": [9, 10, 11], "col": 9, "runtime_mean": 9, "runtime_std": 9, "ymin": 9, "ymax": 9, "plot_timings_by_s": 9, "averag": [9, 10, 11, 15], "class": [10, 11, 12], "default_quantil": [10, 11, 15], "criterion": [10, 11], "squared_error": [10, 11], "min_samples_split": [10, 11], "min_weight_fraction_leaf": [10, 11], "max_leaf_nod": [10, 11], "min_impurity_decreas": [10, 11], "oob_scor": [10, 11, 15], "n_job": [10, 11], "verbos": [10, 11, 13], "warm_start": [10, 11], "ccp_alpha": [10, 11], "max_sampl": [10, 11], "extra": 10, "tree": [10, 11, 12, 15], "provid": [10, 11, 12, 15], "implement": [10, 12], "meta": [10, 11], "decis": [10, 11, 15], "variou": [10, 11], "sub": [10, 11], "improv": [10, 11], "accuraci": [10, 11], "control": [10, 11], "paramet": [10, 11, 15], "default": [10, 11, 15], "model": [10, 11], "tri": [10, 11], "must": [10, 11], "strictli": [10, 11], "If": [10, 11, 13, 15], "absolute_error": [10, 11], "friedman_ms": [10, 11], "poisson": [10, 11], "measur": [10, 11], "qualiti": [10, 11], "support": [10, 11, 15], "criteria": [10, 11], "squar": [10, 11], "error": [10, 11], "which": [10, 11, 15], "equal": [10, 11, 15], "varianc": [10, 11, 15], "reduct": [10, 11], "select": [10, 11, 15], "minim": [10, 11], "l2": [10, 11], "loss": [10, 11], "termin": [10, 11], "node": [10, 11, 15], "friedman": [10, 11], "potenti": [10, 11], "absolut": [10, 11], "l1": [10, 11], "devianc": [10, 11], "find": [10, 11, 15], "significantli": [10, 11], "slower": [10, 11], "depth": [10, 11], "expand": [10, 11], "until": [10, 11], "leav": [10, 11, 15], "pure": [10, 11], "contain": [10, 11], "less": [10, 11, 15], "intern": [10, 11], "consid": [10, 11], "fraction": [10, 11, 15], "ceil": [10, 11], "ani": [10, 11, 15], "onli": [10, 11, 15], "least": [10, 11], "branch": [10, 11], "have": [10, 11, 15], "smooth": [10, 11], "especi": [10, 11], "permit": [10, 11], "unlimit": [10, 11], "input": [10, 11, 15], "sample_weight": [10, 11], "log2": [10, 11], "look": [10, 11], "best": [10, 11], "auto": [10, 11], "equival": [10, 11], "bag": [10, 11, 12], "achiev": [10, 11], "smaller": [10, 11, 15], "g": [10, 11], "note": [10, 11, 15], "search": [10, 11], "stop": [10, 11], "partit": [10, 11], "found": [10, 11], "even": [10, 11], "inspect": [10, 11], "grow": [10, 11], "first": [10, 11, 15], "fashion": [10, 11], "impur": [10, 11], "induc": [10, 11], "decreas": [10, 11], "equat": [10, 11], "follow": [10, 11, 13], "n_t": [10, 11], "n_t_r": [10, 11], "right_impur": [10, 11], "n_t_l": [10, 11], "left_impur": [10, 11], "where": [10, 11, 15], "current": [10, 11], "child": [10, 11], "refer": [10, 11], "pass": [10, 11, 15], "bool": [10, 11], "whether": [10, 11], "build": [10, 11, 13, 15], "whole": [10, 11], "callabl": [10, 11], "out": [10, 11, 12], "By": [10, 11, 15], "accuracy_scor": 10, "signatur": [10, 11], "custom": [10, 11], "avail": [10, 11, 12, 13, 15], "job": [10, 11], "run": [10, 11, 13], "parallel": [10, 11], "decision_path": [10, 11], "unless": [10, 11], "joblib": [10, 11], "parallel_backend": [10, 11], "context": [10, 11], "processor": [10, 11], "instanc": [10, 11], "sourc": [10, 13], "draw": [10, 11], "reus": [10, 11], "solut": [10, 11], "previou": [10, 11], "call": [10, 11, 15], "otherwis": [10, 11], "just": [10, 11, 12], "new": [10, 11, 15], "non": [10, 11, 12, 15], "neg": [10, 11], "complex": [10, 11], "cost": [10, 11], "prune": [10, 11], "subtre": [10, 11], "largest": [10, 11], "chosen": [10, 11], "perform": [10, 11, 12, 15], "thu": [10, 11], "should": [10, 11], "optim": [10, 12, 15], "journal": [10, 11, 14], "machin": [10, 11, 14], "learn": [10, 11, 12, 13, 14, 15], "research": [10, 11, 14], "7": [10, 11, 14], "jun": [10, 11], "983": [10, 11, 14], "999": [10, 11, 14], "2006": [10, 11, 14], "volume7": [10, 11, 14], "pdf": [10, 11, 14], "return_x_i": [10, 11, 15], "3352": 10, "attribut": [10, 11], "estimator_": [10, 11], "extratreeregressor": 10, "templat": [10, 11], "collect": [10, 11], "estimators_": [10, 11], "decisiontreeregressor": [10, 11], "feature_importances_": [10, 11], "ndarrai": [10, 11], "n_features_in_": [10, 11], "seen": [10, 11], "feature_names_in_": [10, 11], "string": [10, 11], "n_outputs_": [10, 11], "oob_score_": [10, 11], "exist": [10, 11], "oob_prediction_": [10, 11], "n_output": [10, 11], "like": [10, 11, 15], "spars": [10, 11], "matrix": [10, 11], "its": [10, 11, 15], "dtype": [10, 11], "float32": [10, 11], "csr_matrix": [10, 11], "x_leav": [10, 11], "datapoint": [10, 11], "end": [10, 11], "up": [10, 11], "path": [10, 11], "version": [10, 11, 12], "18": [10, 11], "n_node": [10, 11], "element": [10, 11], "goe": [10, 11], "through": [10, 11], "csr": [10, 11], "n_nodes_ptr": [10, 11], "give": [10, 11, 15], "th": [10, 11, 15], "estimators_samples_": [10, 11], "subset": [10, 11, 15], "drawn": [10, 11], "dynam": [10, 11], "identifi": [10, 11], "member": [10, 11], "re": [10, 11], "reduc": [10, 11], "object": [10, 11, 15], "memori": [10, 11], "footprint": [10, 11], "store": [10, 11, 15], "fetch": [10, 11], "expect": [10, 11, 15], "brought": [10, 11], "It": [10, 11, 15], "also": [10, 11, 15], "known": [10, 11], "gini": [10, 11], "warn": [10, 11], "mislead": [10, 11], "high": [10, 11, 12, 15], "cardin": [10, 11], "mani": [10, 11, 15], "uniqu": [10, 11], "see": [10, 11], "permutation_import": [10, 11], "altern": [10, 11], "consist": [10, 11], "root": [10, 11], "case": [10, 11], "sparse_pickl": [10, 11], "csc_matrix": [10, 11], "real": [10, 11, 15], "would": [10, 11], "net": [10, 11], "ignor": [10, 11], "classif": [10, 11, 15], "thei": [10, 11, 12, 15], "carri": [10, 11], "either": [10, 11], "pickl": [10, 11], "underli": [10, 11], "structur": [10, 11], "self": [10, 11], "get_metadata_rout": [10, 11], "metadata": [10, 11], "rout": [10, 11], "pleas": [10, 11], "check": [10, 11, 12], "user": [10, 11, 12], "guid": [10, 11, 12], "mechan": [10, 11], "work": [10, 11], "metadatarequest": [10, 11], "encapsul": [10, 11], "inform": [10, 11, 12, 15], "get_param": [10, 11], "deep": [10, 11], "subobject": [10, 11], "param": [10, 11], "dict": [10, 11], "weighted_leav": [10, 11, 15], "duplic": [10, 11], "specifi": [10, 11, 15], "part": [10, 11], "surround": [10, 11], "whichev": [10, 11], "aggreg": [10, 11, 15], "sibl": [10, 11], "small": [10, 11], "effici": [10, 11, 15], "invers": [10, 11, 15], "oob": [10, 11, 15], "n_quantil": [10, 11], "els": [10, 11], "other": [10, 11], "row": [10, 11], "omit": [10, 11], "assum": [10, 11, 15], "proximity_count": [10, 11, 15], "max_proxim": [10, 11, 15], "return_sort": [10, 11], "proxim": [10, 11, 12], "priorit": [10, 11], "descend": [10, 11, 15], "arbitrari": [10, 11, 12, 15], "tupl": [10, 11, 15], "detail": [10, 11, 12], "berkelei": [10, 11], "edu": [10, 11], "breiman": [10, 11], "randomforest": [10, 11], "cc_home": [10, 11], "htm": [10, 11], "prox": [10, 11, 15], "quantile_rank": [10, 11, 15], "kind": [10, 11], "rank": [10, 11, 12], "80": [10, 11], "below": [10, 11, 15], "given": [10, 11, 15], "gap": [10, 11], "ti": [10, 11], "exact": [10, 11], "definit": [10, 11], "option": [10, 11, 15], "keyword": [10, 11], "weak": [10, 11], "strict": [10, 11], "interpret": [10, 11, 15], "match": [10, 11], "cumul": [10, 11], "similar": [10, 11], "except": [10, 11], "y_rank": [10, 11, 15], "n_train": [10, 11], "coeffici": [10, 11], "determin": [10, 11, 15], "r": [10, 11, 13], "u": [10, 11], "residu": [10, 11], "possibl": [10, 11], "becaus": [10, 11, 13], "arbitrarili": [10, 11], "wors": [10, 11], "constant": [10, 11, 15], "alwai": [10, 11], "disregard": [10, 11], "some": [10, 11], "precomput": [10, 11], "kernel": [10, 11], "instead": [10, 11, 15], "n_samples_fit": [10, 11], "wrt": [10, 11], "set_fit_request": [10, 11], "unchang": [10, 11], "request": [10, 11], "relev": [10, 11], "enable_metadata_rout": [10, 11], "set_config": [10, 11], "rais": [10, 11], "alia": [10, 11], "origin": [10, 11], "metadata_rout": [10, 11], "retain": [10, 11, 15], "allow": [10, 11, 15], "you": [10, 11, 12], "chang": [10, 11], "insid": [10, 11], "pipelin": [10, 11], "simpl": [10, 11, 15], "well": [10, 11], "nest": [10, 11], "latter": [10, 11, 15], "form": [10, 11], "compon": [10, 11], "__": [10, 11], "so": [10, 11], "set_predict_request": [10, 11], "set_score_request": [10, 11], "r2_score": 11, "extratreesquantileregressor": 11, "extrem": 11, "3592": 11, "scikit": [12, 13, 15], "compat": 12, "parametr": 12, "applic": 12, "dimension": [12, 15], "uncertainti": 12, "cython": [12, 13], "describ": 12, "mei06": [12, 14, 15], "without": [12, 15], "retrain": [12, 15], "serv": 12, "replac": 12, "variant": 12, "start": 12, "instal": 12, "instruct": 12, "concept": 12, "behind": 12, "api": 12, "want": 12, "python": 13, "23": 13, "pip": 13, "addition": 13, "0a4": 13, "To": [13, 15], "manual": 13, "edit": 13, "ensur": 13, "openbla": 13, "lapack": 13, "access": 13, "On": 13, "maco": 13, "brew": 13, "export": 13, "system_version_compat": 13, "m": 13, "pytest": 13, "doc": 13, "rst": 13, "sphinx_requir": 13, "txt": 13, "mkdir": 13, "p": 13, "_imag": 13, "sphinx": 13, "_build": 13, "nicolai": 14, "6": 14, "url": 14, "proven": 15, "veri": 15, "popular": 15, "power": 15, "approxim": 15, "respons": 15, "That": 15, "let": 15, "covari": 15, "predictor": 15, "howev": 15, "about": 15, "infer": 15, "complet": 15, "alon": 15, "outlier": 15, "detect": 15, "practic": 15, "empir": 15, "sever": 15, "wai": 15, "2c": 15, "c": 15, "degre": 15, "freedom": 15, "straightforward": 15, "rather": 15, "suffici": 15, "statist": 15, "At": 15, "formal": 15, "y_j": 15, "frac": 15, "sum_": 15, "mathbb": 15, "l": 15, "y_i": 15, "denot": 15, "unknown": 15, "same": 15, "across": 15, "wa": 15, "propos": 15, "inher": 15, "relat": 15, "ll": 15, "discuss": 15, "load_diabet": 15, "reg": 15, "per": 15, "randomli": 15, "enabl": 15, "materi": 15, "impact": 15, "notabl": 15, "advantag": 15, "onc": 15, "accordingli": 15, "sinc": 15, "accept": 15, "explicitli": 15, "ndim": 15, "overwritten": 15, "overrid": 15, "need": 15, "monoton": 15, "final": 15, "reg_multi": 15, "co": 15, "occur": 15, "larger": 15, "n_": 15, "gg": 15, "cdot": 15, "leafsampl": 15, "belong": 15, "y_pred_weight": 15, "y_pred_unweight": 15, "allclos": 15, "repeat": 15, "accord": 15, "y_pred_oob": 15, "flag": 15, "arrang": 15, "ib": 15, "x_mix": 15, "y_pred_mix": 15, "y_pred_train_oob": 15, "y_pred_test": 15, "wherebi": 15, "recov": 15, "boolean": 15, "befor": 15, "configur": 15, "essenti": 15, "replic": 15, "process": 15, "share": 15, "y_ranks_oob": 15, "present": 15, "were": 15, "respect": 15}, "objects": {"quantile_forest": [[10, 0, 1, "", "ExtraTreesQuantileRegressor"], [11, 0, 1, "", "RandomForestQuantileRegressor"]], "quantile_forest.ExtraTreesQuantileRegressor": [[10, 1, 1, "", "apply"], [10, 1, 1, "", "decision_path"], [10, 2, 1, "", "estimators_samples_"], [10, 2, 1, "", "feature_importances_"], [10, 1, 1, "", "fit"], [10, 1, 1, "", "get_metadata_routing"], [10, 1, 1, "", "get_params"], [10, 1, 1, "", "predict"], [10, 1, 1, "", "proximity_counts"], [10, 1, 1, "", "quantile_ranks"], [10, 1, 1, "", "score"], [10, 1, 1, "", "set_fit_request"], [10, 1, 1, "", "set_params"], [10, 1, 1, "", "set_predict_request"], [10, 1, 1, "", "set_score_request"]], "quantile_forest.RandomForestQuantileRegressor": [[11, 1, 1, "", "apply"], [11, 1, 1, "", "decision_path"], [11, 2, 1, "", "estimators_samples_"], [11, 2, 1, "", "feature_importances_"], [11, 1, 1, "", "fit"], [11, 1, 1, "", "get_metadata_routing"], [11, 1, 1, "", "get_params"], [11, 1, 1, "", "predict"], [11, 1, 1, "", "proximity_counts"], [11, 1, 1, "", "quantile_ranks"], [11, 1, 1, "", "score"], [11, 1, 1, "", "set_fit_request"], [11, 1, 1, "", "set_params"], [11, 1, 1, "", "set_predict_request"], [11, 1, 1, "", "set_score_request"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:property"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "property", "Python property"]}, "titleterms": {"api": 0, "refer": [0, 14], "quantil": [0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15], "forest": [0, 3, 4, 6, 8, 12, 15], "gener": 1, "exampl": 1, "qrf": 2, "conform": 2, "regress": [2, 3, 4, 6, 7, 8, 15], "predict": [3, 6, 15], "extrapol": 4, "compar": 5, "interpol": 5, "method": 5, "interv": 6, "multipl": 7, "output": 7, "v": [8, 9], "random": [8, 15], "weight": [9, 15], "unweight": 9, "estim": [9, 15], "extratreesquantileregressor": 10, "randomforestquantileregressor": 11, "get": 13, "start": 13, "prerequisit": 13, "instal": 13, "develop": 13, "troubleshoot": 13, "test": 13, "coverag": 13, "document": 13, "user": 15, "guid": 15, "introduct": 15, "fit": 15, "model": 15, "make": 15, "out": 15, "bag": 15, "rank": 15, "proxim": 15, "count": 15}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 60}, "alltitles": {"API Reference": [[0, "api-reference"]], "Quantile Forests": [[0, "quantile-forests"]], "General Examples": [[1, "general-examples"]], "QRFs for Conformalized Quantile Regression": [[2, "qrfs-for-conformalized-quantile-regression"]], "Predicting with Quantile Regression Forests": [[3, "predicting-with-quantile-regression-forests"]], "Extrapolation with Quantile Regression Forests": [[4, "extrapolation-with-quantile-regression-forests"]], "Comparing Quantile Interpolation Methods": [[5, "comparing-quantile-interpolation-methods"]], "Quantile Regression Forests Prediction Intervals": [[6, "quantile-regression-forests-prediction-intervals"]], "Multiple-Output Quantile Regression": [[7, "multiple-output-quantile-regression"]], "Quantile Regression Forests vs. Random Forests": [[8, "quantile-regression-forests-vs-random-forests"]], "Weighted vs. Unweighted Quantile Estimates": [[9, "weighted-vs-unweighted-quantile-estimates"]], "ExtraTreesQuantileRegressor": [[10, "extratreesquantileregressor"]], "RandomForestQuantileRegressor": [[11, "randomforestquantileregressor"]], "quantile-forest": [[12, "quantile-forest"]], "Getting Started": [[13, "getting-started"]], "Prerequisites": [[13, "prerequisites"]], "Install": [[13, "id1"]], "Developer Install": [[13, "developer-install"]], "Troubleshooting": [[13, "troubleshooting"]], "Test and Coverage": [[13, "test-and-coverage"]], "Documentation": [[13, "documentation"]], "References": [[14, "references"]], "User Guide": [[15, "user-guide"]], "Introduction": [[15, "introduction"]], "Quantile Regression Forests": [[15, "quantile-regression-forests"]], "Fitting and Predicting": [[15, "fitting-and-predicting"]], "Fitting a Model": [[15, "fitting-a-model"]], "Making Predictions": [[15, "making-predictions"]], "Quantile Weighting": [[15, "quantile-weighting"]], "Out-of-Bag Estimation": [[15, "out-of-bag-estimation"]], "Random Forest Predictions": [[15, "random-forest-predictions"]], "Quantile Ranks": [[15, "quantile-ranks"]], "Proximity Counts": [[15, "proximity-counts"]]}, "indexentries": {"extratreesquantileregressor (class in quantile_forest)": [[10, "quantile_forest.ExtraTreesQuantileRegressor"]], "apply() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.apply"]], "decision_path() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.decision_path"]], "estimators_samples_ (quantile_forest.extratreesquantileregressor property)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.estimators_samples_"]], "feature_importances_ (quantile_forest.extratreesquantileregressor property)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.feature_importances_"]], "fit() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.fit"]], "get_metadata_routing() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.get_metadata_routing"]], "get_params() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.get_params"]], "predict() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.predict"]], "proximity_counts() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.proximity_counts"]], "quantile_ranks() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.quantile_ranks"]], "score() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.score"]], "set_fit_request() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.set_fit_request"]], "set_params() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.set_params"]], "set_predict_request() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.set_predict_request"]], "set_score_request() (quantile_forest.extratreesquantileregressor method)": [[10, "quantile_forest.ExtraTreesQuantileRegressor.set_score_request"]], "randomforestquantileregressor (class in quantile_forest)": [[11, "quantile_forest.RandomForestQuantileRegressor"]], "apply() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.apply"]], "decision_path() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.decision_path"]], "estimators_samples_ (quantile_forest.randomforestquantileregressor property)": [[11, "quantile_forest.RandomForestQuantileRegressor.estimators_samples_"]], "feature_importances_ (quantile_forest.randomforestquantileregressor property)": [[11, "quantile_forest.RandomForestQuantileRegressor.feature_importances_"]], "fit() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.fit"]], "get_metadata_routing() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.get_metadata_routing"]], "get_params() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.get_params"]], "predict() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.predict"]], "proximity_counts() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.proximity_counts"]], "quantile_ranks() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.quantile_ranks"]], "score() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.score"]], "set_fit_request() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.set_fit_request"]], "set_params() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.set_params"]], "set_predict_request() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.set_predict_request"]], "set_score_request() (quantile_forest.randomforestquantileregressor method)": [[11, "quantile_forest.RandomForestQuantileRegressor.set_score_request"]]}})