__all__ = ["XGBoost"]

from xgboost import XGBClassifier


class XGBoost(XGBClassifier):
    """
    XGBoost classifier

    Classifier based on the XGBClassifier.

    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of a tree.
        Default: 6.
    learning_rate : float, optional
        Boosting learning rate (xgb's "eta").
        Default: 0.3.
    n_estimators : int, optional
        Number of boosting rounds.
        Default: 100.
    verbosity : int, optional
        The degree of verbosity (0: silent, 1: warning, 2: info, 3: debug).
        Default: 1.
    objective : str, optional
        Specify the learning task and the corresponding objective function.
        Default: 'binary:logistic'.
    gamma : int, optional
        Minimum loss reduction required to make a further partition on a leaf node of
        the tree.
        Default: 0.
    reg_lambda : int, optinal
        L2 regularization term on weights.
        Default: 1
    random_state : int, optional
        Random number seed.
        Default: None.
    """

    name = "xgboost"
    label = "XGBoost"

    def __init__(
        self,
        max_depth=6,
        learning_rate=0.3,
        n_estimators=100,
        verbosity=1,
        objective="binary:logistic",
        gamma=0,
        reg_lambda=1,
        random_state=None,
        **kwargs,
    ):
        super().__init__(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            verbosity=verbosity,
            objective=objective,
            gamma=gamma,
            reg_lambda=reg_lambda,
            random_state=random_state,
            **kwargs,
        )
