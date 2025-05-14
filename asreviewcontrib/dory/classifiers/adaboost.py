__all__ = ["AdaBoost"]

from sklearn.ensemble import AdaBoostClassifier


class AdaBoost(AdaBoostClassifier):
    """AdaBoost classifier

    Classifier based on the AdaBoostClassifier from scikit-learn.

    Parameters
    ----------
    estimator : object, optional
        The base estimator from which the boosted ensemble is built.
        Default: None (uses `DecisionTreeClassifier`).
    n_estimators : int, optional
        The maximum number of estimators at which boosting is terminated.
        Default: 50.
    learning_rate : float, optional
        Learning rate shrinks the contribution of each classifier.
        Default: 1.0.
    random_state : int or None, optional
        Controls the random seed given to the base estimator.
        Default: None.
    """

    name = "adaboost"
    label = "AdaBoost"

    def __init__(
        self,
        estimator=None,
        n_estimators=50,
        learning_rate=1.0,
        random_state=None,
        **kwargs,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs,
        )
