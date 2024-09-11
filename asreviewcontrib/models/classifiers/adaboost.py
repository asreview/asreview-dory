__all__ = ["AdaBoost"]

from asreview.models.classifiers.base import BaseTrainClassifier
from sklearn.ensemble import AdaBoostClassifier

class AdaBoost(BaseTrainClassifier):
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
    algorithm : {'SAMME', 'SAMME.R'}, optional
        The boosting algorithm to use.
        Default: 'SAMME.R'.
    random_state : int or None, optional
        Controls the random seed given to the base estimator.
        Default: None.
    """

    name = "adaboost"
    label = "AdaBoost"

    def __init__(
        self,     
        estimator = None,
        n_estimators = 50,
        learning_rate = 1.0,
        algorithm = 'SAMME',
        random_state = None   
    ):
        super().__init__()
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state

        self._model = AdaBoostClassifier(
            estimator=self.estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state
        )

    def fit(self, X, y):
        if self.estimator is not None and hasattr(self.estimator, 'class_weight'):
            self.estimator.set_params(class_weight="balanced")
        self._model.fit(X,y)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X)