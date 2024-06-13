__all__ = ["AdaBoost"]

from asreview.models.classifiers.base import BaseTrainClassifier
from sklearn.ensemble import AdaBoostClassifier

class AdaBoost(BaseTrainClassifier):
    """AdaBoost classifier

    This classifier is based on the AdaBoostClassifier from scikit-learn.

    """

    name = "adaboost"
    label = "AdaBoost"

    def __init__(self):
        super().__init__()
        self._model = AdaBoostClassifier()