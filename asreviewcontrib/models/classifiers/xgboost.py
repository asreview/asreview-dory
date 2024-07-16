__all__ = ["XGBoost"]

from asreview.models.classifiers.base import BaseTrainClassifier

import xgboost as xgb

class XGBoost(BaseTrainClassifier):
    """XGBoost classifier

    This classifier is based on the XGBClassifier.

    """

    name = "xgboost"
    label = "NEMO: XGBoost"

    def __init__(self):
        super().__init__()
        self._model = xgb.XGBClassifier()
