__all__ = ["ScaledNaiveBayes"]

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class ScaledNaiveBayes(Pipeline):
    """Naive Bayes classifier wrapped in a Scikit-Learn pipeline with minmax normalizer.

    This class extends Pipeline to normalize the feature matrix before applying
    the Naive Bayes classifier.
    """

    name = "scaled-nb"
    label = "Scaled Naive Bayes"

    def __init__(self, **kwargs):
        scaler_params = {"feature_range": (1e-9, 1)}
        classifier_params = {}

        for key, value in kwargs.items():
            if key.startswith("scaler__"):
                scaler_params[key.split("__", 1)[1]] = value
            elif key.startswith("classifier__"):
                classifier_params[key.split("__", 1)[1]] = value
            else:
                classifier_params[key] = value
        
        if "feature_range" in scaler_params:
            scaler_params["feature_range"] = tuple(scaler_params["feature_range"])

        super().__init__(
            [
                (
                    "scaler",
                    MinMaxScaler(**scaler_params),
                ),
                ("classifier", MultinomialNB(**classifier_params)),
            ]
        )

    def fit(self, X, y, sample_weight=None):
        """Forward sample_weight parameter since we use a Pipeline"""
        if sample_weight is not None:
            return super().fit(X, y, classifier__sample_weight=sample_weight)
        return super().fit(X, y)
    
    def get_params(self, deep=True, instances=False):
        """Get parameters for this pipeline.
        
        Parameters
        ----------
        deep: bool, default=True
            If True, will return the parameters for this pipeline and
            contained subobjects that are estimators.
        instances: bool, default=False
            If True, will return the instances of the estimators in the pipeline.
        """
        params = super().get_params(deep=deep)
        if not instances:
            params.pop("scaler", None)
            params.pop("classifier", None)
        return params
