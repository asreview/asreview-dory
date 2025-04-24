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

    def __init__(self, feature_range=(1e-9, 1), alpha=3.822, **kwargs):
        self.feature_range = feature_range
        self.alpha = alpha
        super().__init__(
            [
                (
                    "scaler",
                    MinMaxScaler(feature_range=self.feature_range),
                ),
                ("classifier", MultinomialNB(alpha=self.alpha)),
            ]
        )

    def fit(self, X, y, sample_weight=None):
        """Forward sample_weight parameter since we use a Pipeline"""
        if sample_weight is not None:
            return super().fit(X, y, classifier__sample_weight=sample_weight)
        return super().fit(X, y)

