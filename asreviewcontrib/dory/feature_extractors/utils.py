from sentence_transformers import quantize_embeddings
from sklearn.base import BaseEstimator, TransformerMixin


class Quantizer(BaseEstimator, TransformerMixin):
    def __init__(self, precision="float32"):
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return quantize_embeddings(X, precision=self.precision)
