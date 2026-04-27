import numpy as np
import pandas as pd
from sentence_transformers import quantize_embeddings
from sklearn.base import BaseEstimator, TransformerMixin


def clean_text_inputs(X):
    if isinstance(X, pd.Series):
        X = X.fillna("").astype(str).tolist()
    elif isinstance(X, list):
        X = ["" if x is None else str(x) for x in X]
    elif isinstance(X, np.ndarray):
        X = ["" if x is None else str(x) for x in X.tolist()]
    else:
        raise ValueError("Expected a list or ndarray of strings or pandas Series.")
    return X


class Quantizer(BaseEstimator, TransformerMixin):
    def __init__(self, precision="float32"):
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return quantize_embeddings(X, precision=self.precision)
