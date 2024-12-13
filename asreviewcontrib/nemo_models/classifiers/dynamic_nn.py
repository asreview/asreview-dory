__all__ = ["DynamicNNClassifier"]

from math import ceil, log10

import numpy as np
import scipy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from asreview.models.classifiers.base import BaseTrainClassifier
from asreview.models.classifiers.utils import _set_class_weight


class DynamicNNClassifier(BaseTrainClassifier):
    """
    Dynamic Neural Network Classifier

    Fully connected neural network classifier
    with dynamic layer count (``dynamic-nn``) .

    Parameters
    ----------
    verbose : int, optional
        Whether to print progress and model information.
        Default: 0
    patience : int, optional
        How much patience to apply to the model.
        Default: 5
    epochs : int, optional
        Number of epochs.
        Default: 100
    batch_size : int, optional
        Batch size.
        Default: 32
    shuffle : boolean, optional
        Whether to shuffle .
        Default: True
    min_delta : float, optional
        Which min_delta to use.
        Default: 0.01
    class_weight : boolean, optional
        Class weight value.
        Default: 30.0
    max_features : int, optional
        Maximum number of features the model allows.
        Default: 1024
    """

    name = "dynamic-nn"
    label = "Fully connected neural network (dynamic layer count)"

    def __init__(
        self,
        verbose=0,
        patience=5,
        epochs=100,
        batch_size=32,
        shuffle=True,
        min_delta=0.01,
        class_weight=30.0,
        max_features=1024,
    ):
        super().__init__()
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_delta = min_delta
        self.verbose = verbose
        self._model = None
        self.class_weight = class_weight
        self.max_features = max_features

    def fit(self, X, y):
        if X.shape[1] > self.max_features:
            raise ValueError(
                f"Feature size too large: {X.shape[1]} features. "
                f"Maximum allowed is {self.max_features}."
            )

        num_layers = min(3, ceil(log10(max(10, X.shape[0]))))

        if scipy.sparse.issparse(X):
            X = X.toarray()
        if self._model is None or X.shape[1] != self.input_dim:
            self.input_dim = X.shape[1]
            self._model = _create_model(X.shape[1], num_layers, self.verbose)

        callback = EarlyStopping(
            monitor="loss",
            patience=self.patience,
            min_delta=self.min_delta,
            restore_best_weights=True,
        )

        self._model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            callbacks=[callback],
            class_weight=_set_class_weight(self.class_weight),
        )

    def predict_proba(self, X):
        if scipy.sparse.issparse(X):
            X = X.toarray()
        pos_pred = self._model.predict(X, verbose=self.verbose)
        neg_pred = 1 - pos_pred
        return np.hstack([neg_pred, pos_pred])


def _create_model(input_dim, num_layers, verbose=1):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(64, activation="relu"))

    for _ in range(num_layers - 1):
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    if verbose == 1:
        model.summary()

    return model
