__all__ = ["NN2LayerClassifier"]

from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import numpy as np
import scipy

from asreview.models.classifiers.base import BaseTrainClassifier
from asreview.models.classifiers.utils import _set_class_weight

class NN2LayerClassifier(BaseTrainClassifier):
    """Fully connected neural network (2 hidden layers) classifier (``nn-2-layer``).

    Neural network with two hidden, dense layers of the same size.

    Recommended feature extraction model is
    :class:`asreview.models.feature_extraction.Doc2Vec`.

    .. note::

        This model requires ``tensorflow`` to be installed. Use ``pip install
        asreview[tensorflow]`` or install all optional ASReview dependencies
        with ``pip install asreview[all]``

    .. warning::

        Might crash on some systems with limited memory in
        combination with :class:`asreview.models.feature_extraction.Tfidf`.

    Arguments
    ---------
    dense_width: int
        Size of the dense layers.
    optimizer: str
        Name of the Keras optimizer.
    learn_rate: float
        Learning rate multiplier of the default learning rate.
    regularization: float
        Strength of the regularization on the weights and biases.
    verbose: int
        Verbosity of the model mirroring the values for Keras.
    epochs: int
        Number of epochs to train the neural network.
    batch_size: int
        Batch size used for the neural network.
    shuffle: bool
        Whether to shuffle the training data prior to training.
    class_weight: float
        Class weights for inclusions (1's).
    """

    name = "nn-2-layer"
    label = "Fully connected neural network (2 hidden layers)"

    def __init__(self,
                 dense_width=128,
                 optimizer='rmsprop',
                 learn_rate=1.0,
                 regularization=0.01,
                 verbose=0,
                 epochs=35,
                 batch_size=32,
                 shuffle=False,
                 class_weight=30.0):

        super().__init__()
        self.dense_width = int(dense_width)
        self.optimizer = optimizer
        self.learn_rate = learn_rate
        self.regularization = regularization
        self.verbose = verbose
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.class_weight = class_weight

        self._model = None
        self.input_dim = None

    def fit(self, X, y):
        if scipy.sparse.issparse(X):
            X = X.toarray()
        if self._model is None or X.shape[1] != self.input_dim:
            self.input_dim = X.shape[1]
            self._model = _create_dense_nn_model(
                self.input_dim,
                self.dense_width,
                self.optimizer,
                self.learn_rate,
                self.regularization,
                self.verbose,
            )

        self._model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=self.shuffle,
            verbose=self.verbose,
            class_weight=_set_class_weight(self.class_weight),
        )

    def predict_proba(self, X):
        if scipy.sparse.issparse(X):
            X = X.toarray()
        pos_pred = self._model.predict(X, verbose=self.verbose)
        neg_pred = 1 - pos_pred
        return np.hstack([neg_pred, pos_pred])


def _create_dense_nn_model(vector_size=40,
                           dense_width=128,
                           optimizer='rmsprop',
                           learn_rate_mult=1.0,
                           regularization=0.01,
                           verbose=1):
    """
    Create the NN model

    Parameters
    ----------
    vector_size : int, optional

        Default: 40
    dense_width : int, optional
        Width of the dense layers
        Default: 128
    optimizer : text, optional
        Model optimizer
        Default: 'rmsprop'
    learn_rate_mult : float, optional
        Learning rate multiplier
        Default: 1.0
    regularizations : float, optional
        Regularization rate
        Default: 0.01
    verbose : int, optional
        Whether to print progress information
        Default: 1

    Returns
    -------
    callable:
        A function that return the Keras Sklearn model when
        called.
    """

    model = Sequential()

    model.add(
        Dense(
            dense_width,
            input_dim=vector_size,
            kernel_regularizer=regularizers.l2(regularization),
            activity_regularizer=regularizers.l1(regularization),
            activation='relu',
        ))

    # add Dense layer with relu activation
    model.add(
        Dense(
            dense_width,
            kernel_regularizer=regularizers.l2(regularization),
            activity_regularizer=regularizers.l1(regularization),
            activation='relu',
        ))

    # add Dense layer
    model.add(Dense(1, activation='sigmoid'))

    if optimizer == "sgd":
        optimizer_fn = optimizers.SGD(learning_rate=0.01 * learn_rate_mult)
    elif optimizer == "rmsprop":
        optimizer_fn = optimizers.RMSprop(learning_rate=0.001 * learn_rate_mult)
    elif optimizer == "adagrad":
        optimizer_fn = optimizers.Adagrad(learning_rate=0.01 * learn_rate_mult)
    elif optimizer == "adam":
        optimizer_fn = optimizers.Adam(learning_rate=0.001 * learn_rate_mult)
    elif optimizer == "nadam":
        optimizer_fn = optimizers.Nadam(learning_rate=0.002 * learn_rate_mult)
    else:
        raise NotImplementedError

    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer_fn,
        metrics=['acc'])

    if verbose >= 1:
        model.summary()

    return model
