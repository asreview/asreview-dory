__all__ = ["DynamicNNClassifier", "NN2LayerClassifier", "WarmStartNNClassifier"]

import os
from math import ceil, log10

os.environ["KERAS_BACKEND"] = "torch"

from keras import layers, losses, models, optimizers, regularizers, wrappers


class BaseNNClassifier(wrappers.SKLearnClassifier):
    """
    Base Neural Network Classifier.

    Subclasses must implement the _build_nn_model method.
    """

    def __init__(self, **kwargs):
        fit_kwargs = {
            "epochs": kwargs.pop("epochs", 35),
            "verbose": kwargs.pop("verbose", 0),
            "batch_size": kwargs.pop("batch_size", 32),
            "shuffle": kwargs.pop("shuffle", True),
        }

        super().__init__(
            model=self._build_nn_model, model_kwargs=kwargs, fit_kwargs=fit_kwargs
        )

    def _build_nn_model(self, X, y):
        raise NotImplementedError(
            "Subclasses should implement the _build_nn_model method."
        )

    def predict_proba(self, X):
        return self.model_.predict(X, verbose=0)


class DynamicNNClassifier(BaseNNClassifier):
    """
    Dynamic Neural Network Classifier

    Fully connected neural network classifier that dynamically selects
    the number of dense layers based on dataset size.
    """

    name = "dynamic-nn"
    label = "Fully connected neural network (dynamic layer count)"

    def _build_nn_model(self, X, y):
        input_dim = X.shape[1]
        num_layers = min(4, ceil(log10(max(10, X.shape[0]))))

        model = models.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        model.add(layers.BatchNormalization())
        for _ in range(num_layers):
            model.add(
                layers.Dense(
                    64,
                    activation="relu",
                    kernel_regularizer=regularizers.L2(),
                    kernel_initializer="he_normal",
                )
            )
            model.add(layers.Dropout(0.5))
        model.add(
            layers.Dense(
                y.shape[1] if len(y.shape) > 1 else 1,
                activation="sigmoid",
            )
        )

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

        return model


class NN2LayerClassifier(BaseNNClassifier):
    """Fully connected neural network (2 hidden layers) classifier (``nn-2-layer``).

    Neural network with two hidden, dense layers of the same size.
    """

    name = "nn-2-layer"
    label = "Fully connected neural network (2 hidden layers)"

    def _build_nn_model(self, X, y):
        input_dim = X.shape[1]

        model = models.Sequential()

        model.add(layers.Input(shape=(input_dim,)))

        model.add(
            layers.Dense(
                128,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01),
                activation="relu",
            )
        )

        model.add(
            layers.Dense(
                128,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01),
                activation="relu",
            )
        )

        model.add(
            layers.Dense(y.shape[1] if len(y.shape) > 1 else 1, activation="sigmoid")
        )

        # Compile model
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.RMSprop(learning_rate=0.001),
            metrics=["acc"],
        )

        return model


class WarmStartNNClassifier(BaseNNClassifier):
    """
    Neural network with warm-starting behavior.

    Retains previous weights and uses them as initial state on subsequent fits.
    """

    name = "warmstart-nn"
    label = "Neural network (warm start, 2 hidden layers)"

    _last_weights = None

    def _build_nn_model(self, X, y):
        input_dim = X.shape[1]

        model = models.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        model.add(
            layers.Dense(
                128,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01),
                activation="relu",
            )
        )
        model.add(
            layers.Dense(
                128,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01),
                activation="relu",
            )
        )
        model.add(
            layers.Dense(
                y.shape[1] if len(y.shape) > 1 else 1,
                activation="sigmoid",
            )
        )

        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.RMSprop(learning_rate=0.001),
            metrics=["acc"],
        )

        if WarmStartNNClassifier._last_weights is not None:
            try:
                model.set_weights(WarmStartNNClassifier._last_weights)
            except ValueError:
                # Handle cases where the model architecture has changed
                print(
                    "Warning: Previous weights do not match current model architecture."
                )
                WarmStartNNClassifier._last_weights = None
        return model

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        WarmStartNNClassifier._last_weights = self.model_.get_weights()
        return self
