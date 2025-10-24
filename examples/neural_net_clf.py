__all__ = ["NN1LayerClassifier"]

import os

os.environ["KERAS_BACKEND"] = "torch"

from keras import layers, models, optimizers, regularizers

from asreviewcontrib.dory.classifiers.neural_networks import BaseNNClassifier


class NN1LayerClassifier(BaseNNClassifier):
    """Fully connected neural network (1 hidden layer) classifier (``nn-1-layer``).

    Neural network with one hidden, dense layer of the same size.
    """

    name = "nn-1-layer"
    label = "Fully connected neural network (1 hidden layers)"

    def _build_nn_model(self, X, y):
        input_dim = X.shape[1]

        model = models.Sequential()

        # Add input layer
        model.add(layers.Input(shape=(input_dim,)))

        # Add our hidden layer
        model.add(
            layers.Dense(
                128,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01),
                activation="relu",
            )
        )

        # Add output layer
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
