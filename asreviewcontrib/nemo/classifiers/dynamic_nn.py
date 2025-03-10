import os
from math import ceil, log10

from keras import layers, losses, optimizers, regularizers, wrappers, models

os.environ["KERAS_BACKEND"] = "torch"


class DynamicNNClassifier(wrappers.SKLearnClassifier):
    """
    Dynamic Neural Network Classifier

    Fully connected neural network classifier that dynamically selects
    the number of dense layers based on dataset size.
    """

    name = "dynamic-nn"
    label = "Fully connected neural network (dynamic layer count)"

    def __init__(self, **kwargs):

        def build_nn_model(X, y):
            """
            Function that dynamically builds the NN model based on dataset size.
            """
            input_dim = X.shape[1]
            num_layers = min(3, ceil(log10(max(10, X.shape[0]))))

            model = models.Sequential()
            model.add(layers.Input(shape=(input_dim,)))
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

            model.add(layers.Dense(1, activation="sigmoid"))

            model.compile(
                optimizer=optimizers.Adam(),
                loss=losses.BinaryCrossentropy(),
                metrics=["accuracy"],
            )

            return model

        fit_kwargs = {
            "epochs": kwargs.pop("epochs", 35),
            "verbose": kwargs.pop("verbose", 0),
            "batch_size": kwargs.pop("batch_size", 32),
            "shuffle": kwargs.pop("shuffle", True),
        }

        super().__init__(
            model=build_nn_model, model_kwargs=kwargs, fit_kwargs=fit_kwargs
        )

    def predict_proba(self, X):
        return self.model_.predict(X, verbose=0)
