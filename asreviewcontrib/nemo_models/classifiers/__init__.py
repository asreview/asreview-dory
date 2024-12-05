__all__ = [
    "XGBoost",
    "NN2LayerClassifier",
    "AdaBoost",
    "DynamicNNClassifier"
]

def __getattr__(name):
    """
    Dynamically load the requested classifier module when accessed.

    Parameters
    ----------
    name : str
        The name of the classifier class to load.

    Returns
    -------
    object
        The requested classifier class.
    """
    if name == "XGBoost":
        from .xgboost import XGBoost
        return XGBoost
    elif name == "AdaBoost":
        from .adaboost import AdaBoost
        return AdaBoost
    elif name == "DynamicNNClassifier":
        from .dynamic_nn import DynamicNNClassifier
        return DynamicNNClassifier
    elif name == "NN2LayerClassifier":
        from .nn_2_layer import NN2LayerClassifier
        return NN2LayerClassifier
    raise AttributeError(f"Module {__name__} has no attribute {name}.")
