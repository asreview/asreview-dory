__all__ = [
    "SBERT",
    "Doc2Vec",
    "MXBAI",
    "LaBSE",
]


def __getattr__(name):
    """
    Dynamically load the requested feature extractor module when accessed.

    Parameters
    ----------
    name : str
        The name of the feature extractor class to load.

    Returns
    -------
    object
        The requested feature extractor class.
    """
    if name == "SBERT":
        from .sbert import SBERT
        return SBERT
    elif name == "Doc2Vec":
        from .doc2vec import Doc2Vec
        return Doc2Vec
    elif name == "MXBAI":
        from .mxbai import MXBAI
        return MXBAI
    elif name == "LaBSE":
        from .labse import LaBSE
        return LaBSE
    raise AttributeError(f"Module {__name__} has no attribute {name}.")
