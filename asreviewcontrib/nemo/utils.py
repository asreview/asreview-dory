__all__ = ["min_max_normalize"]

import numpy as np


def min_max_normalize(embedding):
    """
    Normalize an embedding vector using min-max normalization.

    This function scales the input embedding vector so that its values range
    between 0 and 1. The normalization is done by subtracting the minimum value
    of the embedding and dividing by the range (max - min).

    Parameters
    ----------
    embedding : np.ndarray
        The input embedding vector to be normalized.

    Returns
    -------
    np.ndarray
        The normalized embedding vector with values scaled between 0 and 1.
    """
    min_val = np.min(embedding)
    max_val = np.max(embedding)
    normalized_embedding = (embedding - min_val) / (max_val - min_val)
    return normalized_embedding
