__all__ = [
    "LaBSE",
    "MXBAI",
    "SBERT",
    "MultilingualE5Large",
    "GTR",
]
import os
from functools import cached_property
from typing import Literal

import numpy as np
import torch
from asreview.models.feature_extractors import TextMerger
from sentence_transformers import SentenceTransformer, quantize_embeddings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

torch.set_num_threads(max(1, os.cpu_count() - 1))


class Quantizer(BaseEstimator, TransformerMixin):
    def __init__(self, precision="float32"):
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return quantize_embeddings(X, precision=self.precision)


class SentenceTransformerPipeline(Pipeline):
    """
    A configurable pipeline for generating sentence embeddings using a transformer-based
    model.

    This pipeline includes text merging and embedding steps. It supports normalization,
    quantization, and configurable precision settings. Primarily designed for textual
    data from columns such as titles and abstracts.

    Parameters
    ----------
    columns : list of str or None, default=None
        List of column names to extract and merge text from.
        Defaults to ["title", "abstract"] if None is provided.
    sep : str, default=" "
        Separator used when joining text from multiple columns.
    model_name : str or None, default=None
        Identifier or path for the embedding model to use.
        If None, uses `default_model_name`.
    normalize : bool or {"l2", "minmax", "standard", None}, default="l2"
        Normalization strategy:
        - None or False: No normalization applied.
        - "l2" or True: Unit vector normalization.
        - "minmax": Scales features to [0, 1] using MinMaxScaler.
        - "standard": Standardizes features using StandardScaler
        (zero mean, unit variance).
    quantize : bool, default=False
        If True, applies quantization to reduce model/vector size.
    precision : {"float32", "int8", "uint8", "binary", "ubinary"}, default="float32"
        Precision format used for quantized embeddings.
    verbose : bool, default=True
        If True, logs progress or debug output during the pipeline execution.
    """

    default_model_name = None
    name = None
    label = None

    def __init__(
        self,
        columns: list[str] | None = None,
        sep: str = " ",
        model_name: str | None = None,
        normalize: bool | Literal["l2", "minmax", "standard", None] = "l2",
        quantize: bool = False,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        verbose=True,
    ):
        self.columns = ["title", "abstract"] if columns is None else columns
        self.sep = sep
        self.model_name = model_name or self.default_model_name
        self.normalize = normalize
        self.quantize = quantize
        self.precision = precision
        self.verbose = verbose

        steps = [
            ("text_merger", TextMerger(columns=self.columns, sep=self.sep)),
            (
                "sentence_transformer",
                BaseSentenceTransformer(
                    model_name=self.model_name,
                    verbose=self.verbose,
                ),
            ),
        ]

        if self.normalize == "l2" or self.normalize is True:
            steps.append(("normalizer", Normalizer(norm="l2")))
        elif self.normalize == "minmax":
            steps.append(("normalizer", MinMaxScaler()))
        elif self.normalize == "standard":
            steps.append(("normalizer", StandardScaler()))
        elif self.normalize not in (None, False):
            raise ValueError(f"Unsupported normalization method: '{self.normalize}'")

        if self.quantize:
            steps.append(("quantizer", Quantizer(self.precision)))

        super().__init__(steps)


class BaseSentenceTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for sentence transformer feature extractors.
    """

    def __init__(
        self,
        model_name,
        verbose,
    ):
        self.model_name = model_name
        self.verbose = verbose

    @cached_property
    def _model(self):
        model = SentenceTransformer(self.model_name)
        if self.verbose:
            print(f"Model '{self.model_name}' has been loaded.")
        return model

    def fit(self, X, y=None):
        # Required func for last Pipeline step, but not required
        # for sentence-transformers, so return self
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print("Embedding text...")

        embeddings = self._model.encode(X, show_progress_bar=self.verbose)
        embeddings = self._to_numpy(embeddings)

        return embeddings

    def _to_numpy(self, arr):
        """Ensure input is a NumPy array."""
        if hasattr(arr, "numpy"):
            return arr.numpy()
        return np.array(arr)


class LaBSE(SentenceTransformerPipeline):
    name = "labse"
    label = "LaBSE Transformer"
    default_model_name = "sentence-transformers/LaBSE"


class MXBAI(SentenceTransformerPipeline):
    name = "mxbai"
    label = "mxbai Sentence BERT"
    default_model_name = "mixedbread-ai/mxbai-embed-large-v1"


class SBERT(SentenceTransformerPipeline):
    name = "sbert"
    label = "mpnet Sentence BERT"
    default_model_name = "all-mpnet-base-v2"


class MultilingualE5Large(SentenceTransformerPipeline):
    name = "multilingual-e5-large"
    label = "Multilingual E5 Large"
    default_model_name = "intfloat/multilingual-e5-large"


class GTR(SentenceTransformerPipeline):
    name = "gtr-t5-large"
    label = "Google GTR"
    default_model_name = "gtr-t5-large"
