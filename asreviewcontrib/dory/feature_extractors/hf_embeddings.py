__all__ = ["XLMRoBERTaLarge"]

import os
from functools import cached_property
from typing import Literal

import numpy as np
import pandas as pd
import torch
from asreview.models.feature_extractors import TextMerger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(max(1, os.cpu_count() - 1))


class Quantizer(BaseEstimator, TransformerMixin):
    def __init__(self, precision="float32"):
        self.precision = precision

    def fit(self, X, y=None):
        # No fitting necessary for quantization
        return self

    def transform(self, X):
        if self.precision == "float32":
            return X.astype(np.float32)
        elif self.precision == "int8":
            # Scale embeddings to int8 range (-128 to 127)
            X_scaled = X / np.max(np.abs(X), axis=1, keepdims=True)
            return (X_scaled * 127).astype(np.int8)
        elif self.precision == "uint8":
            # Scale embeddings to uint8 range (0 to 255)
            X_min = X.min(axis=1, keepdims=True)
            X_max = X.max(axis=1, keepdims=True)
            X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
            return (X_scaled * 255).astype(np.uint8)
        elif self.precision == "binary":
            # Binarize to {-1, +1}
            return np.where(X >= 0, 1, -1).astype(np.int8)
        elif self.precision == "ubinary":
            # Binarize to {0, 1}
            return (X >= 0).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")


class HFEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name, pooling="mean", batch_size=32, verbose=True):
        allowed_poolings = {"cls", "mean", "max"}
        if pooling not in allowed_poolings:
            raise ValueError(
                f"Unsupported pooling method: '{pooling}'. Choose: {allowed_poolings}"
            )
        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @cached_property
    def _tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    @cached_property
    def _model(self):
        model = AutoModel.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()
        if self.verbose:
            print(f"Loaded '{self.model_name}' on {self.device}.")
        return model

    @staticmethod
    def _clean_text_inputs(X):
        if isinstance(X, pd.Series):
            X = X.fillna("").astype(str).tolist()
        elif isinstance(X, list):
            X = ["" if x is None else str(x) for x in X]
        elif isinstance(X, np.ndarray):
            X = ["" if x is None else str(x) for x in X.tolist()]
        else:
            raise ValueError("Expected a list or ndarray of strings or pandas Series.")
        return X

    def fit(self, X, y=None):
        return self

    def _pool(self, output, attention_mask):
        if self.pooling == "cls":
            return output.last_hidden_state[:, 0]
        elif self.pooling == "mean":
            mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(output.last_hidden_state.size())
                .float()
            )
            return (output.last_hidden_state * mask_expanded).sum(
                1
            ) / mask_expanded.sum(1)
        elif self.pooling == "max":
            mask_expanded = attention_mask.unsqueeze(-1).expand(
                output.last_hidden_state.size()
            )
            masked_output = output.last_hidden_state.masked_fill(
                ~mask_expanded.bool(), -1e9
            )
            return masked_output.max(dim=1).values
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

    def transform(self, X, y=None):
        X = self._clean_text_inputs(X)

        if self.verbose:
            print("Embedding using HuggingFace model...")
        embeddings = []

        with torch.no_grad():
            for batch_start in range(0, len(X), self.batch_size):
                batch = X[batch_start : batch_start + self.batch_size]
                encoded = self._tokenizer(
                    batch, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)

                output = self._model(**encoded)
                pooled = self._pool(output, encoded["attention_mask"])
                embeddings.append(pooled.cpu())

        return np.vstack(embeddings)


class HFEmbedderPipeline(Pipeline):
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
    pooling : {"mean", "max", "cls"}, default="mean"
        Pooling strategy to convert token embeddings into a
        fixed-size sentence embedding.
        - "mean": Mean pooling over the token embeddings, weighted by attention mask.
        - "max": Max pooling over token embeddings, ignoring padded tokens.
        - "cls": Use the embedding of the [CLS] token (first token).
    batch_size : int, default=32
        Number of samples processed in each batch during embedding.
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
        pooling: Literal["mean", "max", "cls"] = "mean",
        batch_size: int = 32,
        normalize: bool | Literal["l2", "minmax", "standard", None] = "l2",
        quantize: bool = False,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        verbose=True,
    ):
        self.columns = ["title", "abstract"] if columns is None else columns
        self.sep = sep
        self.model_name = model_name or self.default_model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.normalize = normalize
        self.quantize = quantize
        self.precision = precision
        self.verbose = verbose

        steps = [
            ("text_merger", TextMerger(columns=self.columns, sep=self.sep)),
            (
                "hf_embedder",
                HFEmbedder(
                    model_name=self.model_name,
                    pooling=self.pooling,
                    batch_size=self.batch_size,
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


class XLMRoBERTaLarge(HFEmbedderPipeline):
    name = "xlm-roberta-large"
    label = "XLM-RoBERTa-Large Transformer"
    default_model_name = "FacebookAI/xlm-roberta-large"
