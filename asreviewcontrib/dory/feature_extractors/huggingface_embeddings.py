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

from .utils import Quantizer

torch.set_num_threads(max(1, os.cpu_count() - 1))


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
    device : int, str, torch.device, or None, default=None
        Device to run the model on. If None, automatically selects GPU if available,
        otherwise CPU. Can specify device index (int) -> "cuda:{INDEX}",
        device name (str), or torch.device.
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
        device: int | str | torch.device | None = None,
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
        self.device = device
        self.verbose = verbose

        steps = [
            ("text_merger", TextMerger(columns=self.columns, sep=self.sep)),
            (
                "hf_embedder",
                HFEmbedder(
                    model_name=self.model_name,
                    pooling=self.pooling,
                    batch_size=self.batch_size,
                    device=self.device,
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


class HFEmbedder(BaseEstimator, TransformerMixin):
    """
    Base class for HuggingFace feature extractors.
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        batch_size: int = 32,
        device: int | str | torch.device | None = None,
        verbose: bool = True,
    ):
        allowed_poolings = {"cls", "mean", "max"}
        if pooling not in allowed_poolings:
            raise ValueError(
                f"Unsupported pooling method: '{pooling}'. Choose: {allowed_poolings}"
            )
        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.verbose = verbose

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                # Use GPU if available
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                # Use MPS on macOS if available
                self.device = torch.device("mps")
            else:
                # Fallback to CPU
                self.device = torch.device("cpu")
        elif isinstance(device, int):
            self.device = torch.device(
                f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

    @cached_property
    def _tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    @cached_property
    def _model(self):
        model = AutoModel.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()

        device = self._get_model_device(model)
        if device is not None:
            print(f"Loaded '{self.model_name}' on {device}.")
        else:
            print(
                f"Loaded '{self.model_name}', but could not detect device.",
            )
        return model

    def _get_model_device(self, model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            try:
                return next(model.buffers()).device
            except StopIteration:
                return None

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


class XLMRoBERTaLarge(HFEmbedderPipeline):
    name = "xlm-roberta-large"
    label = "XLM-RoBERTa-Large Transformer"
    default_model_name = "FacebookAI/xlm-roberta-large"
