__all__ = ["Qwen3Embedding"]

from functools import cached_property

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
from tqdm import tqdm
import numpy as np

from asreview.models.feature_extractors import TextMerger

MODEL_SIZES = {"small": "0.6B", "normal": "4B", "large": "8B"}

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="__array__ implementation doesn't accept a copy keyword",
)


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def tokenize_input(tokenizer, texts, eod_id, max_len=8192):
    encoded = tokenizer(texts, padding=False, truncation=True, max_length=max_len - 2)
    for seq, att in zip(encoded["input_ids"], encoded["attention_mask"], strict=True):
        seq.append(eod_id)
        att.append(1)
    return tokenizer.pad(encoded, padding=True, return_tensors="pt")


class Qwen3Embedding:
    """
    Qwen3 Embedding feature extractor.
    """

    name = "qwen3-embedding"
    label = "Qwen3 Embedding"

    # Each query must come with a one-sentence instruction that describes the task
    task = f"Instruct: Embed scientific titles and abstracts to find similar literature for systematic reviews.\nQuery: N/A"  # noqa E501

    def __init__(self, model_size="small", verbose=True):
        if model_size not in MODEL_SIZES:
            raise ValueError(f"model_size must be one of {list(MODEL_SIZES.keys())}")

        self.model_name: str = f"Qwen/Qwen3-Embedding-{MODEL_SIZES[model_size]}"
        self.verbose: bool = verbose
        self.max_len: int = 1024

    @cached_property
    def _model(self) -> Qwen3Model:
        model = AutoModel.from_pretrained(
            self.model_name)
        if self.verbose:
            print(f"Model '{self.model_name}' has been loaded.")
        return model

    @cached_property
    def tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.eod_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if self.verbose:
            print(f"Tokenizer for '{self.model_name}' has been loaded.")
        return tokenizer

    def fit(self, X, y=None):
        return self

    # def transform(self, X, y=None) -> np.ndarray:
    #     X_tokenized = tokenize_input(self.tokenizer, X, self.eod_id, self.max_len)

    #     with torch.no_grad():
    #         if self.verbose:
    #             print(f"Computing embeddings for {len(X)} texts...")
    #         outputs = self._model(**X_tokenized)
    #         embeddings = last_token_pool(
    #             outputs.last_hidden_state, X_tokenized["attention_mask"]
    #         )
    #         return F.normalize(embeddings, p=2, dim=1).cpu().numpy()
    
    def transform(self, X, y=None):
        batch_size = 14
        embeddings = []

        iterator = range(0, len(X), batch_size)
        if self.verbose:
            iterator = tqdm(iterator, desc="Embedding texts", unit="batch")

        for i in iterator:
            batch = X[i:i + batch_size]
            X_tokenized = tokenize_input(self.tokenizer, batch, self.eod_id)
            X_tokenized = X_tokenized.to(self._model.device)
            with torch.no_grad():
                outputs = self._model(**X_tokenized)
                batch_embeddings = last_token_pool(
                    outputs.last_hidden_state, X_tokenized["attention_mask"]
                )
                normalized = F.normalize(batch_embeddings, p=2, dim=1).cpu().numpy()
                embeddings.append(normalized)

        return np.vstack(embeddings)

    def fit_transform(self, X, y=None):
        text_merger = TextMerger(columns=["title", "abstract"], sep=" ")
        X = text_merger.fit_transform(X).tolist()
        return self.fit(X, y).transform(X, y)
