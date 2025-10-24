__all__ = ["Qwen3Embedding8B"]
from asreviewcontrib.dory.feature_extractors.huggingface_embeddings import (
    HFEmbedderPipeline,
)


class Qwen3Embedding8B(HFEmbedderPipeline):
    name = "qwen3-embedding-8b"
    label = "Qwen3 Embedding 8B"
    default_model_name = "Qwen/Qwen3-Embedding-8B"
