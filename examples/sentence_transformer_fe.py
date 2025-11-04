__all__ = ["GemmaMedical"]
from asreviewcontrib.dory.feature_extractors.sentence_transformer_embeddings import (
    SentenceTransformerPipeline,
)


class GemmaMedical(SentenceTransformerPipeline):
    name = "gemma_300m_medical"
    label = "Gemma 300M Medical"
    default_model_name = "sentence-transformers/embeddinggemma-300m-medical"
