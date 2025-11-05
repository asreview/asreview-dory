from asreviewcontrib.dory.feature_extractors.huggingface_embeddings import \
    HFEmbedderPipeline
from asreviewcontrib.dory.feature_extractors.sentence_transformer_embeddings import \
    SentenceTransformerPipeline


class SmallHFEmbedderFE(HFEmbedderPipeline):
    name = "small_hf"
    label = "Google BERT Tiny"
    default_model_name = "google/bert_uncased_L-2_H-128_A-2"


class SmallSentenceTransformerFE(SentenceTransformerPipeline):
    name = "small_st"
    label = "Paraphrase MiniLM"
    default_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"


HF_TEST_CASES = [
    ("hf_l2_norm+mean", {"normalize": "l2", "pooling": "mean"}),
    (
        "hf_standard_norm+quant+cls",
        {"normalize": "standard", "quantize": True, "pooling": "cls"},
    ),
    (
        "hf_no_norm+binary+max",
        {"normalize": None, "quantize": True, "precision": "binary", "pooling": "max"},
    ),
    ("hf_minmax_norm+bs_2", {"normalize": "minmax", "batch_size": 2}),
    ("hf_l2_norm+int8", {"normalize": True, "quantize": True, "precision": "int8"}),
    ("hf_no_norm+uint8", {"normalize": False, "quantize": True, "precision": "uint8"}),
    (
        "hf_standard_norm+ubinary",
        {"normalize": "standard", "quantize": True, "precision": "ubinary"},
    ),
]

HF_BAD_TEST_CASES = [
    ("hf_bad_norm", {"normalize": "invalid-norm"}),
    ("hf_bad_precision", {"quantize": True, "precision": "unsupported"}),
    ("hf_bad_pooling", {"pooling": "average"}),
    ("hf_bad_bs", {"batch_size": -1}),
]

ST_TEST_CASES = [
    ("st_l2_norm", {"normalize": "l2"}),
    ("st_minmax_norm", {"normalize": "minmax"}),
    ("st_standard_norm+quant", {"normalize": "standard", "quantize": True}),
    (
        "st_no_norm+binary",
        {
            "normalize": None,
            "columns": ["title"],
            "quantize": True,
            "precision": "binary",
        },
    ),
    ("st_l2_norm+int8", {"normalize": True, "quantize": True, "precision": "int8"}),
    ("st_no_norm+uint8", {"normalize": False, "quantize": True, "precision": "uint8"}),
]

ST_BAD_TEST_CASES = [
    ("st_bad_norm", {"normalize": "invalid-norm"}),
    ("st_bad_precision", {"quantize": True, "precision": "unsupported"}),
]

DOC2VEC_TEST_CASES = [
    ("doc2vec_60_50", {"vector_size": 60, "epochs": 50}),
    ("doc2vec_100_20", {"vector_size": 100, "epochs": 20}),
    ("doc2vec_80_30", {"vector_size": 80, "epochs": 30}),
]
