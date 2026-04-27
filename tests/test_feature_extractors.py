from pathlib import Path

import asreview as asr
import numpy as np
import pandas as pd
import pytest
from dummy_feature_extractors import (DOC2VEC_TEST_CASES, HF_BAD_TEST_CASES,
                                      HF_TEST_CASES, ST_BAD_TEST_CASES,
                                      ST_TEST_CASES, SmallHFEmbedderFE,
                                      SmallSentenceTransformerFE)

from asreviewcontrib.dory.feature_extractors.doc2vec import Doc2Vec
from asreviewcontrib.dory.feature_extractors.huggingface_embeddings import HFEmbedder
from asreviewcontrib.dory.feature_extractors.sentence_transformer_embeddings import (
    BaseSentenceTransformer,
)
from asreviewcontrib.dory.feature_extractors.utils import clean_text_inputs

# Define dataset path
dataset_path = Path("tests/data/generic_labels.csv")

ALL_FE_VARIANTS = (
    [(SmallHFEmbedderFE, params) for _, params in HF_TEST_CASES]
    + [(SmallSentenceTransformerFE, params) for _, params in ST_TEST_CASES]
    + [(Doc2Vec, params) for _, params in DOC2VEC_TEST_CASES]
)

ALL_FE_VARIANTS_IDS = (
    [test_id for test_id, _ in HF_TEST_CASES]
    + [test_id for test_id, _ in ST_TEST_CASES]
    + [test_id for test_id, _ in DOC2VEC_TEST_CASES]
)


@pytest.mark.parametrize("fe_cls,params", ALL_FE_VARIANTS, ids=ALL_FE_VARIANTS_IDS)
def test_feature_extractor_variants(fe_cls, params):
    db = asr.load_dataset(dataset_path)
    df = db.input.get_df()
    features = fe_cls(**params).fit_transform(df)

    assert features is not None, "Feature matrix is None"
    assert hasattr(features, "shape"), "Feature matrix must have a shape"
    assert features.shape[0] == len(df), "One embedding per record"
    assert features.ndim == 2, "Embeddings must be 2D (samples x features)"
    assert features.dtype.kind in {"f", "i", "u"}, "Expect numeric features"
    assert not np.allclose(features.std(axis=0), 0), "All embeddings are identical"


ALL_FE_BAD_VARIANTS = [
    (SmallHFEmbedderFE, params) for _, params in HF_BAD_TEST_CASES
] + [(SmallSentenceTransformerFE, params) for _, params in ST_BAD_TEST_CASES]

ALL_FE_BAD_VARIANTS_IDS = [test_id for test_id, _ in HF_BAD_TEST_CASES] + [
    test_id for test_id, _ in ST_BAD_TEST_CASES
]


@pytest.mark.parametrize(
    "fe_cls,params", ALL_FE_BAD_VARIANTS, ids=ALL_FE_BAD_VARIANTS_IDS
)
def test_feature_extractor_bad_variants(fe_cls, params):
    with pytest.raises(ValueError):
        db = asr.load_dataset(dataset_path)
        fe_cls(**params).fit_transform(db.input.get_df())


# --- clean_text_inputs unit tests ---

MIXED_INPUTS = ["hello", None, 42, 3.14, True, ""]
EXPECTED_CLEANED = ["hello", "", "42", "3.14", "True", ""]


def test_clean_text_inputs_list():
    result = clean_text_inputs(MIXED_INPUTS)
    assert result == EXPECTED_CLEANED


def test_clean_text_inputs_series():
    s = pd.Series(MIXED_INPUTS)
    result = clean_text_inputs(s)
    assert result == EXPECTED_CLEANED


def test_clean_text_inputs_ndarray():
    arr = np.array(MIXED_INPUTS, dtype=object)
    result = clean_text_inputs(arr)
    assert result == EXPECTED_CLEANED


def test_clean_text_inputs_series_with_nan():
    s = pd.Series([np.nan, "text", None])
    result = clean_text_inputs(s)
    assert result == ["", "text", ""]


def test_clean_text_inputs_invalid_type():
    with pytest.raises(ValueError):
        clean_text_inputs("not a list or array")


# --- Integration: base embedder transform with non-string inputs ---

SMALL_ST_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
SMALL_HF_MODEL = "google/bert_uncased_L-2_H-128_A-2"

NON_STRING_INPUTS = [
    ["hello world", None, 42],
    pd.Series(["hello world", None, 42]),
    np.array(["hello world", None, 42], dtype=object),
]
NON_STRING_INPUT_IDS = ["list", "series", "ndarray"]


@pytest.mark.parametrize("X", NON_STRING_INPUTS, ids=NON_STRING_INPUT_IDS)
def test_sentence_transformer_transform_non_strings(X):
    model = BaseSentenceTransformer(model_name=SMALL_ST_MODEL, verbose=False)
    embeddings = model.transform(X)
    assert embeddings.shape == (3, embeddings.shape[1])
    assert embeddings.dtype.kind == "f"


@pytest.mark.parametrize("X", NON_STRING_INPUTS, ids=NON_STRING_INPUT_IDS)
def test_hf_embedder_transform_non_strings(X):
    model = HFEmbedder(model_name=SMALL_HF_MODEL, verbose=False)
    embeddings = model.transform(X)
    assert embeddings.shape == (3, embeddings.shape[1])
    assert embeddings.dtype.kind == "f"
