from pathlib import Path

import asreview as asr
import numpy as np
import pytest
from dummy_feature_extractors import (DOC2VEC_TEST_CASES, HF_BAD_TEST_CASES,
                                      HF_TEST_CASES, ST_BAD_TEST_CASES,
                                      ST_TEST_CASES, SmallHFEmbedderFE,
                                      SmallSentenceTransformerFE)

from asreviewcontrib.dory.feature_extractors.doc2vec import Doc2Vec

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
    data = asr.load_dataset(dataset_path)
    features = fe_cls(**params).fit_transform(data)

    assert features is not None, "Feature matrix is None"
    assert hasattr(features, "shape"), "Feature matrix must have a shape"
    assert features.shape[0] == len(data), "One embedding per record"
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
        fe_cls(**params).fit_transform(asr.load_dataset(dataset_path))
