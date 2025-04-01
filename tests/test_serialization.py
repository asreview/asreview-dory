from itertools import product

import asreview as asr
import pytest
from asreview.extensions import extensions
from asreview.extensions import get_extension
from asreview.models.queriers import Max


# Get all classifiers and feature extractors from ASReview, filtering contrib models
classifiers = [
    cls for cls in extensions("models.classifiers") if "asreviewcontrib" in str(cls)
] + [get_extension("models.classifiers", "svm")]
feature_extractors = [
    fe for fe in extensions("models.feature_extractors") if "asreviewcontrib" in str(fe)
]

# Generate all combinations of classifier and feature extractor
pairs = list(product(classifiers, feature_extractors))

test_ids = [
    f"{classifier.name}__{feature_extractor.name}"
    for classifier, feature_extractor in pairs
]


@pytest.mark.parametrize("classifier, feature_extractor", pairs, ids=test_ids)
def test_alc_to_and_from_meta(classifier, feature_extractor):
    # Define Active Learning Cycle
    alc1 = asr.ActiveLearningCycle(
        classifier=classifier.load()(),
        feature_extractor=feature_extractor.load()(),
        balancer=None,
        querier=Max(),
    )

    alc2_meta = asr.ActiveLearningCycleData(
        querier="max",
        classifier="nb",
        balancer="balanced",
        feature_extractor="tfidf",
        classifier_param={"alpha": 5},
    )

    alc1_meta = alc1.to_meta()

    alc1_from_meta = asr.ActiveLearningCycle.from_meta(alc1_meta)

    alc2_from_meta = asr.ActiveLearningCycle.from_meta(alc2_meta)

    assert (
        alc1.classifier.name
        == alc1_from_meta.classifier.name
        == alc2_from_meta.classifier.name
    ), "Classifier names do not match"
    assert (
        alc1.classifier.get_params()
        == alc1_from_meta.classifier.get_params()
        == alc2_from_meta.classifier.get_params()
    ), "Classifier parameters do not match"

    assert (
        alc1.feature_extractor.name
        == alc1_from_meta.feature_extractor.name
        == alc2_from_meta.feature_extractor.name
    ), "Feature extractor names do not match"
    assert (
        alc1.feature_extractor.get_params()
        == alc1_from_meta.feature_extractor.get_params()
        == alc2_from_meta.feature_extractor.get_params()
    ), "Feature extractor parameters do not match"

    assert (
        alc1.balancer.name
        == alc1_from_meta.balancer.name
        == alc2_from_meta.balancer.name
    ), "Balancer names do not match"
    assert (
        alc1.balancer.get_params()
        == alc1_from_meta.balancer.get_params()
        == alc2_from_meta.balancer.get_params()
    ), "Balancer parameters do not match"

    assert (
        alc1.querier.name == alc1_from_meta.querier.name == alc2_from_meta.querier.name
    ), "Querier names do not match"
    assert (
        alc1.querier.get_params()
        == alc1_from_meta.querier.get_params()
        == alc2_from_meta.querier.get_params()
    ), "Querier parameters do not match"
