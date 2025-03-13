import pytest
from itertools import product
from pathlib import Path

import asreview as asr
from asreview.extensions import extensions, get_extension
from asreview.models.queriers import Max
from asreviewcontrib.nemo.entrypoint import NemoEntryPoint
from asreviewcontrib.nemo.utils import min_max_normalize

# Define dataset path
dataset_path = Path("tests/data/generic_labels.csv")

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
def test_asreview_simulation(classifier, feature_extractor):
    # Load dataset
    data = asr.load_dataset(dataset_path)

    # Define Active Learning Cycle
    alc = asr.ActiveLearningCycle(
        classifier=classifier.load()(),
        feature_extractor=feature_extractor.load()(),
        balancer=None,
        querier=Max(),
    )

    # Run simulation
    simulate = asr.Simulate(
        X=data[['abstract', 'title']], labels=data["included"], cycles=[alc]
    )
    simulate.label([0, 1])
    simulate.review()

    # Check results
    assert not simulate._results.empty, "Simulation produced no results."


def test_get_all_models():
    assert len(NemoEntryPoint()._get_all_models()) == 8


def test_min_max_normalize():
    normalized = min_max_normalize([5, 10, 15, 20, 25])
    assert normalized.min() >= 0.0, "Minimum value is below 0"
    assert normalized.max() <= 1.0, "Maximum value is above 1"