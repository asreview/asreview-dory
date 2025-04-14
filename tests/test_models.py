from itertools import product
from pathlib import Path

import asreview as asr
import pandas as pd
import pytest
from asreview.extensions import extensions
from asreview.extensions import get_extension
from asreview.models.queriers import Max

from asreviewcontrib.nemo.entrypoint import NemoEntryPoint

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
        X=data[["abstract", "title"]], labels=data["included"], cycles=[alc]
    )
    simulate.label([0, 1])
    simulate.review()

    assert isinstance(simulate._results, pd.DataFrame)
    assert simulate._results.shape[0] > 2 and simulate._results.shape[0] <= 6, (
        "Simulation produced incorrect number of results."
    )
    assert classifier.name in simulate._results["classifier"].unique(), (
        "Classifier is not in results."
    )
    assert feature_extractor.name in simulate._results["feature_extractor"].unique(), (
        "Feature extractor is not in results."
    )


def test_get_all_models():
    assert len(NemoEntryPoint()._get_all_models()) == 12


