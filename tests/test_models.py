import pytest
import pandas as pd
from itertools import product
from pathlib import Path

import asreview
from asreview.models.balancers import Balanced
from asreview.extensions import extensions
from asreview.models.queriers import Max

# Define dataset path
dataset_path = Path("tests/data/generic_labels.csv")

# Get all classifiers and feature extractors from ASReview, filtering contrib models
classifiers = [
    cls for cls in extensions("models.classifiers") if "asreviewcontrib" in str(cls)
]
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
    data = pd.read_csv(dataset_path)
    labels = data["label_included"]

    # Define Active Learning Cycle
    alc = asreview.ActiveLearningCycle(
        classifier=classifier.load(),
        feature_extractor=feature_extractor.load(),
        balancer=Balanced(),
        querier=Max(),
    )

    # Run simulation
    simulate = asreview.Simulate(X=data, labels=labels, cycles=[alc])
    simulate.label([0, 1])
    simulate.review()

    # Check results
    assert not simulate._results.empty, "Simulation produced no results."
   