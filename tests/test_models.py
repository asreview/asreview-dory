from pathlib import Path

import asreview as asr
import pandas as pd
import pytest
from asreview.extensions import extensions, get_extension
from asreview.models.balancers import Balanced
from asreview.models.queriers import Max

from asreviewcontrib.dory.entrypoint import DoryEntryPoint

# Define dataset path
dataset_path = Path("tests/data/generic_labels.csv")

# Get all classifiers and feature extractors from ASReview, filtering contrib models
classifiers = [
    cls for cls in extensions("models.classifiers") if "asreviewcontrib" in str(cls)
] + [get_extension("models.classifiers", "svm")]
feature_extractors = [
    fe for fe in extensions("models.feature_extractors") if "asreviewcontrib" in str(fe)
]

test_ids = [
    f"{feature_extractor.name}__per_classifier"
    for feature_extractor in feature_extractors
]


@pytest.mark.parametrize("feature_extractor", feature_extractors, ids=test_ids)
def test_all_fe_clf_combinations(feature_extractor):
    # Load dataset
    data = asr.load_dataset(dataset_path)

    # Preprocess data using feature extractor
    fm = feature_extractor.load()().fit_transform(data)

    # Test each classifier on the preprocessed FMs
    for classifier in classifiers:
        # Define Active Learning Cycle
        alc = asr.ActiveLearningCycle(
            classifier=classifier.load()(),
            feature_extractor=feature_extractor.load()(),
            balancer=None,
            querier=Max(),
        )

        # Run simulation
        simulate = asr.Simulate(
            X=fm,
            labels=data["included"],
            cycles=[alc],
            skip_transform=True,
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
        assert (
            feature_extractor.name in simulate._results["feature_extractor"].unique()
        ), "Feature extractor is not in results."


def test_language_agnostic_l2_preset():
    # Load dataset
    data = asr.load_dataset(dataset_path)

    # Define Active Learning Cycle
    alc = asr.ActiveLearningCycle(
        classifier=get_extension("models.classifiers", "svm").load()(
            loss="squared_hinge", C=0.19
        ),
        feature_extractor=get_extension(
            "models.feature_extractors", "multilingual-e5-large"
        ).load()(normalize=True),
        balancer=Balanced(ratio=9.9),
        querier=Max(),
    )
    # Run simulation
    simulate = asr.Simulate(
        X=data,
        labels=data["included"],
        cycles=[alc],
    )
    simulate.label([0, 1])
    simulate.review()
    assert isinstance(simulate._results, pd.DataFrame)
    assert simulate._results.shape[0] > 2 and simulate._results.shape[0] <= 6, (
        "Simulation produced incorrect number of results."
    )
    assert (
        get_extension("models.classifiers", "svm").load()().name
        in simulate._results["classifier"].unique()
    ), "Classifier is not in results."
    assert (
        get_extension("models.feature_extractors", "multilingual-e5-large")
        .load()()
        .name
        in simulate._results["feature_extractor"].unique()
    ), "Feature extractor is not in results."


def test_heavy_h3_preset():
    # Load dataset
    data = asr.load_dataset(dataset_path)

    # Define Active Learning Cycle
    alc = asr.ActiveLearningCycle(
        classifier=get_extension("models.classifiers", "svm").load()(
            loss="squared_hinge", C=0.13
        ),
        feature_extractor=get_extension("models.feature_extractors", "mxbai").load()(
            normalize=True
        ),
        balancer=Balanced(ratio=9.7),
        querier=Max(),
    )
    # Run simulation
    simulate = asr.Simulate(
        X=data,
        labels=data["included"],
        cycles=[alc],
    )
    simulate.label([0, 1])
    simulate.review()
    assert isinstance(simulate._results, pd.DataFrame)
    assert simulate._results.shape[0] > 2 and simulate._results.shape[0] <= 6, (
        "Simulation produced incorrect number of results."
    )
    assert (
        get_extension("models.classifiers", "svm").load()().name
        in simulate._results["classifier"].unique()
    ), "Classifier is not in results."
    assert (
        get_extension("models.feature_extractors", "mxbai").load()().name
        in simulate._results["feature_extractor"].unique()
    ), "Feature extractor is not in results."


def test_get_all_models():
    assert len(DoryEntryPoint()._get_all_models()) == 10
