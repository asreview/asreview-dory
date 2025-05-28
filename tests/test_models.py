from pathlib import Path

import asreview as asr
import pandas as pd
import pytest
from asreview.extensions import extensions, get_extension
from asreview.models.balancers import Balanced
from asreview.models.queriers import Max

from asreviewcontrib.dory.entrypoint import DoryEntryPoint

classifier_parameters = {
    "xgboost": {"max_depth": 5, "n_estimators": 250},
    "dynamic-nn": {"epochs": 30, "batch_size": 16},
    "nn-2-layer": {"epochs": 50, "verbose": 1},
    "warmstart-nn": {"epochs": 45, "shuffle": False},
    "adaboost": {"n_estimators": 30, "learning_rate": 0.5},
}

feature_extractor_parameters = {
    "labse": {"normalize": "l2", "quantize": False},
    "mxbai": {
        "normalize": "minmax",
        "precision": "binary",
        "quantize": True,
    },
    "sbert": {
        "normalize": "standard",
        "verbose": False,
        "quantize": False,
    },
    "multilingual-e5-large": {"normalize": False, "sep": ",", "quantize": True},
    "gtr-t5-large": {"normalize": True, "columns": ["title"], "quantize": False},
}

# Define dataset path
dataset_path = Path("tests/data/generic_labels.csv")

# Get all classifiers and feature extractors from ASReview, filtering contrib models
classifiers = [
    cls for cls in extensions("models.classifiers") if "asreviewcontrib" in str(cls)
]
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
    fm = feature_extractor.load()(
        **feature_extractor_parameters.get(feature_extractor.name)
    ).fit_transform(data)

    # Test each classifier on the preprocessed FMs
    for classifier in classifiers:
        # Define Active Learning Cycle
        alc = asr.ActiveLearningCycle(
            classifier=classifier.load()(**classifier_parameters.get(classifier.name)),
            feature_extractor=feature_extractor.load()(
                **feature_extractor_parameters.get(feature_extractor.name)
            ),
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
            loss="squared_hinge", C=0.106, max_iter=5000
        ),
        feature_extractor=get_extension(
            "models.feature_extractors", "multilingual-e5-large"
        ).load()(normalize=True),
        balancer=Balanced(ratio=9.707),
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
            loss="squared_hinge", C=0.067, max_iter=5000
        ),
        feature_extractor=get_extension("models.feature_extractors", "mxbai").load()(
            normalize=True
        ),
        balancer=Balanced(ratio=9.724),
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

def test_invalid_normalization_method():
    fe = get_extension("models.feature_extractors", "multilingual-e5-large").load()

    with pytest.raises(ValueError, match="Unsupported normalization method"):
        fe(normalize="invalid-method")
