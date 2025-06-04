from pathlib import Path

import asreview as asr
import pandas as pd
from asreview.extensions import get_extension
from asreview.models.balancers import Balanced
from asreview.models.queriers import Max

# Define dataset path
dataset_path = Path("tests/data/generic_labels.csv")


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
