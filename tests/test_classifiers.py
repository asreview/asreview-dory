from pathlib import Path

import asreview as asr
import pandas as pd
import pytest
from asreview.extensions import extensions
from asreview.models.queriers import Max
from dummy_feature_extractors import (SmallHFEmbedderFE,
                                      SmallSentenceTransformerFE)

# Define dataset path
dataset_path = Path("tests/data/generic_labels.csv")

classifier_parameters = {
    "xgboost": {"max_depth": 5, "n_estimators": 250},
    "dynamic-nn": {"epochs": 30, "batch_size": 16},
    "nn-2-layer": {"epochs": 50, "verbose": 1},
    "warmstart-nn": {"epochs": 45, "shuffle": False},
    "adaboost": {"n_estimators": 30, "learning_rate": 0.5},
}

# Get all classifiers and feature extractors from ASReview, filtering contrib models
classifiers = [
    cls for cls in extensions("models.classifiers") if "asreviewcontrib" in str(cls)
]

feature_extractors = [SmallHFEmbedderFE, SmallSentenceTransformerFE]


# Parametrize: combine each classifier with each FE
@pytest.mark.parametrize(
    "clf_entry,fe_cls",
    [(clf, fe_cls) for clf in classifiers for fe_cls in feature_extractors],
    ids=lambda val: getattr(val, "name", val) if hasattr(val, "name") else val,
)
def test_all_classifiers_with_extractors(clf_entry, fe_cls):
    data = asr.load_dataset(dataset_path)

    # Feature extraction
    fe = fe_cls()
    fm = fe.fit_transform(data)

    clf_name = clf_entry.name
    clf_kwargs = classifier_parameters.get(clf_name, {})

    alc = asr.ActiveLearningCycle(
        classifier=clf_entry.load()(**clf_kwargs),
        feature_extractor=fe,
        balancer=None,
        querier=Max(),
    )

    sim = asr.Simulate(
        X=fm,
        labels=data["included"],
        cycles=[alc],
        skip_transform=True,
    )
    sim.label([0, 1])
    sim.review()

    assert isinstance(sim._results, pd.DataFrame)
    assert 2 < sim._results.shape[0] <= 6, "Unexpected result row count."
    assert clf_name in sim._results["classifier"].unique()
    assert fe.name in sim._results["feature_extractor"].unique()
