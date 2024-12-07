import shutil
import tempfile
from pathlib import Path

import pytest
from asreview import ASReviewData
from asreview import ASReviewProject
from asreview.models.balance import DoubleBalance
from asreviewcontrib.nemo.entrypoint import _get_all_models
from asreview.extensions import load_extension
from asreview.simulation.simulate import Simulate
from asreview import load_dataset
from asreview.data import DataStore

dataset_path = Path("tests/data/generic_labels.csv")

classifiers = _get_all_models("cl")
feature_extractors = _get_all_models("fe")


@pytest.mark.parametrize("feature_extractor", feature_extractors)
def test_rf_with_all_feature_extractors(feature_extractor):
    _run_simulation_test(
        load_extension("models.classifiers", "rf")(),
        load_extension("models.feature_extraction", feature_extractor)(),
    )


@pytest.mark.parametrize("classifier", classifiers)
def test_all_classifiers_with_onehot(classifier):
    _run_simulation_test(
        load_extension("models.classifiers", classifier)(),
        load_extension("models.feature_extraction", "onehot")(),
    )


def _run_simulation_test(classifier_model, feature_model):
    query_model = load_extension("models.query", "max")()
    balance_model = load_extension("models.balance", "double")()

    records = load_dataset(dataset_path)
    data_store = DataStore(":memory:")
    data_store.create_tables()
    data_store.add_records(records)

    fm = feature_model.fit_transform(data_store)

    sim = Simulate(
        fm,
        data_store["included"],
        classifier=classifier_model,
        query_strategy=query_model,
        balance_strategy=balance_model,
        feature_extraction=feature_model,
    )

    sim.label_random(
        n_included=1,
        n_excluded=1,
        prior=True,
    )
    sim.review()
