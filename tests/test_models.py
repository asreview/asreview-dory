from pathlib import Path

import pytest
from asreviewcontrib.nemo.entrypoint import _get_all_models
from asreview.extensions import load_extension
from asreview.simulation.simulate import Simulate
from asreview import load_dataset
from asreview.data import DataStore

dataset_path = Path("tests/data/generic_labels.csv")

classifiers = _get_all_models("cls")
feature_extractors = _get_all_models("fe")

@pytest.mark.parametrize(
    "feature_extractor", 
    feature_extractors, 
    ids=[f"FeatureExtractor-{fe.name}" for fe in feature_extractors]
)
def test_feature_extractors(feature_extractor):
    _run_simulation_test(
        load_extension("models.classifiers", "logistic")(),
        feature_extractor.load()(),
    )


@pytest.mark.parametrize(
    "classifier", 
    classifiers, 
    ids=[f"Classifier-{clf.name}" for clf in classifiers]
)
def test_classifiers(classifier):
    _run_simulation_test(
        classifier.load()(),
        load_extension("models.feature_extraction", "tfidf")(),
    )


def _run_simulation_test(classifier_model, feature_model):
    print("Loaded Feature Extractor and Classifier succesfully")
    query_model = load_extension("models.query", "max")()
    balance_model = load_extension("models.balance", "double")()

    records = load_dataset(dataset_path, dataset_id=dataset_path.name)
    data_store = DataStore(":memory:")
    data_store.create_tables()
    data_store.add_records(records)

    fm = feature_model.from_data_store(data=data_store)

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
