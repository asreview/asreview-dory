
import shutil
from itertools import product
from pathlib import Path

import pytest
from asreview import ASReviewData
from asreview import ASReviewProject
from asreview.models.balance import DoubleBalance
from asreview.models.classifiers import list_classifiers
from asreview.models.feature_extraction import list_feature_extraction
from asreview.models.query import MaxQuery
from asreview.review import ReviewSimulate

dataset_path = Path("tests/data/generic_labels.csv")

classifiers = [cls for cls in list_classifiers() if 'asreviewcontrib' in str(cls)]
feature_extractors = [fe for fe in list_feature_extraction() if 'asreviewcontrib' in str(fe)] # noqa: E501

pairs = list(product(classifiers,feature_extractors))

@pytest.mark.parametrize("classifier,feature_extractor", pairs)
def test_asreview_simulation(classifier, feature_extractor):

    project_path = Path(f"api_simulation_{classifier.__name__}_{feature_extractor.__name__}")  # noqa: E501
    
    project = ASReviewProject.create(
        project_path=project_path,
        project_id="api_example",
        project_mode="simulate",
        project_name="api_example",
    )

    project_data_path = project_path / 'data' / dataset_path.name

    shutil.copy(dataset_path, project_data_path)
    project.add_dataset(dataset_path.name)

    model = init_classifier(classifier)

    reviewer = ReviewSimulate(
        as_data=ASReviewData.from_file(project_data_path),
        model= model,
        query_model=MaxQuery(),
        balance_model=DoubleBalance(),
        feature_model=feature_extractor(),
        init_seed=535,
        n_instances=1,
        project=project,
        n_prior_included=1,
        n_prior_excluded=1,
    )
    
    project.update_review(status="review")
    reviewer.review()

    shutil.rmtree(project_path)

def init_classifier(model_class):
    try:
        return model_class(random_state=921)
    except TypeError:
        return model_class()