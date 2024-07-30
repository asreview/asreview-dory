from pathlib import Path
from asreview import ASReviewData, ASReviewProject
from asreview.review import ReviewSimulate
from asreview.models.query import MaxQuery
from asreview.models.balance import DoubleBalance
from asreview.models.feature_extraction import list_feature_extraction
from asreview.models.classifiers import list_classifiers
from synergy_dataset import Dataset
import matplotlib.pyplot as plt
import pytest
from itertools import product

from asreview import open_state
from asreviewcontrib.insights.plot import plot_recall

import shutil

# asreview simulate data\Donners_2021_ids.csv -s output.asreview --model dynamic-nn --query_strategy max --feature_extraction doc2vec --init_seed 535 --seed 165 --n_prior_included 5 --n_prior_excluded 10
#d = Dataset('Donners_2021')
#dataset_name = d.name +'.csv'
#d = d.to_frame()

dataset_path = Path("tests/data/generic_labels.csv")

classifiers = [cls for cls in list_classifiers() if 'asreviewcontrib' in str(cls)]
feature_extractors = [fe for fe in list_feature_extraction() if 'asreviewcontrib' in str(fe)]

pairs = list(product(classifiers,feature_extractors))


#file_list = []
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
    #d.to_csv(project_data_path)
    shutil.copy(dataset_path, project_data_path)
    project.add_dataset(dataset_path.name)

    reviewer = ReviewSimulate(
        as_data=ASReviewData.from_file(project_data_path),
        model=classifier(),
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
    
    export_name = f'{classifier.__name__}_{feature_extractor.__name__}_output.asreview'
    #file_list.append(export_name)
    #project.export(export_name) #Export .asreview file to perform metrics later

    shutil.rmtree(project_path)

def ploting(states):
    fig, ax = plt.subplots()

    for state_file in states:
        with open_state(state_file) as state:
            plot_recall(ax, state,x_absolute=True,y_absolute=True)
            label = state_file
            ax.lines[-2].set_label(label)

    ax.legend(loc=4, prop={"size": 8})

    plt.show()

for classifier in classifiers:
    for feature_extractor in feature_extractors:
        test_asreview_simulation(classifier,feature_extractor)

#ploting(file_list)



