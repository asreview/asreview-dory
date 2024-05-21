# Copyright 2019-2024 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["KNN"]

from asreview.models.classifiers.base import BaseTrainClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNN(BaseTrainClassifier):
    """KNN classifier

    This classifier is based on the KNeighborsClassifier from scikit-learn, and
    implementation of the the k-nearest neighbors algorithm (k-NN).

    Needs at least 5 samples to be able to fit the model.

    """

    name = "knn"
    label = "k-NN"

    def __init__(self):
        super().__init__()
        self._is_fit = False
        self._model = KNeighborsClassifier()

    def fit(self, X, y):
        if X.shape[0] > 5:
            self._is_fit = True
            return self._model.fit(X, y)

    def predict_proba(self, X):
        if self._is_fit:
            return self._model.predict_proba(X)
        else:
            return np.zeros((X.shape[0], 2))