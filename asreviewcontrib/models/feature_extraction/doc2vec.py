__all__ = ["Doc2Vec"]

import numpy as np

from gensim.models.doc2vec import Doc2Vec as GenSimDoc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

from asreview.models.feature_extraction.base import BaseFeatureExtraction
from asreviewcontrib.models.utils import min_max_normalize


class Doc2Vec(BaseFeatureExtraction):
    """Doc2Vec feature extraction technique (``doc2vec``).

    Feature extraction technique provided by the `gensim
    <https://radimrehurek.com/gensim/>`__ package. It takes relatively long to
    create a feature matrix with this method. However, this only has to be
    done once per simulation/review. The upside of this method is the
    dimension- reduction that generally takes place, which makes the modelling
    quicker.

    .. note::

        Note that for a fully deterministically-reproducible run, you must also
        limit the model to a single worker thread (`n_jobs=1`), to eliminate
        ordering jitter from OS thread scheduling.

    Parameters
    ----------
    vector_size : int
        Output size of the vector.
    epochs : int
        Number of epochs to train the doc2vec model.
    min_count : int
        Minimum number of occurrences for a word in the corpus for it to
        be included in the model.
    n_jobs : int
        Number of threads to train the model with.
    window : int
        Maximum distance over which word vectors influence each other.
    dm_concat : bool
        Whether to concatenate word vectors or not.
        See paper for more detail.
    dm : int
        Model to use.
        0: Use distribute bag of words (DBOW).
        1: Use distributed memory (DM).
        2: Use both of the above with half the vector size and concatenate
        them.
    dbow_words : bool
        Whether to train the word vectors using the skipgram method.
    normalize : bool
        Whether to normalize the embeddings.
    verbose : bool
        Whether to print progress information.
    """

    name = "doc2vec"
    label = "Doc2Vec"

    def __init__(
        self,
        *args,
        vector_size=40,
        epochs=33,
        min_count=1,
        n_jobs=1,
        window=7,
        dm_concat=False,
        dm=2,
        dbow_words=False,
        normalize=True,
        verbose=True,
        **kwargs,
    ):
        """Initialize the doc2vec model."""
        super().__init__(*args, **kwargs)
        self.vector_size = int(vector_size)
        self.epochs = int(epochs)
        self.min_count = int(min_count)
        self.n_jobs = int(n_jobs)
        self.window = int(window)
        self.dm_concat = 0 if dm_concat is False else 1
        self.dm = int(dm)
        self.dbow_words = 0 if dbow_words is False else 1
        self.normalize = normalize
        self.verbose = verbose
        self._model = None
        self._model_dm = None
        self._model_dbow = None

    def fit(self, texts):
        model_param = {
            "vector_size": self.vector_size,
            "epochs": self.epochs,
            "min_count": self.min_count,
            "workers": self.n_jobs,
            "window": self.window,
            "dm_concat": self.dm_concat,
            "dbow_words": self.dbow_words,
        }

        corpus = [
            TaggedDocument(simple_preprocess(text), [i]) for i, text in enumerate(texts)
        ]

        if self.dm == 2:
            model_param["vector_size"] = int(model_param["vector_size"] / 2)
            if self.verbose:
                print("Training DM model...")
            self._model_dm = _train_model(corpus,
                                          **model_param, dm=1, verbose=self.verbose)
            if self.verbose:
                print("Training DBOW model...")
            self._model_dbow = _train_model(corpus,
                                            **model_param, dm=0, verbose=self.verbose)
        else:
            if self.verbose:
                print(f"Training model with dm={self.dm}...")
            self._model = _train_model(corpus,
                                       **model_param, dm=self.dm, verbose=self.verbose)


    def transform(self, texts):
        corpus = [
            TaggedDocument(simple_preprocess(text), [i]) for i, text in enumerate(texts)
        ]

        if self.dm == 2:
            X_dm = _transform_text(self._model_dm, corpus)
            X_dbow = _transform_text(self._model_dbow, corpus)
            X = np.concatenate((X_dm, X_dbow), axis=1)
        else:
            X = _transform_text(self._model, corpus)
        if self.verbose:
            print("Finished transforming texts to vectors")
        if self.normalize:
            X = min_max_normalize(X)
        return X


def _train_model(corpus, *args, verbose=False, **kwargs):
    model = GenSimDoc2Vec(*args, **kwargs)
    if verbose:
        print("Building vocabulary...")
    model.build_vocab(corpus)
    if verbose:
        print("Training model...")
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    if verbose:
        print("Model training complete.")
    return model


def _transform_text(model, corpus, verbose=False):
    if verbose:
        print("Inferring vectors for documents...")
    X = []
    for doc_id in range(len(corpus)):
        doc_vec = model.infer_vector(corpus[doc_id].words)
        X.append(doc_vec)
    if verbose:
        print("Vector inference complete.")
    return np.array(X)
