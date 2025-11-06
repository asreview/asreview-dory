__all__ = ["Doc2Vec"]

import numpy as np
from asreview.models.feature_extractors import TextMerger
from gensim.models.doc2vec import Doc2Vec as GenSimDoc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.pipeline import Pipeline


class Doc2Vec(Pipeline):
    name = "doc2vec"
    label = "Doc2Vec"

    def __init__(self, **kwargs):
        if "ngram_range" in kwargs:
            kwargs["ngram_range"] = tuple(kwargs["ngram_range"])

        super().__init__(
            [
                ("text_merger", TextMerger(columns=["title", "abstract"])),
                ("doc2vec", Doc2VecBase(**kwargs)),
            ]
        )


class Doc2VecBase:
    """
    Doc2Vec feature extraction technique (``doc2vec``).

    Feature extraction technique provided by the `gensim
    <https://radimrehurek.com/gensim/>`__ package. It trains a model to generate
    document embeddings, which can reduce dimensionality and accelerate modeling.

    .. note::

        For fully reproducible runs, limit the model to a single worker thread
        (`n_jobs=1`) to eliminate potential variability due to thread scheduling.

    Parameters
    ----------
    vector_size : int, optional
        Dimensionality of the feature vectors. Default: 40
    epochs : int, optional
        Number of epochs to train the model. Default: 33
    min_count : int, optional
        Ignores all words with total frequency lower than this. Default: 1
    n_jobs : int, optional
        Number of threads to use during training. Default: 1
    window : int, optional
        Maximum distance between the current and predicted word. Default: 7
    dm_concat : bool, optional
        If True, concatenate word vectors. Default: False
    dm : int, optional
        Training model:
        - 0: Distributed Bag of Words (DBOW)
        - 1: Distributed Memory (DM)
        - 2: Both DBOW and DM (concatenated embeddings). Default: 2
    dbow_words : bool, optional
        Train word vectors alongside DBOW. Default: False
    verbose : bool, optional
        Print progress and status updates. Default: True
    """

    def __init__(
        self,
        vector_size=40,
        epochs=33,
        min_count=1,
        n_jobs=1,
        window=7,
        dm_concat=False,
        dm=2,
        dbow_words=False,
        verbose=True,
    ):
        self.vector_size = int(vector_size)
        self.epochs = int(epochs)
        self.min_count = int(min_count)
        self.n_jobs = int(n_jobs)
        self.window = int(window)
        self.dm_concat = 1 if dm_concat else 0
        self.dm = int(dm)
        self.dbow_words = 1 if dbow_words else 0
        self.verbose = verbose
        self._model_instance = None

        self._tagged_document = TaggedDocument
        self._simple_preprocess = simple_preprocess
        self._model = GenSimDoc2Vec

    def fit(self, X, y=None):
        if self.verbose:
            print("Preparing corpus...")
        corpus = [
            self._tagged_document(self._simple_preprocess(text), [i])
            for i, text in enumerate(X)
        ]

        model_param = {
            "vector_size": self.vector_size,
            "epochs": self.epochs,
            "min_count": self.min_count,
            "workers": self.n_jobs,
            "window": self.window,
            "dm_concat": self.dm_concat,
            "dbow_words": self.dbow_words,
        }

        if self.dm == 2:
            # Train both DM and DBOW models
            model_param["vector_size"] = int(self.vector_size / 2)
            if self.verbose:
                print("Training DM model...")
            self._model_dm = self._train_model(corpus, **model_param, dm=1)
            if self.verbose:
                print("Training DBOW model...")
            self._model_dbow = self._train_model(corpus, **model_param, dm=0)
        else:
            if self.verbose:
                print(f"Training single model with dm={self.dm}...")
            self._model_instance = self._train_model(corpus, **model_param, dm=self.dm)

    def transform(self, texts):
        if self.verbose:
            print("Preparing corpus for transformation...")
        corpus = [
            self._tagged_document(self._simple_preprocess(text), [i])
            for i, text in enumerate(texts)
        ]

        if self.dm == 2:
            X_dm = self._infer_vectors(self._model_dm, corpus)
            X_dbow = self._infer_vectors(self._model_dbow, corpus)
            X = np.concatenate((X_dm, X_dbow), axis=1)
        else:
            X = self._infer_vectors(self._model_instance, corpus)

        if self.verbose:
            print("Finished transforming texts to vectors.")

        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _train_model(self, corpus, *args, **kwargs):
        model = self._model(*args, **kwargs)
        if self.verbose:
            print("Building vocabulary...")
        model.build_vocab(corpus)
        if self.verbose:
            print("Training model...")
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        if self.verbose:
            print("Model training complete.")
        return model

    def _infer_vectors(self, model, corpus):
        if self.verbose:
            print("Inferring vectors for documents...")
        X = [model.infer_vector(doc.words) for doc in corpus]
        if self.verbose:
            print("Vector inference complete.")
        return np.array(X)
