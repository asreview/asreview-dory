__all__ = [
    "LaBSE",
    "MXBAI",
    "SBERT",
    "MultilingualE5Large",
    "GTR",
]

from functools import cached_property

from asreview.models.feature_extractors import TextMerger
from sentence_transformers import SentenceTransformer, quantize_embeddings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class BaseSentenceTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for sentence transformer feature extractors.
    """

    def __init__(
        self,
        model_name,
        normalize,
        quantize,
        precision,
        verbose,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.quantize = quantize
        self.precision = precision
        self.verbose = verbose

    @cached_property
    def _model(self):
        return self._load_model()

    def _load_model(self):
        model = SentenceTransformer(self.model_name)
        if self.verbose:
            print(f"Model '{self.model_name}' has been loaded.")
        return model

    def fit(self, X, y=None):
        # Necessary for being last step of a pipeline
        return self

    def fit_transform(self, X, y=None):
        if self.verbose:
            print("Embedding text...")

        embeddings = self._model.encode(
            X, show_progress_bar=self.verbose, normalize_embeddings=self.normalize
        )

        if self.quantize:
            embeddings = quantize_embeddings(
                embeddings, precision=self.precision
            ).numpy()
        return embeddings


class LaBSE(Pipeline):
    """
    LaBSE Feature Extractor using the 'sentence-transformers/LaBSE' model.
    """

    name = "labse"
    label = "LaBSE Transformer"

    def __init__(
        self,
        columns=None,
        sep=" ",
        model_name="sentence-transformers/LaBSE",
        normalize=True,
        quantize=False,
        precision="ubinary",
        verbose=True,
    ):
        self.columns = ["title", "abstract"] if columns is None else columns
        self.sep = sep
        self.model_name = model_name
        self.normalize = normalize
        self.quantize = quantize
        self.precision = precision
        self.verbose = verbose

        super().__init__(
            [
                ("text_merger", TextMerger(columns=self.columns, sep=self.sep)),
                (
                    "sentence_transformer",
                    BaseSentenceTransformer(
                        model_name=self.model_name,
                        normalize=self.normalize,
                        quantize=self.quantize,
                        precision=self.precision,
                        verbose=self.verbose,
                    ),
                ),
            ]
        )


class MXBAI(Pipeline):
    """
    MXBAI Feature Extractor based on 'mixedbread-ai/mxbai-embed-large-v1'.
    """

    name = "mxbai"
    label = "mxbai Sentence BERT"

    def __init__(
        self,
        columns=None,
        sep=" ",
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        normalize=True,
        quantize=False,
        precision="ubinary",
        verbose=True,
    ):
        self.columns = ["title", "abstract"] if columns is None else columns
        self.sep = sep
        self.model_name = model_name
        self.normalize = normalize
        self.quantize = quantize
        self.precision = precision
        self.verbose = verbose

        super().__init__(
            [
                ("text_merger", TextMerger(columns=self.columns, sep=self.sep)),
                (
                    "sentence_transformer",
                    BaseSentenceTransformer(
                        model_name=self.model_name,
                        normalize=self.normalize,
                        quantize=self.quantize,
                        precision=self.precision,
                        verbose=self.verbose,
                    ),
                ),
            ]
        )


class SBERT(Pipeline):
    """
    Sentence BERT feature extractor.
    """

    name = "sbert"
    label = "mpnet Sentence BERT"

    def __init__(
        self,
        columns=None,
        sep=" ",
        model_name="all-mpnet-base-v2",
        normalize=True,
        quantize=False,
        precision="ubinary",
        verbose=True,
    ):
        self.columns = ["title", "abstract"] if columns is None else columns
        self.sep = sep
        self.model_name = model_name
        self.normalize = normalize
        self.quantize = quantize
        self.precision = precision
        self.verbose = verbose

        super().__init__(
            [
                ("text_merger", TextMerger(columns=self.columns, sep=self.sep)),
                (
                    "sentence_transformer",
                    BaseSentenceTransformer(
                        model_name=self.model_name,
                        normalize=self.normalize,
                        quantize=self.quantize,
                        precision=self.precision,
                        verbose=self.verbose,
                    ),
                ),
            ]
        )


class MultilingualE5Large(Pipeline):
    """
    Multilingual E5 Large Feature Extractor using the
    'intfloat/multilingual-e5-large' model.
    """

    name = "multilingual-e5-large"
    label = "Multilingual E5 Large"

    def __init__(
        self,
        columns=None,
        sep=" ",
        model_name="intfloat/multilingual-e5-large",
        normalize=True,
        quantize=False,
        precision="ubinary",
        verbose=True,
    ):
        self.columns = ["title", "abstract"] if columns is None else columns
        self.sep = sep
        self.model_name = model_name
        self.normalize = normalize
        self.quantize = quantize
        self.precision = precision
        self.verbose = verbose

        super().__init__(
            [
                ("text_merger", TextMerger(columns=self.columns, sep=self.sep)),
                (
                    "sentence_transformer",
                    BaseSentenceTransformer(
                        model_name=self.model_name,
                        normalize=self.normalize,
                        quantize=self.quantize,
                        precision=self.precision,
                        verbose=self.verbose,
                    ),
                ),
            ]
        )


class GTR(Pipeline):
    """
    GTR-T5-Large Feature Extractor using the
    'gtr-t5-large' model.
    """

    name = "gtr-t5-large"
    label = "Google GTR"

    def __init__(
        self,
        columns=None,
        sep=" ",
        model_name="gtr-t5-large",
        normalize=True,
        quantize=False,
        precision="ubinary",
        verbose=True,
    ):
        self.columns = ["title", "abstract"] if columns is None else columns
        self.sep = sep
        self.model_name = model_name
        self.normalize = normalize
        self.quantize = quantize
        self.precision = precision
        self.verbose = verbose

        super().__init__(
            [
                ("text_merger", TextMerger(columns=self.columns, sep=self.sep)),
                (
                    "sentence_transformer",
                    BaseSentenceTransformer(
                        model_name=self.model_name,
                        normalize=self.normalize,
                        quantize=self.quantize,
                        precision=self.precision,
                        verbose=self.verbose,
                    ),
                ),
            ]
        )
