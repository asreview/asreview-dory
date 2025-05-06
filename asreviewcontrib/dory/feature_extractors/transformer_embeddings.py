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


class SentenceTransformerPipeline(Pipeline):
    default_model_name = None
    name = None
    label = None

    def __init__(
        self,
        columns=None,
        sep=" ",
        model_name=None,
        normalize=True,
        quantize=False,
        precision="ubinary",
        verbose=True,
    ):
        self.columns = ["title", "abstract"] if columns is None else columns
        self.sep = sep
        self.model_name = model_name or self.default_model_name
        self.normalize = normalize
        self.quantize = quantize
        self.precision = precision
        self.verbose = verbose

        steps = [
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

        super().__init__(steps)

    def __repr__(self):
        return f"<{self.__class__.__name__} model='{self.model_name}'>"

    def __str__(self):
        return self.__repr__()


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
        model = SentenceTransformer(self.model_name)
        if self.verbose:
            print(f"Model '{self.model_name}' has been loaded.")
        return model

    def fit(self, X, y=None):
        # Required func for last Pipeline step, but not required
        # for sentence-transformers, so return self
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


class LaBSE(SentenceTransformerPipeline):
    name = "labse"
    label = "LaBSE Transformer"
    default_model_name = "sentence-transformers/LaBSE"


class MXBAI(SentenceTransformerPipeline):
    name = "mxbai"
    label = "mxbai Sentence BERT"
    default_model_name = "mixedbread-ai/mxbai-embed-large-v1"


class SBERT(SentenceTransformerPipeline):
    name = "sbert"
    label = "mpnet Sentence BERT"
    default_model_name = "all-mpnet-base-v2"


class MultilingualE5Large(SentenceTransformerPipeline):
    name = "multilingual-e5-large"
    label = "Multilingual E5 Large"
    default_model_name = "intfloat/multilingual-e5-large"


class GTR(SentenceTransformerPipeline):
    name = "gtr-t5-large"
    label = "Google GTR"
    default_model_name = "gtr-t5-large"
