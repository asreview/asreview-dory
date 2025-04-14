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


class BaseSentenceTransformer:
    """
    Base class for sentence transformer feature extractors.
    """

    def __init__(
        self,
        model_name,
        normalize=True,
        quantize=False,
        precision="ubinary",
        verbose=True,
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

    def fit_transform(self, texts):
        texts = TextMerger(columns=["title", "abstract"]).transform(texts)

        if self.verbose:
            print("Embedding text...")

        embeddings = self._model.encode(
            texts, show_progress_bar=self.verbose, normalize_embeddings=self.normalize
        )

        if self.quantize:
            embeddings = quantize_embeddings(
                embeddings, precision=self.precision
            ).numpy()
        return embeddings


class LaBSE(BaseSentenceTransformer):
    """
    LaBSE Feature Extractor using the 'sentence-transformers/LaBSE' model.
    """

    name = "labse"
    label = "LaBSE Transformer"

    def __init__(self, model_name="sentence-transformers/LaBSE"):
        super().__init__(model_name)


class MXBAI(BaseSentenceTransformer):
    """
    MXBAI Feature Extractor based on 'mixedbread-ai/mxbai-embed-large-v1'.
    """

    name = "mxbai"
    label = "mxbai Sentence BERT"

    def __init__(
        self,
        model_name="mixedbread-ai/mxbai-embed-large-v1",
    ):
        super().__init__(model_name)


class SBERT(BaseSentenceTransformer):
    """
    Sentence BERT feature extractor.
    """

    name = "sbert"
    label = "mpnet Sentence BERT"

    def __init__(self, model_name="all-mpnet-base-v2"):
        super().__init__(model_name)


class MultilingualE5Large(BaseSentenceTransformer):
    """
    Multilingual E5 Large Feature Extractor using the
    'intfloat/multilingual-e5-large' model.
    """

    name = "multilingual-e5-large"
    label = "Multilingual E5 Large"

    def __init__(self, model_name="intfloat/multilingual-e5-large"):
        super().__init__(model_name)


class GTR(BaseSentenceTransformer):
    """
    GTR-T5-Large Feature Extractor using the
    'gtr-t5-large' model.
    """

    name = "gtr-t5-large"
    label = "Google GTR"

    def __init__(self, model_name="gtr-t5-large"):
        super().__init__(model_name)
