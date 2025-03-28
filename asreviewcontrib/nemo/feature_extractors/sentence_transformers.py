__all__ = [
    "LaBSE",
    "MXBAI",
    "SBERT",
    "MultilingualE5Large",
    "GTR",
]

from asreview.models.feature_extractors import TextMerger
from sentence_transformers import SentenceTransformer, models


class BaseSentenceTransformer:
    """
    Base class for sentence transformer feature extractors.
    """

    def __init__(
        self,
        model_name,
        normalize=False,
        quantize=False,
        precision="ubinary",
        verbose=True,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.quantize = quantize
        self.precision = precision
        self.verbose = verbose

    def _load_model(self):
        model = SentenceTransformer(self.model_name)
        if self.verbose:
            print(f"Model '{self.model_name}' has been loaded.")
        return model

    def fit_transform(self, texts):
        self._model = self._load_model()

        texts = TextMerger(columns=["title", "abstract"]).transform(texts)

        if self.verbose:
            print("Embedding text...")

        embeddings = self._model.encode(texts, show_progress_bar=self.verbose)

        if self.normalize:
            from asreviewcontrib.nemo.utils import min_max_normalize

            embeddings = min_max_normalize(embeddings)

        if self.quantize:
            from sentence_transformers.quantization import quantize_embeddings

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

    def __init__(
        self,
        model_name="all-mpnet-base-v2",
        is_pretrained_sbert=True,
        pooling_mode="mean",
    ):
        self.model_name = model_name
        self.is_pretrained_sbert = is_pretrained_sbert
        self.pooling_mode = pooling_mode
        super().__init__(model_name)

    def _load_model(self):
        if self.is_pretrained_sbert:
            model = SentenceTransformer(self.model_name)
            if self.verbose:
                print(f"Model '{self.model_name}' has been loaded.")
            return model
        else:
            word_embedding_model = models.Transformer(self.model_name)
            pooling_layer = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode=self.pooling_mode,
            )
            return SentenceTransformer(modules=[word_embedding_model, pooling_layer])


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
