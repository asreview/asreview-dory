__all__ = ["SBERT"]

from asreview.models.feature_extraction.base import BaseFeatureExtraction
from asreviewcontrib.nemo_models.utils import min_max_normalize


class SBERT(BaseFeatureExtraction):
    """
    Sentence BERT feature extraction technique (``sbert``).

    By setting the ``transformer_model`` parameter, you can use other
    transformer models. For example, ``transformer_model='bert-base-nli-stsb-large'``.
    For a list of available models, see the `Sentence BERT documentation 
    <https://huggingface.co/sentence-transformers>`__.

    Sentence BERT is a sentence embedding model that is trained on a large
    corpus of human written text. It is a fast and accurate model that can
    be used for many tasks.

    The huggingface library includes multilingual text classification models. If
    your dataset contains records with multiple languages, you can use the
    ``transformer_model`` parameter to select the model that is most suitable
    for your data.

    Parameters
    ----------
    transformer_model : str, optional
        The transformer model to use.
        Default: 'all-mpnet-base-v2'
    is_pretrained_sbert : bool, optional
        Whether to use a pretrained SBERT model.
        Default: True
    pooling_mode : str, optional
        Pooling mode to get sentence embeddings from word embeddings.
        Options are 'mean', 'max', and 'cls'.
        Default: 'mean'
        Only used if is_pretrained_sbert=False
    normalize : bool, optional
        Whether to normalize the embeddings.
        Default: False
    verbose : bool, optional
        Whether to print progress information.
        Default: True
    """
    name = "sbert"
    label = "mpnet Sentence BERT"

    def __init__(
        self,
        *args,
        transformer_model="all-mpnet-base-v2",
        is_pretrained_sbert=True,
        pooling_mode="mean",
        normalize=False,
        verbose=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.transformer_model = transformer_model
        self.is_pretrained_sbert = is_pretrained_sbert
        self.pooling_mode = pooling_mode
        self.normalize = normalize
        self.verbose = verbose
        self._model_instance = None

    @property
    def _model(self):
        if self._model_instance is None:
            # Lazy import
            from sentence_transformers import models, SentenceTransformer

            if self.is_pretrained_sbert:
                self._model_instance = SentenceTransformer(self.transformer_model)
            else:
                word_embedding_model = models.Transformer(self.transformer_model)
                pooling_layer = models.Pooling(
                    word_embedding_model.get_word_embedding_dimension(),
                    pooling_mode=self.pooling_mode,
                )
                self._model_instance = SentenceTransformer(
                    modules=[word_embedding_model, pooling_layer]
                )
            if self.verbose:
                print(f"Model '{self.transformer_model}' has been loaded.")
        return self._model_instance

    def transform(self, texts):
        if self.verbose:
            print(f"Encoding texts using {self.transformer_model}, \
                  this may take a while...")
        embeddings = self._model.encode(texts, show_progress_bar=self.verbose)
        if self.normalize:
            embeddings = min_max_normalize(embeddings)
        return embeddings
