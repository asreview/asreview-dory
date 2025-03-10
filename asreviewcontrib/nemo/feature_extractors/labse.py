__all__ = ["LaBSE"]

from asreviewcontrib.nemo.utils import min_max_normalize


class LaBSE:
    """
    LaBSE Feature Extractor

    Multilingual transformer-based feature extractor using the 
    'sentence-transformers/LaBSE' model. Designed for tasks involving 
    text embedding across multiple languages.

    Parameters
    ----------
    model_name : str, optional
        The name of the LaBSE model to use.
        Default: 'sentence-transformers/LaBSE'
    normalize : bool, optional
        Whether to normalize the embeddings.
        Default: False
    verbose : bool, optional
        Whether to print progress information.
        Default: True
    """

    name = "labse"
    label = "LaBSE Transformer"

    def __init__(
        self,
        model_name="sentence-transformers/LaBSE",
        normalize=False,
        verbose=True,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.verbose = verbose
        self._model_instance = None

    @property
    def _model(self):
        if self._model_instance is None:
            # Lazy import
            from sentence_transformers import SentenceTransformer
            self._model_instance = SentenceTransformer(self.model_name)
            if self.verbose:
                print(f"Model '{self.model_name}' has been loaded.")
        return self._model_instance

    def transform(self, texts):
        if self.verbose:
            print(f"Encoding texts using {self.model_name}, this may take a while...")
        embeddings = self._model.encode(texts, show_progress_bar=self.verbose)
        if self.normalize:
            if self.verbose:
                print("Normalizing embeddings.")
            embeddings = min_max_normalize(embeddings)
        return embeddings
