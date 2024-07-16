__all__ = ["MXBAI"]

from asreview.models.feature_extraction.base import BaseFeatureExtraction
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from asreviewcontrib.models.utils import min_max_normalize


class MXBAI(BaseFeatureExtraction):
    """
    MXBAI Feature Extractor

    This transformer-based feature extractor is based on 
    'mixedbread-ai/mxbai-embed-large-v1'.

    Parameters
    ----------
    model_name : str, optional
        The name of the transformer model to use.
        Default: 'mixedbread-ai/mxbai-embed-large-v1'
    quantize : bool, optional
        Whether to quantize the embeddings.
        Default: False
    precision : str, optional
        The precision type to use for quantization.
        Default: 'ubinary'
    normalize : bool, optional
        Whether to normalize the embeddings.
        Default: False
    verbose : bool, optional
        Whether to print progress information.
        Default: True
    """

    name = "mxbai"
    label = "Mixedbread.AI Large V1"

    def __init__(
        self,
        *args,
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        quantize=False,
        precision="ubinary",
        normalize=False,
        verbose=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.quantize = quantize
        self.precision = precision
        self.normalize = normalize
        self.verbose = verbose

    @property
    def _model(self):
        if not hasattr(self, "model"):
            self.model = SentenceTransformer(self.model_name)
            if self.verbose:
                print(f"Model '{self.model_name}' has been loaded.")
        return self.model

    def transform(self, texts):
        if self.verbose:
            print("Encoding texts, this may take a while.")
        embeddings = self._model.encode(
            texts, convert_to_tensor=self.quantize, show_progress_bar=self.verbose      # if quantizing, the output needs to be a tensor    # noqa: E501
        )                                                                               # therefore, convert_to_tensor=self.quantize        # noqa: E501
        if self.quantize:
            if self.verbose:
                print(f"Quantizing embeddings with precision '{self.precision}'.")
            embeddings = quantize_embeddings(embeddings, precision=self.precision).numpy()  # noqa: E501
        if self.normalize:
            if self.verbose:
                print("Normalizing embeddings.")
            embeddings = min_max_normalize(embeddings)
        return embeddings
