from sentence_transformers import SentenceTransformer
from asreview.models.feature_extraction.base import BaseFeatureExtraction


class LaBSE(BaseFeatureExtraction):
    """LaBSE Feature Extractor

    This multilingual feature extractor is based on 'sentence-transformers/LaBSE'.

    """

    name = "labse"
    label = "LaBSE Transformer (max_seq_length: 256)"
    
    def fit(self, texts):
        self.model = SentenceTransformer("sentence-transformers/LaBSE")

    def transform(self, texts):
        print("Max sequence length:", self.model.max_seq_length)
        return self.model.encode(texts, show_progress_bar=True)
