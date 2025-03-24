__all__ = ["Doc2Vec"]

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from asreview.models.feature_extractors import TextMerger
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader


class Doc2Vec(Pipeline):
    name = "doc2vec"
    label = "Doc2Vec"

    def __init__(self, **kwargs):
        if "ngram_range" in kwargs:
            kwargs["ngram_range"] = tuple(kwargs["ngram_range"])

        super().__init__(
            [
                ("text_merger", TextMerger(columns=["title", "abstract"])),
                ("tfidf", Doc2VecWrapper(**kwargs)),
            ]
        )


class Doc2VecModel(nn.Module):
    def __init__(
        self, vocab_size, doc_size, embedding_dim, dm, num_neg_samples, concat
    ):
        super().__init__()
        self.dm = dm
        self.num_neg_samples = num_neg_samples
        self.concat = concat

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.doc_embeddings = nn.Parameter(torch.randn(doc_size, embedding_dim))
        self.dropout = nn.Dropout(p=0.3)

        if self.concat and self.dm == 1:
            self.fc = nn.Linear(embedding_dim * 2, vocab_size)

        else:
            self.fc = nn.Linear(embedding_dim, vocab_size)

        self.register_buffer("word_freqs", torch.ones(vocab_size))
        self.precomputed_neg_samples = None

    def forward(self, doc_ids, word_ids, neg_samples=None):
        doc_embeds = self.dropout(self.doc_embeddings[doc_ids])

        if self.dm == 1:  # DM model
            word_embeds = self.dropout(self.word_embeddings(word_ids))

            if self.concat:
                combined = torch.cat([doc_embeds, word_embeds], dim=-1)
            else:
                combined = (doc_embeds + word_embeds) / 2
        else:  # DBOW model
            combined = doc_embeds

        output = self.fc(combined)

        if neg_samples is not None:
            return output, self.word_embeddings(neg_samples)

        return output

    def set_word_freqs(self, word_freqs):
        self.register_buffer("word_freqs", word_freqs)
        self._precompute_negative_samples()

    @torch.no_grad()
    def get_embeddings(self, index):
        return self.doc_embeddings[index, :].data.tolist()

    def get_neg_samples(self, batch_size, exclude_word):
        """
        Fetch negative samples from the precomputed buffer while avoiding the
        exclude_word.
        """
        if self.precomputed_neg_samples is None:
            raise ValueError("Negative samples not precomputed!")

        valid_samples = []
        while len(valid_samples) < batch_size * self.num_neg_samples:
            sample = self.precomputed_neg_samples[self.neg_sample_index]
            self.neg_sample_index += 1

            # Now check element-wise if any element in 'sample' is equal to exclude_word
            if torch.all(
                torch.ne(sample, exclude_word)
            ):  # If none of the samples equals exclude_word
                valid_samples.append(sample)

            # If buffer exhausted, refill it
            if self.neg_sample_index >= len(self.precomputed_neg_samples):
                self._precompute_negative_samples()
                self.neg_sample_index = 0

        return torch.tensor(valid_samples).view(batch_size, self.num_neg_samples)

    def _precompute_negative_samples(self, buffer_size=10000):
        """Precompute a buffer of negative samples."""
        weights = self.word_freqs.clone()
        weights = weights / weights.sum()  # Normalize probabilities
        self.precomputed_neg_samples = torch.multinomial(
            weights, buffer_size, replacement=True
        )
        self.neg_sample_index = 0


class Doc2VecWrapper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        vector_size=40,
        epochs=33,
        min_count=1,
        window=7,
        dm=1,
        num_neg_samples=5,
        concat=False,
        batch_size=64,
        normalize=True,
        subsampling_threshold=1e-4,
        verbose=False,
    ):
        self.vector_size = vector_size
        self.epochs = epochs
        self.min_count = min_count
        self.window = window
        self.dm = dm
        self.num_neg_samples = num_neg_samples
        self.concat = concat
        self.batch_size = batch_size
        self.normalize = normalize
        self.subsampling_threshold = subsampling_threshold
        self.verbose = verbose

        self.model = None

    def fit(self, X, y=None):
        self.vocab, self.doc_ids, self.data, word_freqs = self._prepare_data(
            X, self.subsampling_threshold
        )
        self.vocab_size = len(self.vocab)
        self.doc_size = len(self.doc_ids)
        self.model = Doc2VecModel(
            self.vocab_size,
            self.doc_size,
            self.vector_size,
            self.dm,
            self.num_neg_samples,
            self.concat,
        )
        self.model.set_word_freqs(word_freqs)

        if torch.cuda.is_available():
            self.model.cuda()

        self.criterion = NegativeSampling(num_neg_samples=self.num_neg_samples)

        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.95
        )

        return (
            self.train_dm(self.batch_size)
            if self.dm == 1
            else self.train_dbow(self.batch_size)
        )

    def train_dm(self, batch_size):
        dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)

        for epoch in range(self.epochs):
            total_loss = 0

            for batch in dataloader:
                doc_ids, word_ids, target_word_ids = batch
                doc_tensor = torch.LongTensor(doc_ids)
                word_tensor = torch.LongTensor(word_ids)
                target_tensor = torch.LongTensor(target_word_ids)

                self.optimizer.zero_grad()

                output = self.model(doc_tensor, word_tensor)
                neg_samples = self.model.get_neg_samples(len(doc_ids), target_tensor)
                loss = self.criterion(output, target_tensor, neg_samples)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        return self

    def train_dbow(self, batch_size):
        dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)

        for epoch in range(self.epochs):
            total_loss = 0

            for batch in dataloader:
                doc_ids, word_ids = batch
                doc_tensor = torch.LongTensor(doc_ids)
                word_tensor = torch.LongTensor(word_ids)

                self.optimizer.zero_grad()

                output = self.model(doc_tensor, word_tensor)
                neg_samples = self.model.get_neg_samples(len(doc_ids), word_tensor)
                loss = self.criterion(output, word_tensor, neg_samples)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        return self

    def transform(self, X):
        embeddings = []

        for doc_id, _ in enumerate(X):
            embeddings.append(self.model.get_embeddings(doc_id))

        embeddings = np.array(embeddings)

        if self.normalize:
            embeddings = (embeddings - embeddings.min()) / (
                embeddings.max() - embeddings.min()
            )

        return embeddings

    def _prepare_data(self, documents, subsampling_threshold):
        word_counts = Counter()
        for doc in documents:
            word_counts.update(doc.split())

        word_total_count = sum(word_counts.values())

        vocab = {
            word: i
            for i, (word, count) in enumerate(word_counts.items())
            if count >= self.min_count
        }

        # Convert word counts to frequencies for negative sampling
        word_freqs = np.array([word_counts[word] for word in vocab])
        word_freqs = word_freqs**0.75  # Smoothing as in original word2vec
        word_freqs = word_freqs / word_freqs.sum()

        doc_ids = list(range(len(documents)))
        training_data = []

        for doc_id, doc in enumerate(documents):
            words = doc.split()
            words_filtered = [w for w in words if w in vocab]

            # Compute probability of keeping each word
            word_probs = {
                w: max(
                    0, 1 - np.sqrt(subsampling_threshold / (count / word_total_count))
                )
                for w, count in word_counts.items()
            }

            # Filter words based on their probabilities
            words = [
                w for w in words_filtered if word_probs.get(w, 0) < np.random.random()
            ]

            if not words:
                words = words_filtered

            if self.dm == 1:
                for i, target_word in enumerate(words):
                    target_word_id = vocab[target_word]

                    start = max(0, i - self.window)
                    end = min(len(words), i + self.window + 1)
                    context_words = words[start:i] + words[i + 1 : end]

                    for context_word in context_words:
                        context_word_id = vocab[context_word]
                        training_data.append((doc_id, context_word_id, target_word_id))

            else:
                for target_word in words:
                    target_word_id = vocab[target_word]
                    training_data.append((doc_id, target_word_id))

        return vocab, doc_ids, training_data, torch.from_numpy(word_freqs).float()


class NegativeSampling(nn.Module):
    def __init__(self, num_neg_samples):
        super().__init__()
        self.num_neg_samples = num_neg_samples
        self._log_sigmoid = nn.LogSigmoid()

    def forward(self, output, target, noise):
        # Gather the scores for positive samples
        pos_scores = output.gather(1, target.unsqueeze(1))
        pos_loss = self._log_sigmoid(pos_scores).squeeze(1)

        # Gather the scores for negative samples
        neg_scores = output.gather(1, noise)
        neg_loss = torch.sum(self._log_sigmoid(-neg_scores), dim=1)

        return -(pos_loss + neg_loss).mean()
