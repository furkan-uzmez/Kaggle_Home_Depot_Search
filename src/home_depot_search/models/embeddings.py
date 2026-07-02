import numpy as np

from home_depot_search.utils.reproducibility import set_reproducibility


class TransformerEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", seed=42):
        set_reproducibility(seed)
        self.model_name = model_name
        self._model = None

    def fit(self, texts):
        self.fitted_ = True
        return self

    def transform(self, texts):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is required for TransformerEmbedder")

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(texts)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


class Word2VecEmbedder:
    def __init__(self, vector_size=100, window=5, seed=42):
        set_reproducibility(seed)
        self.vector_size = vector_size
        self.window = window
        self._model = None
        self._vector_size = vector_size

    def fit(self, texts):
        try:
            from gensim.models import Word2Vec
        except ImportError:
            raise ImportError("gensim is required for Word2VecEmbedder")

        tokenized = [text.lower().split() for text in texts]
        self._model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            workers=1,
            seed=42,
        )
        return self

    def transform(self, texts):
        if self._model is None:
            raise RuntimeError("Word2VecEmbedder must be fitted before transform")

        tokenized = [text.lower().split() for text in texts]
        result = np.zeros((len(texts), self._vector_size))
        for i, tokens in enumerate(tokenized):
            vectors = [self._model.wv[token] for token in tokens if token in self._model.wv]
            if vectors:
                result[i] = np.mean(vectors, axis=0)
        return result

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
