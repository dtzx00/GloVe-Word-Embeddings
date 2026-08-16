import os
import pickle
import re
import shutil

import numpy as np
import requests

BASE_URL = "https://embedding-files-open-access.s3.us-east-1.amazonaws.com"
CACHE_DIR = os.environ.get("GLOVE_WORD_EMBEDDINGS_CACHE") or os.path.expanduser(
    "~/.cache/glove-word-embeddings"
)

FILES = {
    "glove-840b-300d": "glove.840B.300d.pickle",
    "glove-6b-300d": "glove.6B.300d.pickle",
    "dolma-300-2024-1m": "dolma_300_2024_1M.pickle",
    "wiki-news-300d-1m": "wiki-news-300d-1M.pickle",
    "glove-olson-validated": "glove_olson_validated.pickle",
    "flair-olson-glove": "flair_olson_common_words_glove.pickle",
    "flair-olson-extvec": "flair_olson_common_words_extvec.pickle",
    "flair-olson-crawl": "flair_olson_common_words_crawl.pickle",
    "flair-olson-news": "flair_olson_common_words_news.pickle",
    "flair-olson-twitter": "flair_olson_common_words_twitter.pickle",
    "flair-olson-turian": "flair_olson_common_words_turian.pickle",
    "flair-olson-random": "flair_olson_common_words_random.pickle",
    "olson-validated-words": "olson_validated_100k_words.txt",
}


_validated_words = None


def list_models():
    return FILES


def clean_up():
    global _validated_words
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    _validated_words = None


def load(key: str, force_download: bool = False):
    """Download `key` if not cached, then return a Model (or word set for the wordlist)."""
    if key not in FILES:
        raise KeyError(f"Unknown key {key!r}. Valid keys: {sorted(FILES)}")

    filename = FILES[key]
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)

    if force_download or not os.path.exists(path):
        resp = requests.get(f"{BASE_URL}/{filename}", stream=True)
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    if filename.endswith(".pickle"):
        with open(path, "rb") as f:
            return Model(pickle.load(f))
    else:
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}


# --- preprocess / data cleaning -----------------------------------------

class Preprocess:
    STOPWORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with"}

    @staticmethod
    def validate(word: str) -> bool:
        """True only for single words that appear in Olson’s validated list.
        Multi-word phrases always return False.
        """
        global _validated_words
        if _validated_words is None:
            _validated_words = load("olson-validated-words")
        if " " in word.strip():
            return False
        return word in _validated_words

    @staticmethod
    def remove_stopwords(phrase: str) -> list[str]:
        """Return the non-stopword tokens of a phrase (lower-cased)."""
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", phrase.lower())
        return [t for t in tokens if t not in Preprocess.STOPWORDS]


class Model:
    def __init__(self, vectors: dict):
        self.vectors = vectors

    def embed(self, word: str):
        """Exact match only."""
        return self.vectors.get(word)

    def embed_phrase(self, phrase: str):
        """Original paper logic: try space/_/- variants, otherwise average non-stopword parts."""
        phrase = phrase.strip().lower()
        if not phrase:
            return None

        for variant in (phrase, phrase.replace(" ", "_"), phrase.replace(" ", "-")):
            if variant in self.vectors:
                return self.vectors[variant]

        parts = [self.vectors[p] for p in Preprocess.remove_stopwords(phrase) if p in self.vectors]
        if parts and len({p.shape for p in parts}) == 1:
            return np.mean(np.stack(parts).astype(np.float32), axis=0)
        return None

    def vocab(self):
        return set(self.vectors)
