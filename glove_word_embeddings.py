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


# --- prep / data cleaning -----------------------------------------

class prep:
    STOPWORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with"}

    # category -> set of seed words  (WordNet ancestry can be added later)
    CATEGORIES = {
        "animals": {"dog", "cat", "lion", "tiger", "bear", "elephant", "horse", "cow", "sheep", "goat", "pig", "wolf", "fox", "deer", "rabbit", "monkey", "zebra", "giraffe", "kangaroo", "whale", "dolphin", "shark", "snake", "frog", "goose", "duck", "chicken", "hippo", "rhino", "leopard", "cheetah", "panda", "koala", "otter"},
        "birds": {"robin", "sparrow", "eagle", "hawk", "owl", "parrot", "penguin", "crow", "pigeon", "swan", "flamingo", "peacock", "finch"},
        "insects": {"ant", "bee", "wasp", "hornet", "beetle", "fly", "moth", "butterfly", "dragonfly", "cricket", "grasshopper", "ladybug"},
        "foods": {"bread", "cheese", "rice", "pasta", "pizza", "burger", "cake", "soup", "egg", "butter", "sugar", "salt", "chocolate", "cookie", "sandwich", "cereal", "biscuit", "jam", "honey"},
        "fruits": {"apple", "banana", "orange", "grape", "pear", "peach", "plum", "cherry", "mango", "melon", "lemon", "lime", "kiwi", "strawberry", "pineapple", "raspberry", "blueberry"},
        "vegetables": {"carrot", "potato", "onion", "pea", "peas", "bean", "beans", "broccoli", "spinach", "lettuce", "cabbage", "cucumber", "pepper", "tomato", "corn", "celery"},
        "plants": {"tree", "fern", "moss", "rose", "tulip", "daisy", "oak", "pine", "cactus", "ivy", "bamboo", "bush", "shrub", "vine"},
        "colors": {"red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "grey", "gray", "violet", "indigo", "cyan", "magenta", "turquoise", "maroon", "beige"},
        "body_parts": {"arm", "leg", "hand", "foot", "head", "eye", "ear", "nose", "mouth", "finger", "toe", "knee", "elbow", "shoulder", "heart", "liver", "lung", "brain", "teeth", "tooth", "hair", "skin"},
        "metals": {"gold", "silver", "iron", "copper", "bronze", "steel", "tin", "zinc", "lead", "aluminium", "aluminum", "nickel", "platinum"},
        "planets": {"mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "sun", "moon", "star", "comet", "asteroid", "supernova", "galaxy", "nebula", "meteor"},
        "sports": {"football", "soccer", "basketball", "tennis", "cricket", "golf", "rugby", "hockey", "baseball", "swimming", "boxing", "cycling", "skiing"},
        "tools": {"hammer", "screwdriver", "wrench", "drill", "saw", "pliers", "chisel", "axe", "spanner", "clamp"},
        "countries": {"france", "germany", "spain", "italy", "china", "japan", "india", "brazil", "canada", "egypt", "kenya", "peru", "chile", "mexico"},
        "environment": {
            "desk", "table", "chair", "bed", "sofa", "couch", "shelf", "bookshelf", "stool", "bench", "cabinet", "drawer", "wardrobe", "dresser", "nightstand",
            "wall", "walls", "floor", "ceiling", "door", "window", "roof", "stairs", "carpet", "rug", "curtain", "curtains", "blind", "blinds", "tile", "tiles",
            "corner", "room", "hallway", "fireplace", "radiator", "switch", "socket",
            "shoe", "shoes", "shirt", "tshirt", "sock", "socks", "hat", "cap", "jacket", "coat", "trousers", "pants", "jeans", "belt", "glasses", "watch", "sweater",
            "scarf", "glove", "gloves", "boot", "boots",
            "pen", "pencil", "paper", "cup", "mug", "glass", "keyboard", "mouse", "phone", "laptop", "computer", "monitor", "screen", "lamp", "book", "books",
            "bottle", "notebook", "charger", "cable", "wallet", "key", "keys", "clock", "remote", "tissue", "plate", "bowl", "fork", "spoon", "knife", "napkin",
            "sky", "tree", "trees", "grass", "cloud", "clouds", "sun", "car", "cars", "street", "road", "garden", "fence", "bush", "bird",
        },
        "places": {
            "england", "scotland", "wales", "ireland", "france", "germany", "spain", "italy", "china", "japan", "india", "brazil", "canada", "mexico",
            "egypt", "kenya", "russia", "america", "usa", "uk", "europe", "asia", "africa", "australia",
            "london", "paris", "berlin", "rome", "madrid", "tokyo", "beijing", "moscow", "newyork", "chicago", "boston", "sydney", "dublin", "edinburgh", "manchester",
        },
        "brands": {
            "google", "apple", "microsoft", "amazon", "facebook", "tesla", "nike", "adidas", "coca", "cola", "pepsi", "mcdonalds", "starbucks",
            "samsung", "sony", "toyota", "ford", "bmw", "gucci", "prada", "disney", "netflix", "spotify", "ikea", "lego",
        },
        "names": set(),  # to be filled together
    }

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
        return [t for t in tokens if t not in prep.STOPWORDS]

    @staticmethod
    def check_category(words: list, number_of_words: int = 5, category: str | None = None) -> bool:
        """True if ≥ number_of_words of the words belong to the same category.

        - category=None          → any category (SI Rule 2)
        - category="environment" → environment objects (SI Rule 1)
        - category="places"/"names"/"brands" with number_of_words=1 → SI Rule 3
        """
        targets = [category] if category else list(prep.CATEGORIES)
        counts = {c: 0 for c in targets}

        for w in words:
            for name in targets:
                if w in prep.CATEGORIES[name]:
                    counts[name] += 1

        return max(counts.values()) >= number_of_words


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

        parts = [self.vectors[p] for p in prep.remove_stopwords(phrase) if p in self.vectors]
        if parts and len({p.shape for p in parts}) == 1:
            return np.mean(np.stack(parts).astype(np.float32), axis=0)
        return None

    def vocab(self):
        return set(self.vectors)
