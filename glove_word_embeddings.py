import os
import pickle
import re
import shutil

import numpy as np
import requests
import nltk

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
_names = None
_nltk_ready = False


def list_models():
    return FILES


def clean_up():
    global _validated_words, _names, _nltk_ready
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    _validated_words = None
    _names = None
    _nltk_ready = False


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

    # category -> (WordNet ancestor synset or None, seed words)
    CATEGORIES = {
        "animals": ("animal.n.01", {"dog", "cat", "lion", "tiger", "bear", "elephant", "horse", "cow", "sheep", "goat", "pig", "wolf", "fox", "deer", "rabbit", "monkey", "zebra", "giraffe", "kangaroo", "whale", "dolphin", "shark", "snake", "frog", "goose", "duck", "chicken", "hippo", "rhino", "leopard", "cheetah", "panda", "koala", "otter"}),
        "birds": ("bird.n.01", {"robin", "sparrow", "eagle", "hawk", "owl", "parrot", "penguin", "crow", "pigeon", "swan", "flamingo", "peacock", "finch"}),
        "insects": ("insect.n.01", {"ant", "bee", "wasp", "hornet", "beetle", "fly", "moth", "butterfly", "dragonfly", "cricket", "grasshopper", "ladybug"}),
        "foods": ("food.n.01", {"bread", "cheese", "rice", "pasta", "pizza", "burger", "cake", "soup", "egg", "butter", "sugar", "salt", "chocolate", "cookie", "sandwich", "cereal", "biscuit", "jam", "honey"}),
        "fruits": ("fruit.n.01", {"apple", "banana", "orange", "grape", "pear", "peach", "plum", "cherry", "mango", "melon", "lemon", "lime", "kiwi", "strawberry", "pineapple", "raspberry", "blueberry"}),
        "vegetables": ("vegetable.n.01", {"carrot", "potato", "onion", "pea", "peas", "bean", "beans", "broccoli", "spinach", "lettuce", "cabbage", "cucumber", "pepper", "tomato", "corn", "celery"}),
        "plants": ("plant.n.02", {"tree", "fern", "moss", "rose", "tulip", "daisy", "oak", "pine", "cactus", "ivy", "bamboo", "bush", "shrub", "vine"}),
        "colors": ("color.n.01", {"red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "grey", "gray", "violet", "indigo", "cyan", "magenta", "turquoise", "maroon", "beige"}),
        "body_parts": ("body_part.n.01", {"arm", "leg", "hand", "foot", "head", "eye", "ear", "nose", "mouth", "finger", "toe", "knee", "elbow", "shoulder", "heart", "liver", "lung", "brain", "teeth", "tooth", "hair", "skin"}),
        "metals": ("metal.n.01", {"gold", "silver", "iron", "copper", "bronze", "steel", "tin", "zinc", "lead", "aluminium", "aluminum", "nickel", "platinum"}),
        "planets": (None, {"mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "sun", "moon", "star", "comet", "asteroid", "supernova", "galaxy", "nebula", "meteor"}),
        "sports": (None, {"football", "soccer", "basketball", "tennis", "cricket", "golf", "rugby", "hockey", "baseball", "swimming", "boxing", "cycling", "skiing"}),
        "tools": ("tool.n.01", {"hammer", "screwdriver", "wrench", "drill", "saw", "pliers", "chisel", "axe", "spanner", "clamp"}),
        "countries": ("country.n.02", {"france", "germany", "spain", "italy", "china", "japan", "india", "brazil", "canada", "egypt", "kenya", "peru", "chile", "mexico"}),
        "environment": (None, {
            "desk", "table", "chair", "bed", "sofa", "couch", "shelf", "bookshelf", "stool", "bench", "cabinet", "drawer", "wardrobe", "dresser", "nightstand",
            "wall", "walls", "floor", "ceiling", "door", "window", "roof", "stairs", "carpet", "rug", "curtain", "curtains", "blind", "blinds", "tile", "tiles",
            "corner", "room", "hallway", "fireplace", "radiator", "switch", "socket",
            "shoe", "shoes", "shirt", "tshirt", "sock", "socks", "hat", "cap", "jacket", "coat", "trousers", "pants", "jeans", "belt", "glasses", "watch", "sweater",
            "scarf", "glove", "gloves", "boot", "boots",
            "pen", "pencil", "paper", "cup", "mug", "glass", "keyboard", "mouse", "phone", "laptop", "computer", "monitor", "screen", "lamp", "book", "books",
            "bottle", "notebook", "charger", "cable", "wallet", "key", "keys", "clock", "remote", "tissue", "plate", "bowl", "fork", "spoon", "knife", "napkin",
            "sky", "tree", "trees", "grass", "cloud", "clouds", "sun", "car", "cars", "street", "road", "garden", "fence", "bush", "bird",
        }),
        "places": (None, {
            "england", "scotland", "wales", "ireland", "france", "germany", "spain", "italy", "china", "japan", "india", "brazil", "canada", "mexico",
            "egypt", "kenya", "russia", "america", "usa", "uk", "europe", "asia", "africa", "australia",
            "london", "paris", "berlin", "rome", "madrid", "tokyo", "beijing", "moscow", "newyork", "chicago", "boston", "sydney", "dublin", "edinburgh", "manchester",
        }),
        "brands": (None, {
            "google", "apple", "microsoft", "amazon", "facebook", "tesla", "nike", "adidas", "coca", "cola", "pepsi", "mcdonalds", "starbucks",
            "samsung", "sony", "toyota", "ford", "bmw", "gucci", "prada", "disney", "netflix", "spotify", "ikea", "lego",
        }),
        "names": (None, None),  # filled lazily from nltk.corpus.names
    }

    @staticmethod
    def _ensure_nltk():
        """Download WordNet and names on first use; raise clearly on failure."""
        global _nltk_ready
        if _nltk_ready:
            return
        for resource, path in (("wordnet", "corpora/wordnet"), ("names", "corpora/names")):
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Downloading NLTK '{resource}' data (one-time)…")
                ok = nltk.download(resource, quiet=True)
                if not ok:
                    raise RuntimeError(
                        f"Failed to download NLTK '{resource}' data. "
                        f"Run: python -c \"import nltk; nltk.download('{resource}')\""
                    )
        _nltk_ready = True

    @staticmethod
    def _get_names():
        global _names
        prep._ensure_nltk()
        if _names is None:
            from nltk.corpus import names as nltk_names
            _names = {n.lower() for n in nltk_names.words()}
        return _names

    @staticmethod
    def clean_word(word: str) -> str:
        """Strip and lowercase a word."""
        return word.strip().lower()

    @staticmethod
    def space_check(word: str) -> bool:
        """True if the word is a single token (no spaces, hyphens, underscores)."""
        w = word.strip()
        return not any(c in w for c in (" ", "-", "_"))

    @staticmethod
    def word_validation(word: str, clean_word: bool = True, space_check: bool = True) -> bool:
        """True if the word appears in Olson’s validated single-word list."""
        global _validated_words
        if _validated_words is None:
            _validated_words = load("olson-validated-words")

        w = prep.clean_word(word) if clean_word else word
        if space_check and not prep.space_check(w):
            return False
        return w in _validated_words

    @staticmethod
    def remove_stopwords(phrase: str) -> list[str]:
        """Return the non-stopword tokens of a phrase (lower-cased)."""
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", phrase.lower())
        return [t for t in tokens if t not in prep.STOPWORDS]

    @staticmethod
    def check_category(word: str, check_common: bool = False) -> set[str]:
        """Return the set of categories a single word belongs to.

        Membership is decided by seed list or WordNet hypernym ancestry.
        If check_common=True, a seed hit is kept only when the word has no
        WordNet sense (pure proper noun). Use check_common=True for names/brands
        and check_common=False for places.
        """
        prep._ensure_nltk()
        from nltk.corpus import wordnet as wn

        word = prep.clean_word(word)
        out = set()

        for name, (syn, seeds) in prep.CATEGORIES.items():
            if name == "names":
                seeds = prep._get_names()

            if seeds is not None and word in seeds:
                if check_common and len(wn.synsets(word)) > 0:
                    continue
                out.add(name)
                continue

            if syn is not None:
                cat = wn.synset(syn)
                if any(cat in p for s in wn.synsets(word, pos=wn.NOUN) for p in s.hypernym_paths()):
                    out.add(name)

        return out

    @staticmethod
    def count_categories(
        words: list,
        number_of_words: int = 5,
        category: str | None = None,
        check_common: bool = False,
    ) -> bool:
        """True if ≥ number_of_words of the words belong to the same category.

        - category=None          → any category (SI Rule 2)
        - category="environment" → environment objects (SI Rule 1)
        - category="places" with check_common=False, number_of_words=1 → SI Rule 3 places
        - category="names"/"brands" with check_common=True, number_of_words=1 → SI Rule 3
        """
        if category is not None and category not in prep.CATEGORIES:
            raise ValueError(
                f"Unknown category {category!r}. Valid categories: {sorted(prep.CATEGORIES)}"
            )

        targets = [category] if category else list(prep.CATEGORIES)
        counts = {c: 0 for c in targets}

        for w in words:
            cats = prep.check_category(w, check_common=check_common)
            for name in targets:
                if name in cats:
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
