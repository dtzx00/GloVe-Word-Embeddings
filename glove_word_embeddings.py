from __future__ import annotations
import nltk,requests,os,pickle,re,shutil,numpy as np
from importlib.metadata import PackageNotFoundError, version
try:
    __version__ = version("glove-word-embeddings")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

BASE_URL = "https://embedding-files-open-access.s3.us-east-1.amazonaws.com"
CACHE_DIR = os.environ.get(
    "GLOVE_WORD_EMBEDDINGS_CACHE") or os.path.expanduser(
    "~/.cache/glove-word-embeddings")

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

_valid_words = None
_nltk_names = None
_nltk_ready = False


def list_models():
    return FILES


def clean_up():
    global _valid_words, _nltk_names, _nltk_ready
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    _valid_words = None
    _nltk_names = None
    _nltk_ready = False


# --- 1. pre: clean / normalize only ----------------------------------------

class pre:
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "nor", "of", "to", "in", "on",
        "for", "with", "at", "by", "from", "as", "into", "onto", "upon", "over",
        "under", "about", "between", "through", "is", "are", "was", "were", "ah",
        "be", "been", "being", "this", "that", "these", "those", "it", "its"}

    @staticmethod
    def check_lowercase(word: str) -> bool:
        """True if the word has no uppercase characters."""
        return word == word.lower()

    @staticmethod
    def return_lowercase(word: str) -> str:
        """Return word in lowercase."""
        return word.lower()

    @staticmethod
    def check_stopword(word: str) -> bool:
        """True if the word / phrase contains a stopword."""
        return not (word == pre.strip_stopword(word))

    @staticmethod
    def strip_stopword(word: str) -> str:
        """Return the phrase with stopwords removed (as joined string)."""
        tokens = (word or "").split()
        return " ".join(w for w in tokens if w.lower() not in pre.STOPWORDS)

    @staticmethod
    def remove_stopwords(phrase: str) -> list[str]:
        """Return list of non-stopword tokens. Matches README and embed_phrase usage."""
        if not phrase:
            return []
        return [w for w in phrase.split() if w.lower() not in pre.STOPWORDS]

    @staticmethod
    def check_marks(word: str) -> bool:
        """True if the word has punctuations, ignore spacing."""
        return not (word == pre.strip_marks(word))

    @staticmethod
    def strip_marks(word: str) -> str:
        """Replace punctuation with space; keep letters, digits, spaces."""
        return re.sub(r"[^\w\s]", " ", word)

    @staticmethod
    def check_space(word: str) -> bool:
        """True if the word has spacing (whitespace / - / _)."""
        return bool(re.search(r"[\s\-_]+", word or ""))

    @staticmethod
    def space_check(word: str) -> bool:
        """True if the word is a single token with no spacing. Matches README: cat→True, 'jar of jam'→False."""
        return not pre.check_space(word)

    @staticmethod
    def strip_space(word: str) -> str:
        """Return word without spacing. Punctuation unchanged."""
        return re.sub(r"[\s\-_]+", "", word or "")

    @staticmethod
    def clean_word(word: str,
                   return_lowercase_bool: bool = True,
                   strip_stopword_bool: bool = True,
                   strip_marks_bool: bool = True,
                   strip_space_bool: bool = True) -> str:
        """Normalize a word or phrase. Optionally drop stopwords, punctuation, spacing."""
        if word:
            if return_lowercase_bool:
                word = pre.return_lowercase(word)
            if strip_stopword_bool:
                word = pre.strip_stopword(word)
            if strip_marks_bool:
                word = pre.strip_marks(word)
            if strip_space_bool:
                word = pre.strip_space(word)
            return word
        else:
            return None


# --- 2. val: validate words ------------------------------------------------

class val:
    @staticmethod
    def word(word: str, clean: bool = True, space_check: bool = True) -> bool:
        """Return True if the word appears in Olson’s validated word list."""
        global _valid_words
        if _valid_words is None:
            _valid_words = mod.load("olson-validated-words")

        w = pre.clean_word(word) if clean else word
        if space_check and not pre.space_check(w):
            return False
        return w in _valid_words

    @staticmethod
    def noun(word: str) -> bool:
        """Return True if the word has at least one noun synset in WordNet."""
        cat._ensure_nltk()
        from nltk.corpus import wordnet as wn

        w = pre.clean_word(word)
        if not w:
            return False
        return any(s.pos() == "n" for s in wn.synsets(w))
    
    @staticmethod
    def vocab(word: str, vocab: set) -> bool:
        """Return True if a compound word, e.g., ice-cream, is in the embedding vocab."""
        w = pre.clean_word(word)
        if not w:
            return False
        for cand in (w, w.replace(" ", "_"), w.replace(" ", "-"), w.replace("-", "_")):
            if cand in vocab:
                return True
        return False    


# --- 3. cat: semantic / proper-noun category checks ------------------------

class cat:
    # name -> (WordNet ancestor synset or None, seed words)
    CATEGORIES = {
        "animals": ("animal.n.01", {
            "dog", "cat", "lion", "tiger", "bear", "elephant", "horse",
            "cow", "sheep", "goat", "pig", "wolf", "fox", "deer", "rabbit",
            "monkey", "zebra", "giraffe", "kangaroo", "whale", "dolphin",
            "shark", "snake", "frog", "goose", "duck", "chicken", "hippo",
            "rhino", "leopard", "cheetah", "panda", "koala", "otter",
            "mouse", "rat", "bat", "squirrel", "raccoon", "skunk", "moose",
            "buffalo", "camel", "donkey", "mule", "llama", "alpaca",
            "crocodile", "alligator", "turtle", "tortoise", "lizard",
            "iguana", "gecko", "salamander", "toad", "seal", "walrus",
            "beaver", "porcupine", "hedgehog", "hamster", "guinea",
            "pony", "foal", "calf", "lamb", "puppy", "kitten", "cub",
        }),
        "birds": ("bird.n.01", {
            "robin", "sparrow", "eagle", "hawk", "owl", "parrot", "penguin",
            "crow", "pigeon", "swan", "flamingo", "peacock", "finch",
            "dove", "raven", "magpie", "jay", "cardinal", "bluebird",
            "canary", "parakeet", "cockatoo", "macaw", "toucan", "heron",
            "crane", "stork", "pelican", "seagull", "albatross", "duck",
            "goose", "turkey", "chicken", "rooster", "hen", "chick",
            "ostrich", "emu", "kiwi", "woodpecker", "hummingbird",
            "nightingale", "lark", "swallow", "starling",
        }),
        "insects": ("insect.n.01", {
            "ant", "bee", "wasp", "hornet", "beetle", "fly", "moth",
            "butterfly", "dragonfly", "cricket", "grasshopper", "ladybug",
            "mosquito", "gnat", "flea", "tick", "spider", "scorpion",
            "centipede", "millipede", "termite", "cockroach", "roach",
            "locust", "cicada", "mantis", "firefly", "glowworm",
            "caterpillar", "larva", "grub", "aphid", "weevil", "earwig",
        }),
        "foods": ("food.n.01", {
            "bread", "cheese", "rice", "pasta", "pizza", "burger", "cake",
            "soup", "egg", "butter", "sugar", "salt", "chocolate", "cookie",
            "sandwich", "cereal", "biscuit", "jam", "honey", "meat", "fish",
            "chicken", "beef", "pork", "bacon", "sausage", "steak", "ham",
            "noodles", "salad", "sauce", "yogurt", "yoghurt", "milk",
            "cream", "ice-cream", "pie", "pudding", "donut", "doughnut",
            "waffle", "pancake", "toast", "bagel", "muffin", "cracker",
            "chip", "chips", "fries", "noodle", "dumpling", "sushi",
            "taco", "burrito", "nacho", "ketchup", "mustard", "mayo",
            "mayonnaise", "vinegar", "oil", "flour", "spice", "pepper",
        }),
        "fruits": ("fruit.n.01", {
            "apple", "banana", "orange", "grape", "pear", "peach", "plum",
            "cherry", "mango", "melon", "lemon", "lime", "kiwi",
            "strawberry", "pineapple", "raspberry", "blueberry",
            "watermelon", "cantaloupe", "honeydew", "papaya", "guava",
            "fig", "date", "coconut", "avocado", "pomegranate", "apricot",
            "nectarine", "tangerine", "clementine", "grapefruit",
            "blackberry", "cranberry", "gooseberry", "currant", "lychee",
            "passionfruit", "dragonfruit", "starfruit", "persimmon",
        }),
        "vegetables": ("vegetable.n.01", {
            "carrot", "potato", "onion", "pea", "peas", "bean", "beans",
            "broccoli", "spinach", "lettuce", "cabbage", "cucumber",
            "pepper", "tomato", "corn", "celery", "garlic", "ginger",
            "mushroom", "zucchini", "courgette", "eggplant", "aubergine",
            "pumpkin", "squash", "radish", "beet", "beetroot", "turnip",
            "parsnip", "leek", "shallot", "scallion", "asparagus",
            "artichoke", "cauliflower", "kale", "chard", "okra", "yam",
            "sweetpotato", "lentil", "chickpea", "tofu",
        }),
        "plants": ("plant.n.02", {
            "tree", "fern", "moss", "rose", "tulip", "daisy", "oak", "pine",
            "cactus", "ivy", "bamboo", "bush", "shrub", "vine", "grass",
            "flower", "leaf", "leaves", "branch", "root", "seed", "weed",
            "willow", "maple", "birch", "cedar", "fir", "spruce", "palm",
            "lily", "orchid", "sunflower", "lavender", "mint", "basil",
            "thyme", "rosemary", "sage", "parsley", "cilantro", "herb",
            "algae", "seaweed", "mushroom", "fungus", "clover", "dandelion",
        }),
        "colors": ("color.n.01", {
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "brown", "black", "white", "grey", "gray", "violet", "indigo",
            "cyan", "magenta", "turquoise", "maroon", "beige", "gold",
            "silver", "navy", "teal", "olive", "lime", "coral", "salmon",
            "crimson", "scarlet", "amber", "ivory", "cream", "tan", "khaki",
            "lavender", "lilac", "peach", "mint", "aqua", "azure", "charcoal",
        }),
        "body_parts": ("body_part.n.01", {
            "arm", "leg", "hand", "foot", "head", "eye", "ear", "nose",
            "mouth", "finger", "toe", "knee", "elbow", "shoulder", "heart",
            "liver", "lung", "brain", "teeth", "tooth", "hair", "skin",
            "neck", "back", "chest", "stomach", "belly", "hip", "thigh",
            "ankle", "wrist", "palm", "thumb", "nail", "tongue", "lip",
            "cheek", "chin", "forehead", "eyebrow", "eyelash", "throat",
            "bone", "muscle", "blood", "vein", "nerve", "spine", "rib",
            "kidney", "intestine", "bladder", "skull", "jaw",
        }),
        "metals": ("metal.n.01", {
            "gold", "silver", "iron", "copper", "bronze", "steel", "tin",
            "zinc", "lead", "aluminium", "aluminum", "nickel", "platinum",
            "titanium", "brass", "chrome", "chromium", "cobalt", "mercury",
            "tungsten", "uranium", "magnesium", "lithium", "sodium",
            "potassium", "calcium", "alloy", "metal", "ore", "rust",
        }),
        "planets": (None, {
            "mercury", "venus", "earth", "mars", "jupiter", "saturn",
            "uranus", "neptune", "pluto", "sun", "moon", "star", "comet",
            "asteroid", "supernova", "galaxy", "nebula", "meteor",
            "planet", "satellite", "orbit", "cosmos", "universe", "space",
            "constellation", "quasar", "pulsar", "blackhole", "meteorite",
            "eclipse", "solstice", "equinox", "aurora",
        }),
        "sports": (None, {
            "football", "soccer", "basketball", "tennis", "cricket", "golf",
            "rugby", "hockey", "baseball", "swimming", "boxing", "cycling",
            "skiing", "volleyball", "badminton", "squash", "wrestling",
            "karate", "judo", "taekwondo", "fencing", "archery", "rowing",
            "sailing", "surfing", "skating", "snowboarding", "climbing",
            "running", "jogging", "marathon", "sprint", "gymnastics",
            "diving", "polo", "lacrosse", "softball", "handball", "bowling",
            "billiards", "pool", "darts", "chess", "esports",
        }),
        "tools": ("tool.n.01", {
            "hammer", "screwdriver", "wrench", "drill", "saw", "pliers",
            "chisel", "axe", "spanner", "clamp", "nail", "screw", "bolt",
            "nut", "washer", "tape", "ruler", "knife", "scissors", "glue",
            "file", "rasp", "mallet", "sledgehammer", "crowbar", "pickaxe",
            "shovel", "rake", "hoe", "trowel", "brush", "sandpaper",
        }),
        "countries": ("country.n.02", {
            "france", "germany", "spain", "italy", "china", "japan", "india",
            "brazil", "canada", "egypt", "kenya", "peru", "chile", "mexico",
            "usa", "america", "england", "britain", "uk", "russia",
            "australia", "argentina", "colombia", "venezuela", "cuba",
            "nigeria", "ghana", "ethiopia", "morocco", "algeria",
            "turkey", "iran", "iraq", "israel", "saudi", "korea",
            "vietnam", "thailand", "indonesia", "malaysia", "philippines",
            "portugal", "greece", "poland", "sweden", "norway", "denmark",
            "finland", "ireland", "scotland", "wales", "switzerland",
            "austria", "belgium", "netherlands", "holland",
        }),
        "environment": (None, {
            "desk", "table", "chair", "bed", "sofa", "couch", "shelf",
            "bookshelf", "stool", "bench", "cabinet", "drawer", "wardrobe",
            "dresser", "nightstand", "wall", "walls", "floor", "ceiling",
            "door", "window", "roof", "stairs", "carpet", "rug", "curtain",
            "curtains", "blind", "blinds", "tile", "tiles", "corner", "room",
            "hallway", "fireplace", "radiator", "switch", "socket",
            "shoe", "shoes", "shirt", "tshirt", "sock", "socks", "hat",
            "cap", "jacket", "coat", "trousers", "pants", "jeans", "belt",
            "glasses", "watch", "sweater", "scarf", "glove", "gloves",
            "boot", "boots", "pen", "pencil", "paper", "cup", "mug", "glass",
            "keyboard", "mouse", "phone", "laptop", "computer", "monitor",
            "screen", "lamp", "book", "books", "bottle", "notebook",
            "charger", "cable", "wallet", "key", "keys", "clock", "remote",
            "tissue", "plate", "bowl", "fork", "spoon", "knife", "napkin",
            "sky", "tree", "trees", "grass", "cloud", "clouds", "sun",
            "car", "cars", "street", "road", "garden", "fence", "bush",
            "bird", "pillow", "blanket", "mirror", "picture", "frame",
            "vase", "plant", "bin", "trash", "bag", "box", "umbrella",
        }),
        "places": (None, {
            "england", "scotland", "wales", "ireland", "france", "germany",
            "spain", "italy", "china", "japan", "india", "brazil", "canada",
            "mexico", "egypt", "kenya", "russia", "america", "usa", "uk",
            "europe", "asia", "africa", "australia", "london", "paris",
            "berlin", "rome", "madrid", "tokyo", "beijing", "moscow",
            "newyork", "chicago", "boston", "sydney", "dublin", "edinburgh",
            "manchester", "losangeles", "sanfrancisco", "seattle", "miami",
            "toronto", "vancouver", "montreal", "mumbai", "delhi", "shanghai",
            "hongkong", "singapore", "dubai", "cairo", "lagos", "nairobi",
            "athens", "vienna", "prague", "amsterdam", "brussels", "lisbon",
            "stockholm", "oslo", "copenhagen", "helsinki", "warsaw",
        }),
        "brands": (None, {
            "google", "apple", "microsoft", "amazon", "facebook", "tesla",
            "nike", "adidas", "coca", "cola", "pepsi", "mcdonalds",
            "starbucks", "samsung", "sony", "toyota", "ford", "bmw",
            "gucci", "prada", "disney", "netflix", "spotify", "ikea", "lego",
            "intel", "nvidia", "ibm", "oracle", "cisco", "uber", "lyft",
            "airbnb", "twitter", "instagram", "whatsapp", "youtube",
            "walmart", "target", "costco", "cvs", "walgreens", "shell",
            "exxon", "bp", "chevron", "visa", "mastercard", "paypal",
            "honda", "nissan", "hyundai", "volkswagen", "audi", "mercedes",
            "chanel", "lv", "louisvuitton", "hermes", "rolex", "cartier",
        }),
        "names": (None, None),  # filled lazily from nltk.corpus.names
    }

    @staticmethod
    def _ensure_nltk():
        """Download WordNet and names on first use; raise clearly on failure.

        Uses a functional check (can we look up a word?) instead of path probing,
        because WordNet is often stored as corpora/wordnet.zip and never unpacked.
        """
        global _nltk_ready
        if _nltk_ready:
            return

        from nltk.corpus import wordnet, names

        try:
            wordnet.synsets("dog")
        except LookupError:
            print("Downloading NLTK 'wordnet' data (one-time)…")
            ok = nltk.download("wordnet", quiet=True)
            if not ok:
                raise RuntimeError(
                    "Failed to download NLTK 'wordnet' data. "
                    'Run: python -c "import nltk; nltk.download(\'wordnet\')"'
                )
            try:
                wordnet.synsets("dog")
            except LookupError as e:
                raise RuntimeError(
                    "NLTK 'wordnet' data is still unavailable after download."
                ) from e

        try:
            names.words()
        except LookupError:
            print("Downloading NLTK 'names' data (one-time)…")
            ok = nltk.download("names", quiet=True)
            if not ok:
                raise RuntimeError(
                    "Failed to download NLTK 'names' data. "
                    'Run: python -c "import nltk; nltk.download(\'names\')"'
                )
            try:
                names.words()
            except LookupError as e:
                raise RuntimeError(
                    "NLTK 'names' data is still unavailable after download."
                ) from e

        _nltk_ready = True

    @staticmethod
    def _get_nltk_names():
        global _nltk_names
        cat._ensure_nltk()
        if _nltk_names is None:
            from nltk.corpus import names as nltk_names
            _nltk_names = {n.lower() for n in nltk_names.words()}
        return _nltk_names

    @staticmethod
    def check_common(word: str, n_senses: int = 2) -> bool:
        """True if the word has at least n_senses common-noun senses in WordNet.

        Proper nouns (Aaron, London, Nike) are stored in WordNet as 'instance'
        synsets, so they are not counted. A word only counts as common if it has
        enough ordinary noun senses of its own — this is what separates 'apple'
        (the fruit) from 'nike' (only ever a name).
        """
        cat._ensure_nltk()
        from nltk.corpus import wordnet as wn

        word = pre.clean_word(word)
        common = [s for s in wn.synsets(word, pos=wn.NOUN) if not s.instance_hypernyms()]
        return len(common) >= n_senses

    @staticmethod
    def check(word: str, check_common: bool = False, n_senses: int = 2) -> set[str]:
        """Return the set of category names a single word belongs to.

        Membership is decided by seed list or WordNet hypernym ancestry.
        If check_common=True, a seed hit is dropped when the word is an ordinary
        common word — so 'apple' is not flagged as a brand, while 'nike' is.
        """
        cat._ensure_nltk()
        from nltk.corpus import wordnet as wn

        word = pre.clean_word(word)
        out = set()

        for name, (syn, seeds) in cat.CATEGORIES.items():
            if name == "names":
                seeds = cat._get_nltk_names()

            if seeds is not None and word in seeds:
                if not (check_common and cat.check_common(word, n_senses)):
                    out.add(name)
                    continue
                # common word: fall through to the WordNet check below

            if syn is not None:
                ancestor = wn.synset(syn)
                if any(
                    ancestor in path
                    for s in wn.synsets(word, pos=wn.NOUN)
                    for path in s.hypernym_paths()
                ):
                    out.add(name)

        return out

    @staticmethod
    def count(
        words: list,
        number_of_words: int = 5,
        name: str | None = None,
        check_common: bool = False,
        n_senses: int = 2,
    ) -> bool:
        """True if ≥ number_of_words of the words belong to the same category.

        - name=None          → any category (SI Rule 2)
        - name="environment" → environment objects (SI Rule 1)
        - name="places" with check_common=False, number_of_words=1 → SI Rule 3 places
        - name="names"/"brands" with check_common=True, number_of_words=1 → SI Rule 3
        """
        if name is not None and name not in cat.CATEGORIES:
            raise ValueError(
                f"Unknown category {name!r}. Valid names: {sorted(cat.CATEGORIES)}"
            )

        targets = [name] if name else list(cat.CATEGORIES)
        counts = {c: 0 for c in targets}

        for w in words:
            found = cat.check(w, check_common=check_common, n_senses=n_senses)
            for key in targets:
                if key in found:
                    counts[key] += 1

        return max(counts.values()) >= number_of_words


# --- 4. mod: word embeddings -----------------------------------------------

class mod:
    def __init__(self, vectors: dict):
        self.vectors = vectors

    @staticmethod
    def load(key: str, force_download: bool = False):
        """Download `key` if not cached, then return a mod (or word set for the wordlist)."""
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
                return mod(pickle.load(f))
        else:
            with open(path, "r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}

    def embed_exact(self, word: str):
        """Embed using exact match only."""
        return self.vectors.get(word)

    def embed_phrase(self, phrase: str):
        """Embed by trying space/_/- variants, otherwise
        average embeddings of non-stopword parts."""
        phrase = phrase.strip().lower()
        if not phrase:
            return None

        for variant in (
            phrase,
            phrase.replace(" ", "_"),
            phrase.replace(" ", "-"),
        ):
            if variant in self.vectors:
                return self.vectors[variant]

        parts = [self.vectors[p] for p in pre.remove_stopwords(phrase) if p in self.vectors]
        if parts and len({p.shape for p in parts}) == 1:
            return np.mean(np.stack(parts).astype(np.float32), axis=0)
        return None

    def vocab_set(self):
        """Full set of vocabulary in the embedding."""
        return set(self.vectors)
