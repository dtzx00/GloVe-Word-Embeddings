# GloVe-Word-Embeddings

Library to quickly validate words and get them embedded for academic research.

## Install

```bash
pip install glove-word-embeddings
```

NLTK data (WordNet + names) is downloaded automatically on first use of the category functions.

## Preprocessing

Helpers for preparing words or phrases before embedding.

```python
from glove_word_embeddings import prep

# Normalize a word
prep.clean_word("  Cat ")       # "cat"

# True if single token (no spaces / hyphens / underscores)
prep.space_check("cat")         # True
prep.space_check("jar of jam")  # False

# Look up Olson’s validated single-word list
prep.word_validation("cat")                    # True
prep.word_validation("jar of jam")             # False
prep.word_validation("  Cat ")                 # True  (cleaned by default)

# Remove common stopwords from a phrase
prep.remove_stopwords("jar of jam")            # ['jar', 'jam']

# Which categories does a single word belong to?
prep.check_category("dog")                     # {"animals"}
prep.check_category("paris", check_common=False)   # {"places"}
prep.check_category("apple", check_common=True)    # {"fruits"}  (not brands)

# Count categories across a list of words (SI Rules 1–3)
prep.count_categories(words)                                      # Rule 2 (any category ≥ 5)
prep.count_categories(words, category="environment")              # Rule 1
prep.count_categories(words, category="places", number_of_words=1, check_common=False)  # Rule 3 places
prep.count_categories(words, category="names",  number_of_words=1, check_common=True)   # Rule 3 names
prep.count_categories(words, category="brands", number_of_words=1, check_common=True)   # Rule 3 brands
```

## Usage

```python
import glove_word_embeddings as gwe

gwe.list_models()              # {key: filename, ...}

model = gwe.load("glove-6b-300d")   # downloads on first use, caches locally
model.embed("cat")             # exact match only -> np.ndarray or None
model.vocab()                  # -> set of all words in the model

# Multi-word phrases
model.embed_phrase("jar of jam")   # tries "jar of jam" / "jar_of_jam" / "jar-of-jam",
                                   # otherwise averages the non-stopword parts

gwe.clean_up()                 # deletes all cached files
```

Files are cached in `~/.cache/glove-word-embeddings`.

## Citations

Wang et al., (2026)
```
Wang, D., Huang, D., Shen, H., & Uzzi, B. (2026). A large-scale comparison of
divergent creativity in humans and large language models. Nature Human
Behaviour, 10(3), 531–540. https://doi.org/10.1038/s41562-025-02331-1
```
Olson et al., (2021)
```
Olson, J. A., Nahas, J., Chmoulevitch, D., Cropper, S. J., & Webb, M. E.
(2021). Naming unrelated words predicts creativity. Proceedings of the
National Academy of Sciences, 118(25), e2022340118.
https://doi.org/10.1073/pnas.2022340118
```
