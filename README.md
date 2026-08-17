# GloVe-Word-Embeddings

Library to quickly validate words and get them embedded for academic research. Design of this package is to reproduce and extend existing, published works: Wang et al., 2026 (Nature Human Behaviour) and Olson et al., 2021 (PNAS).

This package has three parts: **prep (preprocessing)** cleans and validates words, **cat (categorization)** assigns semantic and proper-noun categories, and **model (word embedding)** loads embeddings and turns words or phrases into vectors (embeddings are saved on my AWS S3 bucket and this package will pull from it automatically).

## Install

```bash
pip install glove-word-embeddings
```

This package is based on NLTK WordNet and names. NLTK data are downloaded automatically on first use of the category helpers (you will see a one-time message only when data is actually missing). Another convenience this package brings is I've pickled and saved embeddings on my AWS S3 bucket and this package will pull from it automatically, similar to NLTK data. 

## 1. Preprocessing (prep)

Clean raw text and check it against Olson et al., (2021)’s validated word list.

```python
from glove_word_embeddings import prep

prep.clean_word("  Cat ")            # "cat"
prep.space_check("cat")              # True
prep.space_check("jar of jam")       # False
prep.remove_stopwords("jar of jam")  # ['jar', 'jam']

# Ends with validation against Olson’s single-word list
prep.word_validation("cat")          # True
prep.word_validation("jar of jam")   # False
prep.word_validation("  Cat ")       # True  (cleaned by default)
```

## 2. Categorization (cat)

Flag responses that lean too heavily on one semantic group, room objects, or pure proper nouns.

```python
from glove_word_embeddings import cat

# Ordinary common word? (ignores WordNet proper-noun instances)
cat.check_common("apple")   # True
cat.check_common("nike")    # False
cat.check_common("aaron")   # False

# Which categories does one word belong to?
cat.check("dog")                           # {"animals"}
cat.check("paris", check_common=False)     # {"places"}
cat.check("apple", check_common=True)      # {"fruits"}  (not brands)

# Count across a response
cat.count(words)  # Rule 2: any category ≥ 5
cat.count(words, name="environment")  # Rule 1
cat.count(words, name="places", number_of_words=1, check_common=False)  # Rule 3 places
cat.count(words, name="names",  number_of_words=1, check_common=True)   # Rule 3 names
cat.count(words, name="brands", number_of_words=1, check_common=True)   # Rule 3 brands
```

`check_common=False` trusts the seed list (places, environment, semantic groups).  
`check_common=True` drops a seed hit when the word is an ordinary common word (names / brands).

Unknown `name=` values raise `ValueError`.

## 3. Word Embedding (model)

Load a vector model and embed single words or short phrases.

```python
import glove_word_embeddings as gwe

gwe.list_models()                 # {key: filename, ...}

m = gwe.load("glove-6b-300d")     # downloads on first use, caches locally
m.embed_exact("cat")              # exact match only -> np.ndarray or None
m.vocab_set()                     # -> set of all words in the model

# Multi-word phrases
m.embed_phrase("jar of jam")      # tries "jar of jam" / "jar_of_jam" / "jar-of-jam",
                                  # otherwise averages the non-stopword parts

gwe.clean_up()                    # deletes all cached files
```

Files are cached in `~/.cache/glove-word-embeddings`.

## Citations

Wang et al., (2026) -- Please cite my work, thanks. 
```
Wang, D., Huang, D., Shen, H., & Uzzi, B. (2026). A large-scale comparison of
divergent creativity in humans and large language models. Nature Human
Behaviour, 10(3), 531–540. https://doi.org/10.1038/s41562-025-02331-1
```
Olson et al., (2021) -- Embedding based creativity task.
```
Olson, J. A., Nahas, J., Chmoulevitch, D., Cropper, S. J., & Webb, M. E.
(2021). Naming unrelated words predicts creativity. Proceedings of the
National Academy of Sciences, 118(25), e2022340118.
https://doi.org/10.1073/pnas.2022340118
```
