# GloVe-Word-Embeddings

Library to quickly validate words and get them embedded for academic research. Design is based on Wang et al., 2026 (Nature Human Behaviour) and Olson et al., 2021 (PNAS).

This package has four parts:

1. **prep** — clean and normalize words  
2. **val** — validate against Olson’s list and WordNet nouns  
3. **cat** — assign semantic / proper-noun categories (SI Rules 1–3)  
4. **model** — load embeddings and turn words or phrases into vectors  

Embeddings are hosted on an AWS S3 bucket and downloaded automatically on first use.

## Install

```bash
pip install glove-word-embeddings
```

NLTK WordNet and names data are downloaded automatically on first use of the category or noun helpers (a message appears only when data is actually missing).

## 1. Preprocessing (`prep`)

Clean raw text before validation or embedding.

```python
from glove_word_embeddings import prep

prep.strip_word("  Cat! ")           # "cat"
prep.space_check("cat")              # True
prep.space_check("jar of jam")       # False
prep.remove_stopwords("jar of jam")  # ['jar', 'jam']

prep.clean_word("  The Cat! ")       # "cat"  (strip + drop stopwords)
prep.clean_word("jar of jam")        # "jar jam"
prep.clean_word("the", stopwords=False)  # "the"
```

## 2. Validation (`val`)

Check words against Olson’s validated list and WordNet.

```python
from glove_word_embeddings import val

val.word("telescope")    # True  (in Olson list, single token)
val.word("jar of jam")   # False (fails space check after cleaning)
val.noun("telescope")    # True  (has a WordNet noun synset)
val.noun("quickly")      # False
```

## 3. Categorization (`cat`)

Flag responses that lean too heavily on one semantic group, room objects, or pure proper nouns (SI Rules 1–3).

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

## 4. Embedding (`model`)

Load a vector model and embed single words or short phrases.

```python
import glove_word_embeddings as gwe

gwe.__version__                   # installed package version
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
