# GloVe-Word-Embeddings

Library to quickly validate words and get them embedded for academic research.

## Install

```bash
pip install glove-word-embeddings
```

## Preprocessing

Helpers for preparing the words or phrases before embedding.

```python
from glove_word_embeddings import prep

# Check against Olson’s validated single-word list
prep.validate("cat")          # True
prep.validate("jar of jam")   # False  (multi-word phrases always return False)

# Remove common stopwords from a phrase
prep.remove_stopwords("jar of jam")   # ['jar', 'jam']
```

`prep.validate` returns `True` for words that appear in Olson et al. (2021)'s validated word list. 

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

## Citation

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
