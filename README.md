# GloVe-Word-Embeddings

Library to quickly validate words and get them embedded for academic research.

## Install

```bash
pip install glove-word-embeddings
```

## Usage

```python
import glove_word_embeddings as gwe

gwe.list_models()              # {key: filename, ...}

model = gwe.load("glove-6b-300d")   # downloads on first use, caches locally
model.embed("cat")             # -> np.ndarray or None if not in vocab
model.vocab()                  # -> set of all words in the model

gwe.validate("cat")            # -> True / False, checked against the validated word list

gwe.clean_up()                 # deletes all cached files
```

Files are cached in `~/.cache/glove-word-embeddings` (override with `GLOVE_WORD_EMBEDDINGS_CACHE`).
