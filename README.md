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

