# GloVe-Word-Embeddings

Library to quickly clean, validate, categorise and embed words for academic research.  
Design is based on Wang et al., 2026 (*Nature Human Behaviour*) and Olson et al., 2021 (*PNAS*).

This package has four parts:

1. **pre** — clean, normalise or lemmatize words  
2. **val** — validate against Olson’s list or WordNet nouns  
3. **cat** — classify to proper-noun buckets or WordNet categories
4. **mod** — load embeddings and turn words or phrases into vectors  

Embedding models are hosted on an AWS S3 bucket and downloaded automatically on first use.  
NLTK WordNet and names data are also downloaded automatically on first use of any category helper.

#### Installation

```bash
pip install glove-word-embeddings
```

### Preprocessing (`pre`)

Clean, normalize or lemmatize raw text before validation, classfication or embedding.

```python
from glove_word_embeddings import pre

pre.return_lowercase("  Cat! ")          # "  cat! "
pre.strip_marks("  Cat! ")               # "  Cat  "
pre.strip_stopword("The Cat")            # "Cat"
pre.strip_space("jar of jam")            # "jarofjam"
pre.remove_stopwords("jar of jam")       # ['jar', 'jam']
pre.space_check("cat")                   # True   (single token)
pre.space_check("jar of jam")            # False

# Full clean (defaults: lower + drop stopwords + strip marks + strip space)
pre.clean_word("  The Cat! ")            # "cat"
pre.clean_word("jar of jam")             # "jarjam"
pre.clean_word("jar of jam", strip_space_bool=False)  # "jar jam"
pre.clean_word("the", strip_stopword_bool=False)      # "the"

# WordNet lemma (cached)
pre.lemmatize("dogs")                    # "dog"
pre.lemmatize("running", pos="v")        # "run"
```

### Validation (`val`)

Check words against Olson’s validated list and WordNet.

```python
from glove_word_embeddings import mod, val

val.word("telescope")    # True  (in Olson list, single token)
val.word("jar of jam")   # False (fails space check after cleaning)
val.noun("telescope")    # True  (has a WordNet noun synset)
val.noun("quickly")      # False

m = mod.load("glove-olson-validated")
v = m.vocab_set()
val.vocab("telescope", v)   # True
val.vocab("ice cream", v)   # True if any of "ice cream" / "ice_cream" / "ice-cream" is in v
```


### Categorization (`cat`)

##### Categorize against a bucket of proper nouns

Check if words are in certain semantic group, room objects, or pure proper nouns.

```python
from glove_word_embeddings import cat

# Ordinary common word? (ignores WordNet proper-noun instances)
cat.check_common("apple")   # True
cat.check_common("nike")    # False
cat.check_common("aaron")   # False

# Which curated buckets does a word belong to?
cat.check_bucket("dog")                              # {"animals"}
cat.check_bucket("paris", check_common=False)        # {"places"}
cat.check_bucket("apple", check_common=True)         # {"fruits"}  (not brands)

# Count across a list of words
cat.count_bucket(words)                              # any bucket ≥ 5
cat.count_bucket(words, name="environment")          # environment objects
cat.count_bucket(words, name="places", number_of_words=1, check_common=False)
cat.count_bucket(words, name="names",  number_of_words=1, check_common=True)
cat.count_bucket(words, name="brands", number_of_words=1, check_common=True)
```

`check_common=False` trusts the seed list (places, environment, semantic groups).  
`check_common=True` drops a seed hit when the word is an ordinary common word (names / brands).

##### Categorize against WordNet category chains

Return the hypernym ladder of a word (general → specific). The word itself is **never** included in the list.

```python
from glove_word_embeddings import cat

# Default: longest path of the primary (most frequent) sense
cat.category_chain("dog")
# ['entity', 'physical_entity', 'object', 'whole', 'living_thing',
#  'organism', 'animal', 'chordate', 'vertebrate', 'mammal',
#  'placental', 'carnivore', 'canine']

# Shortest path
cat.category_chain("dog", path="short")
# ['entity', 'physical_entity', 'object', 'whole', 'living_thing',
#  'organism', 'animal', 'domestic_animal']

# Full set of ancestors (union of every path of the primary sense)
cat.category_chain("dog", path="full")
# ['entity', 'physical_entity', 'object', 'whole', 'living_thing',
#  'organism', 'animal', 'chordate', 'domestic_animal', 'vertebrate',
#  'mammal', 'placental', 'carnivore', 'canine']

# Ambiguous words – primary sense only (default)
cat.category_chain("badger", path="full")
# → hierarchy of the “Wisconsin inhabitant” sense

# Ambiguous words – union of *all* senses
cat.category_chain("badger", path="full", sense="union")
# → merges the person hierarchy and the animal hierarchy
```

### Embedding (`mod`)

Load a vector model and embed single words or short phrases.

```python
from glove_word_embeddings import mod

m = mod.load("glove-6b-300d")     # downloads on first use, caches locally
m.embed_exact("cat")              # exact match only → np.ndarray or None
m.vocab_set()                     # → set of all words in the model

# Multi-word phrases
m.embed_phrase("jar of jam")      # tries exact / underscore / hyphen variants,
                                  # otherwise averages the non-stopword parts
```

Files are cached in `~/.cache/glove-word-embeddings`.

### Citations

If you like this package, please cite the relevant works. 

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
