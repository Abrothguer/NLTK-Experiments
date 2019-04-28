"""
    Synsets and Wordnet
"""
# pylint: disable=C0103

from nltk.corpus import wordnet

# Getting synsets from a sample word.
word = "balance"
syn = wordnet.synsets(word)
print(f"Synsets of {word} = {syn}\n")

# Getting the name of the synonym and definiton.
syn_name = syn[0].name()
syn_def = syn[0].definition()
print(f"N: {syn_name}, def: {syn_def}\n")

# Hypernyms: generalization, Hyponyms: specification. Up and down a tree.
hypernyms = syn[0].hypernyms()
print(f"Hypernyms = {hypernyms}\n")
hyponyms = hypernyms[0].hyponyms()
print(f"Hyponyms of the first hypernyms = {hyponyms}\n")

# Part of speech. Noun (n), Adjective (a), Adverb (r), Verb(v)
pos = syn[0].pos()
print(f"Our word is a '{pos}'\n")

# Using lemmas to get all synonyms for our word.
synonyms = []
for synset in wordnet.synsets(word):
    for lemma in synset.lemmas():
        synonyms.append(lemma.name())
synonyms = set(synonyms)
print(f"Synonyms for {word} : {synonyms}\n")

# Using lemmas to get antonyms
antonyms = []
for synset in wordnet.synsets(word):
    for lemma in synset.lemmas():
        antonyms += lemma.antonyms()
antonyms = set([antn.name() for antn in antonyms])
print(f"Antonyms for {word} : {antonyms}\n")

# Calculating similarity between words -> proximity in the tree of synsets.
# Wu-Palmer Similarity
wup_simi = hypernyms[0].wup_similarity(syn[0])
print(f"Wup similarity of {hypernyms[0].name()} and {syn[0].name()} = {wup_simi}\n")

# Path similarity
path_simi = hypernyms[0].path_similarity(syn[0])
print(f"Path similarity of {hypernyms[0].name()} and {syn[0].name()} = {path_simi}\n")

# Leacock Chordorow similarity
lch_simi = hypernyms[0].lch_similarity(syn[0])
print(f"LCH similarity of {hypernyms[0].name()} and {syn[0].name()} = {lch_simi}\n")
