"""
    Tagging POS
"""
# pylint: disable=C0103

import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import treebank, wordnet
from nltk.probability import FreqDist
from nltk.tag.sequential import ClassifierBasedPOSTagger
from nltk.tag import brill, brill_trainer, tnt, SequentialBackoffTagger
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger, AffixTagger
from samples import sample

# Test and training variables
test_sents = treebank.tagged_sents()[3000:]
train_sents = treebank.tagged_sents()[:3000]
tk_sample = word_tokenize(sample)

# Default tagger - Nouns
df_tagger = DefaultTagger('NN')
tagged = df_tagger.tag(tk_sample)
accuracy = df_tagger.evaluate(test_sents)
print(f"Tagged text: {tagged}; acc = {accuracy}\n")

# Unigram tagger
ug_tagger = UnigramTagger(train_sents)
tagged = ug_tagger.tag(tk_sample)
accuracy = ug_tagger.evaluate(test_sents)
print(f"Tagged text: {tagged}; acc = {accuracy}\n")

# Backoff tagger: rely on other tagger(backoff) when the current one does not know how to evaluate
ugb_tagger = UnigramTagger(train_sents, backoff=df_tagger)
accuracy = ugb_tagger.evaluate(test_sents)
print(f"Accuracy of backoff: {accuracy}\n")

# Saving pickle and testing it.
with open('pickles/pos-taggers/unigram_backoff_tagger.pickle', 'wb') as file:
    pickle.dump(ugb_tagger, file)

with open('pickles/pos-taggers/unigram_backoff_tagger.pickle', 'rb') as file:
    pk_tagger = pickle.load(file)

accuracy = pk_tagger.evaluate(test_sents)
print(f"Accuracy of pickled backoff: {accuracy}\n")

# Testing bigram and trigram taggers
bg_tagger = BigramTagger(train_sents)
accuracy = bg_tagger.evaluate(test_sents)
print(f"Accuracy of bigram: {accuracy}\n")

tg_tagger = TrigramTagger(train_sents)
accuracy = tg_tagger.evaluate(test_sents)
print(f"Accuracy of trigram: {accuracy}\n")


def make_backoffs(training, tagger_classes, backoff=None):
    """
        Function for training and make chains of backoff tagger
    """
    # Make a tagger using the previous one as a backoff
    for cls in tagger_classes:
        backoff = cls(training, backoff=backoff)
    return backoff


# Testing the function with all 4 taggers
bc_tagger = make_backoffs(
    train_sents, [UnigramTagger, BigramTagger, TrigramTagger], backoff=df_tagger)
accuracy = bc_tagger.evaluate(test_sents)
print(f"Accuracy of the backoff chain tagger: {accuracy}\n")

# Saving pickle
with open('pickles/pos-taggers/backoff_chain_tagger.pickle', 'wb') as file:
    pickle.dump(bc_tagger, file)

# Affix tagger: context is either the prefix or the suffix
af_tagger = AffixTagger(train_sents)
accuracy = af_tagger.evaluate(test_sents)
print(f"Accuracy of the affix tagger: {accuracy}\n")


# Brill Tagging
def train_brill_tagger(initial_tagger, training, **kwargs):
    """
        Function to train a brill tagger. Uses rules to correct the results of a tagger
    """
    templates = [
        brill.Template(brill.Pos([-1])),
        brill.Template(brill.Pos([1])),
        brill.Template(brill.Pos([-2])),
        brill.Template(brill.Pos([2])),
        brill.Template(brill.Pos([-2, -1])),
        brill.Template(brill.Pos([1, 2])),
        brill.Template(brill.Pos([-3, -2, -1])),
        brill.Template(brill.Pos([1, 2, 3])),
        brill.Template(brill.Pos([-1]), brill.Pos([1])),
        brill.Template(brill.Word([-1])),
        brill.Template(brill.Word([1])),
        brill.Template(brill.Word([-2])),
        brill.Template(brill.Word([2])),
        brill.Template(brill.Word([-2, -1])),
        brill.Template(brill.Word([1, 2])),
        brill.Template(brill.Word([-3, -2, -1])),
        brill.Template(brill.Word([1, 2, 3])),
        brill.Template(brill.Word([-1]), brill.Word([1])),
    ]
    trainer = brill_trainer.BrillTaggerTrainer(initial_tagger, templates, deterministic=True)
    return trainer.train(training, **kwargs)


# Brill tagger using the previous backoff chain tagger
br_tagger = train_brill_tagger(bc_tagger, train_sents)
accuracy = br_tagger.evaluate(test_sents)
print(f"Accuracy of the brill tagger: {accuracy}\n")

# Saving pickle
with open('pickles/pos-taggers/brill_tagger.pickle', 'wb') as file:
    pickle.dump(br_tagger, file)

# TnT tagger with default tagger for unknown words
tnt_tagger = tnt.TnT(unk=df_tagger, Trained=True, N=200)
tnt_tagger.train(train_sents)
accuracy = tnt_tagger.evaluate(test_sents)
print(f"Accuracy of the tnt tagger: {accuracy}\n")

# Saving pickle
with open('pickles/pos-taggers/tnt_tagger.pickle', 'wb') as file:
    pickle.dump(tnt_tagger, file)


# Tagging using the wordnet
class WordNetTagger(SequentialBackoffTagger):
    """
        Class implementation of the wordnet tagger
    """

    def __init__(self, *args, **kwargs):
        SequentialBackoffTagger.__init__(self, *args, **kwargs)
        self.wordnet_tag_map = {
            'n': 'NN',
            's': 'JJ',
            'a': 'JJ',
            'r': 'RB',
            'v': 'VB'
        }
        self.fd = FreqDist(treebank.words())

    def choose_tag(self, tokens, index, history):
        """
            Choses a POS tag based on the wordnet tag
        """

        word = tokens[index]
        for synset in wordnet.synsets(word):
            self.fd[synset.pos()] += 1
        return self.wordnet_tag_map.get(self.fd.max())


# Using the wordnet tagger
wn_tagger = WordNetTagger()
accuracy = wn_tagger.evaluate(test_sents)
print(f"Accuracy of the wordnet tagger: {accuracy}\n")

# Classifier tagging
cl_tagger = ClassifierBasedPOSTagger(train=train_sents)
accuracy = cl_tagger.evaluate(test_sents)
print(f"Accuracy of the classifier tagger: {accuracy}\n")

# Saving pickle - Heavy one
with open('pickles/pos-taggers/classifier_tagger.pickle', 'wb') as file:
    pickle.dump(cl_tagger, file)
