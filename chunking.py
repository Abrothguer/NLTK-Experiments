"""
    Chunking: patterns of POS tags that make up a chunk of words
    Chink: what words should not be in a chunk
"""
# pylint: disable=C0103

import pickle
from nltk.corpus import conll2000, treebank_chunk
from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser, ChunkParserI
from nltk.chunk.util import conlltags2tree, tree2conlltags
from nltk.tag import BigramTagger, UnigramTagger, ClassifierBasedTagger
from samples import quote_1

# Samples and loadings

with open("pickles/pos-taggers/brill_tagger.pickle", "rb") as file:
    pos_tagger = pickle.load(file)

tagged_1 = pos_tagger.tag(word_tokenize(quote_1))

# Starts with a determiner, followed by any nouns, followed by an arbritary number of any words
# until another noun is reached. Verbs should be chinked, separating the chunk
chunker_1 = RegexpParser(r"""
NP:
    {<DT><NN.*><.*>*<NN.*>}
    }<VB.*>{ """)

# Creating the chunks
chunks = chunker_1.parse(tagged_1)
print(f"Chunks: {chunks}")
chunks.draw()

# Chunking with merge and split rules.
# Line 1 - Chunk rule: grab from a determiner to a noun.
# Line 2 - Split rule: split between a noun and anything else.
# Line 3 - Split rule: split between anything and a determiner.
# Line 4 - Merge rule: merge nouns together
chunker_2 = RegexpParser(r"""
NP:
    {<DT><.*>*<NN.*>}
    <NN.*>}{<.*>
    <.*>}{<DT>
    <NN.*>{}<NN.*> """)

chunks = chunker_2.parse(tagged_1)
print(f"Chunks: {chunks}")
chunks.draw()

# Making a chunker and testing it's accuracy
# 1.1: Chunk optional determiner with nouns
# 1.2: Merge adjective with noun chunk
# 2.1: Chunk preposition
# 3.1: Chunk optional modal with verb
chunker = RegexpParser(r'''
NP:
    {<DT>?<NN.*>+}
    <JJ>{}<NN.*>
PP:
    {<IN>}
VP:
    {<MD>?<VB.*>}
''')
score = chunker.evaluate(conll2000.chunked_sents())
print(f"Accuracy of regex chunker: {score.accuracy()}")

# Tagger-based chunker


def conll_tag_chunks(chunk_sents):
    """
        Extracts a list of tuples (pos, iob) from a list of trees.
    """
    tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
    return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]


def make_backoffs(training, tagger_classes, backoff=None):
    """
        Function for training and make chains of backoff tagger
    """
    # Make a tagger using the previous one as a backoff
    for cls in tagger_classes:
        backoff = cls(training, backoff=backoff)
    return backoff


class TagChunker(ChunkParserI):  # pylint: disable = W0223
    """
        Class implementation of the tag-based chunker
    """

    def __init__(self, train_chunks, tagger=None, tagger_classes=[UnigramTagger, BigramTagger]):  # pylint: disable=W0102
        train_sents = conll_tag_chunks(train_chunks)
        if tagger is None:
            self.tagger = make_backoffs(train_sents, tagger_classes)
        else:
            self.tagger = tagger

    def parse(self, tokens):
        """
            Parse sentence to chunks
        """
        if not tokens:
            return None
        (words, tags) = zip(*tokens)
        gen_chunks = self.tagger.tag(tags)
        wtc = zip(words, gen_chunks)
        return conlltags2tree([(w, t, c) for (w, (t, c)) in wtc])


# Separating data and getting chunker accuracy
train_ck = treebank_chunk.chunked_sents()[:3000]
test_ck = treebank_chunk.chunked_sents()[3000:]
train_conll = conll2000.chunked_sents("train.txt")
test_conll = conll2000.chunked_sents("test.txt")


# With unigram and bigram taggers
chunker = TagChunker(train_ck)
score = chunker.evaluate(test_ck)
print(f"Accuracy of tag chunker on treebank: {score.accuracy()}")

# Saving pickle
with open('pickles/chunkers/tag_chunker_treebank.pickle', 'wb') as file:
    pickle.dump(chunker, file)

chunker = TagChunker(train_conll)
score = chunker.evaluate(test_conll)
print(f"Accuracy of tag chunker on conll2000: {score.accuracy()}")

# Saving pickle
with open('pickles/chunkers/tag_chunker_conll2000.pickle', 'wb') as file:
    pickle.dump(chunker, file)

# Classification-based chunking


def chunk_trees2train_chunks(chunk_sents):
    """
        Convert tuples (word, pos, iob) ot ((word, pos), iob)
    """
    tag_sents = [tree2conlltags(sent) for sent in chunk_sents]
    return [[((w, t), c) for (w, t, c) in sent] for sent in tag_sents]


def prev_next_pos_iob(tokens, index, history):
    """
        Feature detector function for the classifier
    """
    word, pos = tokens[index]
    if index == 0:
        prevword, prevpos, previob = ('<START>',) * 3
    else:
        prevword, prevpos = tokens[index - 1]
        previob = history[index - 1]
    if index == len(tokens) - 1:
        nextword, nextpos = ('<END>',) * 2
    else:
        nextword, nextpos = tokens[index + 1]
    feats = {
        'word': word,
        'pos': pos,
        'nextword': nextword,
        'nextpos': nextpos,
        'prevword': prevword,
        'prevpos': prevpos,
        'previob': previob
    }
    return feats


class ClassifierChunker(ChunkParserI):  # pylint: disable = W0223
    """
        Classifier-based chunker class implementation
    """

    def __init__(self, train_sents, feature_detector, **kwargs):
        train_chunks = chunk_trees2train_chunks(train_sents)
        self.tagger = ClassifierBasedTagger(
            train=train_chunks, feature_detector=feature_detector, **kwargs)

    def parse(self, tokens):
        """
            Parse sentence into chunks
        """
        if not tokens:
            return None
        chunked = self.tagger.tag(tokens)
        return conlltags2tree([(w, t, c) for ((w, t), c) in chunked])


# Testing accuracy with treebank
cl_chunker = ClassifierChunker(train_ck, prev_next_pos_iob)
score = cl_chunker.evaluate(test_ck)
print(f"Accuracy of classifier chunker on treebank: {score.accuracy()}")

# Saving pickle
with open('pickles/chunkers/classifier_chunker_treebank.pickle', 'wb') as file:
    pickle.dump(cl_chunker, file)

# Testing accuracy with conll2000
cl_chunker = ClassifierChunker(train_conll, prev_next_pos_iob)
score = cl_chunker.evaluate(test_conll)
print(f"Accuracy of classifier chunker on conll2000: {score.accuracy()}")

# Saving pickle
with open('pickles/chunkers/classifier_chunker_conll2000.pickle', 'wb') as file:
    pickle.dump(cl_chunker, file)
