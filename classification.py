"""
    Text classification
"""
# pylint: disable=C0103

import collections
from nltk.corpus import stopwords, movie_reviews
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from samples import quote_8

# Bag of words - word presence feature set from all the words of an instance


def bag_of_words(sentence):
    """
        Creates and returns a bag of words
    """
    return {word: True for word in sentence}


def bag_of_non_stopwords(sentence, language='english'):
    """
        Creates and returns a bag of words without the stop words of the language
    """
    return bag_of_words(set(sentence) - set(stopwords.words(language)))


def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    """
        Creates a bag of words with the 200 most common bigrams.
    """
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)


# Examples of bags of words
naive_bow = bag_of_words(word_tokenize(quote_8))
print(f"Naive bag of words: {naive_bow}\n")

nostp_bow = bag_of_non_stopwords(word_tokenize(quote_8))
print(f"No stopwords bag of words: {nostp_bow}\n")

bigram_bow = bag_of_bigrams_words(word_tokenize(quote_8))
print(f"Bigram bag of words: {bigram_bow}\n")

# Naive Bayes Classifier - Binary
# P(label | features) = P(label) * P(features | label) / P(features)


def label_feats_from_corpus(corp, feature_detector=bag_of_words):
    """
        Takes a corpus and a feature detector and returns {label: [featureset]}
    """
    label_feats = collections.defaultdict(list)

    for label in corp.categories():
        for fileid in corp.fileids(categories=[label]):
            feats = feature_detector(corp.words(fileids=[fileid]))
            label_feats[label].append(feats)
    return label_feats


def split_label_feats(lfeats, split=0.75):
    """
        Splits each list of feature into two sets
    """
    train_feats = []
    test_feats = []

    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])

    return train_feats, test_feats


# Getting corpus, showing label and features, splitting data
print(f"Categories of the corpus: {movie_reviews.categories()}\n")

lbl_feats = label_feats_from_corpus(movie_reviews)
print(f"Labels: {lbl_feats.keys()}\n")

train_data, test_data = split_label_feats(lbl_feats)
print(f"Training size: {len(train_data)}; Test size: {len(test_data)}\n")

# Training the classifier
nb_classifier = NaiveBayesClassifier.train(train_data)

# Most important feature words in the corpus
nb_classifier.show_most_informative_features(n=10)

# Accuracy
acc = accuracy(nb_classifier, test_data)
print(f"Accuracy of Naive Bayes: {acc}\n")

# Taking classification probability
probs = nb_classifier.prob_classify(test_data[1][0])
print(f"Probability of 1st sample: {probs.max()}, with pos={probs.prob('pos'):.3} and " +
      f"neg={probs.prob('neg'):.3}; REAL RESULT = {test_data[1][1]}\n")
