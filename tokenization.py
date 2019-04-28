"""
    Module for tokenization process in nltk
"""
# pylint: disable=C0103

import nltk.data
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import MWETokenizer, TreebankWordTokenizer, TweetTokenizer
from samples import sample, sample_ct, sample_tw

"""
    Tokenization of sentences
"""

# Tokenization of sentence
sample_tok = sent_tokenize(sample)
print(f"Sent tok = {sample_tok}\n")

# Loading tokenizer from pickle
s_tokenizer = nltk.data.load("tokenizers/punkt/PY3/english.pickle")
print(f"Pickle sent tok = {s_tokenizer.tokenize(sample)}\n")

"""
    Tokenization of words
"""

# Tokenization of sentence into words
word_tok = word_tokenize(sample)
print(f"Word tok default = {word_tok}\n")

# Default Treebank word tokenizer
treebank_wt = TreebankWordTokenizer()
print(f"Treebank = {treebank_wt.tokenize(sample)}\n")

# Tweet tokenizer
tweet_tk = TweetTokenizer()
print(f"Tweet tok = {tweet_tk.tokenize(sample_tw)}\n")

# Multi word expression tokenizer
mwe_tk = MWETokenizer()
print(f"MWE tok = {mwe_tk.tokenize(sample.split())}\n")

# Testing contractions
print(f"Word tok contractions = {word_tokenize(sample_ct)}\n")
print(f"Tweet contractions = {tweet_tk.tokenize(sample_ct)}\n")
