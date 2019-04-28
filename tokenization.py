"""
    Module for tokenization process in nltk
"""
# pylint: disable=C0103

import nltk.data
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import MWETokenizer, TreebankWordTokenizer, TweetTokenizer

# Samples

# Sample text - Excerpt from Do not go gentle into that good nigth by Dylan Thomas
sample = """Do not go gentle into that good night.
            Old age should burn and rave at close of day.
            Rage, rage against the dying of the light.

            Though wise men at their end know dark is right.
            Because their words had forked no lightning they.
            Do not go gentle into that good night."""
sample = " ".join(sample.replace('\n', '').split())

# Sample tweet
sample_tw = """.@SpaceX is now targeting May 1 at 3:59am ET for the next
                cargo launch to the @Space_Station. Onboard will be more
                than 5,500 pounds of @ISS_Research, supplies and hardware
                for crew members living and working on our orbiting outpost.
                Details: https://go.nasa.gov/2GExQpL """
sample_tw = " ".join(sample_tw.replace('\n', '').split())

# Sample contractions
sample_ct = """ Money, get back.
                I'm all right Jack keep your hands off of my stack.
                Money, it's a hit.
                Don't give me that do goody good bullshit.
                I'm in the high-fidelity first class traveling set.
                And I think I need a Lear jet."""
sample_ct = " ".join(sample_ct.replace('\n', '').split())

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
