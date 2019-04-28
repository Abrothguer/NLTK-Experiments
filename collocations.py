"""
    Collocations - Words that usually go together in sentences
"""
# pylint: disable=C0103

from nltk.corpus import webtext, stopwords
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

# Words from the script of Monty Python and the Holy Grail.
holy_grail = [wd.lower() for wd in webtext.words('grail.txt')]
bc_finder = BigramCollocationFinder.from_words(holy_grail)

# Naive
collocations = bc_finder.nbest(BigramAssocMeasures.likelihood_ratio, 20)
print(f"Naive top 20 bigram collocations of the holy grail: {collocations}\n")

# Refined
bc_finder.apply_word_filter(lambda wd: len(wd) < 3 or wd in set(stopwords.words('english')))
collocations = bc_finder.nbest(BigramAssocMeasures.likelihood_ratio, 20)
print(f"Refined top 20 bigram collocations of the holy grail: {collocations}\n")

# Trigrams collocations
tc_finder = TrigramCollocationFinder.from_words(holy_grail)
tc_finder.apply_word_filter(lambda wd: len(wd) < 3 or wd in set(stopwords.words('english')))

trigrams = tc_finder.nbest(TrigramAssocMeasures.likelihood_ratio, 20)
print(f"Top 20 trigram collocations of the holy grail: {trigrams}\n")
