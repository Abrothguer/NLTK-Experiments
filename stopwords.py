"""
    Filtering stopwords
"""
# pylint: disable=C0103

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from samples import sample, sample_ct

# Loading and showing stopwords from various modules.
en_sw = set(stopwords.words("english"))
pt_sw = set(stopwords.words("portuguese"))
es_sw = set(stopwords.words("spanish"))
fr_sw = set(stopwords.words("french"))

print(f"en stopwords: {en_sw}\n")
print(f"pt stopwords: {pt_sw}\n")
print(f"es stopwords: {es_sw}\n")
print(f"fr stopwords: {fr_sw}\n")

# Showing text withou stopwords.
words = [word for word in word_tokenize(sample.lower()) if word not in en_sw]
print(f"Sample(default) with no stopwords: {words}\n")

# Contractions
words = [word for word in word_tokenize(sample_ct.lower()) if word not in en_sw]
print(f"Sample(contractions) with no stopwords: {words}\n")

# Work around removing 'am 's n't as it is not in the stopwords list

words = [word for word in words if "'" not in word and word.isalpha()]
print(f"Contractions revised: {words}\n")
