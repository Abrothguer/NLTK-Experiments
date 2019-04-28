"""
    Stemming - get the core of the word, not always valid
    Lemmatizing - replaced by synonym, always valid
"""
# pylint: disable=C0103

from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

word = "dying"

# Stemming
# Defalt Porter Stemmer
p_stemmer = PorterStemmer()
stem = p_stemmer.stem(word)
print(f"The porter stem of {word} is {stem}\n")

# Snowball Stemmer - supports different languages
s_stemmer = SnowballStemmer('english')
stem = s_stemmer.stem(word)
print(f"The snowball stem of {word} is {stem}\n")

# Lemmatizing
lemmatizer = WordNetLemmatizer()
lemma = lemmatizer.lemmatize(word)
print(f"Lemma of {word} is {lemma}\n")

lemma = lemmatizer.lemmatize(word, pos='v')
print(f"Lemma(pos: verb) of {word} is {lemma}\n")
