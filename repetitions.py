"""
    Removing repeating characters
"""
# pylint: disable=C0103

import re
from nltk.corpus import wordnet

# Class Implementation


class RepeatReplacer(object):  # pylint: disable = R0903
    """
        Replace repeated characters in a word
    """

    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        """
            Does the replacement
        """
        # Check if word exists in the wordnet
        if wordnet.synsets(word):
            return word

        # Replaces
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        return repl_word


# Sample words
wrd_0 = "looooooove"
wrd_1 = "hippopotamus"
wrd_2 = "coordination"
wrd_3 = "uuuuuuuuuuuuh"

# Replacing
rep_replacer = RepeatReplacer()
print(rep_replacer.replace(wrd_0))
print(rep_replacer.replace(wrd_1))
print(rep_replacer.replace(wrd_2))
print(rep_replacer.replace(wrd_3))
