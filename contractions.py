"""
    Replacement of contractions
"""
# pylint: disable=C0103

import re
from samples import sample_ct

# Replacement patterns
replacement_patts = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'i\'m', 'i am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', r'\g<1> will'),
    (r'(\w+)n\'t', r'\g<1> not'),
    (r'(\w+)\'ve', r'\g<1> have'),
    (r'(\w+)\'s', r'\g<1> is'),
    (r'(\w+)\'re', r'\g<1> are'),
    (r'(\w+)\'d', r'\g<1> would')
]

# Class implementation


class RegexpReplacer(object):  # pylint: disable = R0903
    """
        Implements a Replacer using regular expressions
    """

    def __init__(self, patterns=replacement_patts):  # pylint: disable = W0102
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        """
            Replaces the patterns found on a text
        """
        temp = text
        for (pattern, repl) in self.patterns:
            temp = re.sub(pattern, repl, temp)
        return temp


# Getting the replaced text
regex_rep = RegexpReplacer(replacement_patts)
replaced = regex_rep.replace(sample_ct.lower())
print(f"Text without contractions = {replaced}\n")
