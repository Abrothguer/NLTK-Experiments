"""
    Transforming chunks/trees
"""
# pylint: disable=C0103

import pickle
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from samples import quote_2, quote_3, quote_4, quote_5, quote_6, quote_7, wrong_1, wrong_2

# Loading tagger
with open("pickles/pos-taggers/brill_tagger.pickle", "rb") as file:
    pos_tagger = pickle.load(file)


def filter_insignificant(chunk, tag_suffixes=['DT', 'CC']):  # pylint: disable = W0102
    """
        Removes insignificant words from the chunk
    """
    good = []

    for word, tag in chunk:
        relevant = True

        for suffix in tag_suffixes:
            if tag.endswith(suffix):
                relevant = False
                break

        if relevant:
            good.append((word, tag))

    return good


# Filtering
filtered = filter_insignificant(pos_tagger.tag(word_tokenize(quote_2)))
print(f"Filtered: {filtered}\n")

# Verb forms
plural_verb_forms = {
    ('is', 'VBZ'): ('are', 'VBP'),
    ('was', 'VBD'): ('were', 'VBD')
}

singular_verb_forms = {
    ('are', 'VBP'): ('is', 'VBZ'),
    ('were', 'VBD'): ('was', 'VBD')
}

# Functions


def tag_startswith(prefix):
    """
        High order - builds a function with the given prefix
    """
    def func(wtuple):
        """
            Identifies words that start with a prefix
        """
        return wtuple[1].startswith(prefix)
    return func


def first_chunk_index(chunk, pred, start=0, step=1):
    """
        Searches the chunk for a word where the function pred returns True
    """
    l = len(chunk)
    end = l if step > 0 else -1

    for i in range(start, end, step):
        if pred(chunk[i]):
            return i
    return None


def correct_verbs(chunk):
    """
        Correct verb forms in a chunk
    """
    vb_index = first_chunk_index(chunk, tag_startswith('VB'))

    # No verbs -> no correction needed
    if vb_index is None:
        return chunk

    verb, vbtag = chunk[vb_index]
    nnpred = tag_startswith('NN')

    # Nearest noun to the right of verb
    nn_index = first_chunk_index(chunk, nnpred, start=vb_index + 1)

    # No noun found to right -> look to the left
    if nn_index is None:
        nn_index = first_chunk_index(chunk, nnpred, start=vb_index - 1, step=-1)

    # No nouns -> no correction needed
    if nn_index is None:
        return chunk

    _, nntag = chunk[nn_index]

    # Get correct verb form and insert into chunk
    if nntag.endswith('S'):
        chunk[vb_index] = plural_verb_forms.get((verb, vbtag), (verb, vbtag))
    else:
        chunk[vb_index] = singular_verb_forms.get((verb, vbtag), (verb, vbtag))
    return chunk


corrected = correct_verbs(pos_tagger.tag(word_tokenize(wrong_1)))
print(f"Corrected = {corrected}\n")
corrected = correct_verbs(pos_tagger.tag(word_tokenize(wrong_2)))
print(f"Corrected = {corrected}\n")

# Swapping


def swap_verb_phrase(chunk):
    """
        Swaps the right side with the left side, using the verb as the pivot point
    """
    def vbpred(wtuple):
        """
            Defining pred function for first_chunk_index
        """
        _, tag = wtuple
        return tag != 'VBG' and tag.startswith('VB') and len(tag) > 2

    vb_index = first_chunk_index(chunk, vbpred)
    if vb_index is None:
        return chunk
    return chunk[vb_index + 1:] + chunk[:vb_index]


swapped = swap_verb_phrase(pos_tagger.tag(word_tokenize(quote_3)))
print(f"Swapping verb phrases: {swapped}\n")


def tag_equals(tag):
    """
        High order - builds a function with the given tag
    """
    def func(wtuple):
        """
            Identifies words that start with a prefix
        """
        return wtuple[1] == tag
    return func


def swap_noun_cardinal(chunk):
    """
        Swaps cardinals, so that they occur always before the noun
    """
    cd_index = first_chunk_index(chunk, tag_equals('CD'))

    # cd_index must be > 0 and there must be a noun immediately before it
    if not cd_index or not chunk[cd_index - 1][1].startswith('NN'):
        return chunk

    noun, nntag = chunk[cd_index - 1]
    chunk[cd_index - 1] = chunk[cd_index]
    chunk[cd_index] = noun, nntag
    return chunk


swapped = swap_noun_cardinal(pos_tagger.tag(word_tokenize(quote_4)))
print(f"Swapping noun cardinals: {swapped}\n")


def swap_infinitive_phrase(chunk):
    """
        Swaps infinitives phrases to a more concise form
    """
    def infpred(wtuple):
        """
            Defining pred function for first_chunk_index
        """
        word, tag = wtuple
        return tag == 'IN' and word != 'like'

    in_index = first_chunk_index(chunk, infpred)
    if in_index is None:
        return chunk

    nn_index = first_chunk_index(chunk, tag_startswith('NN'), start=in_index, step=-1) or 0
    return chunk[:nn_index] + chunk[in_index + 1:] + chunk[nn_index:in_index]


swapped = swap_infinitive_phrase(pos_tagger.tag(word_tokenize(quote_5)))
print(f"Swapping infinitive phrases: {swapped}\n")


def singularize_plural_noun(chunk):
    """
        Depluralize a noun that is followed by other noun
    """
    nns_index = first_chunk_index(chunk, tag_equals('NNS'))
    if nns_index is not None and nns_index + 1 < len(chunk) and chunk[
            nns_index + 1][1][:2] == 'NN':

        noun, nnstag = chunk[nns_index]
        chunk[nns_index] = (noun.rstrip('s'), nnstag.rstrip('S'))

    return chunk


sing = singularize_plural_noun(pos_tag(word_tokenize(quote_6)))
print(f"Singularized plural nouns: {sing}\n")

# Chaining


def transform_chunk(chunk, chain=[filter_insignificant, swap_verb_phrase,  # pylint: disable = W0102
                                  swap_infinitive_phrase, singularize_plural_noun], trace=0):
    """
        Chain all the chunk transformations in one function
    """
    for func in chain:
        chunk = func(chunk)
        if trace:
            print(func.__name__, ':', chunk)

    return chunk


transformed = transform_chunk(pos_tag(word_tokenize(quote_7)), trace=1)
print(f"\nTransformed sentence: {transformed}\n")
