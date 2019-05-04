"""
    Converting chunks to trees and tree manipulation
"""
# pylint: disable=C0103

import re
from nltk.corpus import treebank_chunk, treebank
from nltk.tree import Tree

# Sample treebank tree
tb_tree = treebank_chunk.chunked_sents()[0]
print(f"Treebank = {tb_tree}\n")

# Naive way
joined = " ".join([word for word, _ in tb_tree.leaves()])
print(f"Naive join: {joined}\n")

# Using regexp
punct_re = re.compile(r'\s([,\.;\?])')


def chunk_tree_to_sent(tree, concat=' '):
    """
        Converts a chunk tree to a sentence
    """
    sentence = concat.join([word for word, _ in tree.leaves()])
    return re.sub(punct_re, r'\g<1>', sentence)


joined = chunk_tree_to_sent(tb_tree)
print(f"Regexp join: {joined}\n")


# Flattening a deep tree


def flatten_childtrees(trees):
    """
        Flattens all the child trees
    """
    children = []

    for tree in trees:
        # Small tree, only one word -> convert to tuple
        if tree.height() < 3:
            children.extend(tree.pos())
        # Medium tree -> keep all tuples under a same tree
        elif tree.height() == 3:
            children.append(Tree(tree.label(), tree.pos()))
        # Large trees -> recursive call
        else:
            children.extend(flatten_childtrees([child for child in tree]))
    return children


def flatten_deeptree(tree):
    """
        Flattens a deep tree
    """
    return Tree(tree.label(), flatten_childtrees([child for child in tree]))


flat = flatten_deeptree(treebank.parsed_sents()[0])
print(f"Flattened tree : {flat}\n")
flat.draw()


# Shallow trees

def shallow_tree(tree):
    """
        Eliminates all nested subtrees
    """
    children = []
    for branch in tree:
        # Small tree -> convert to tuple
        if branch.height() < 3:
            children.extend(branch.pos())
        # Deep tree -> Keep the tree, convert children to tuples
        else:
            children.append(Tree(branch.label(), branch.pos()))

    return Tree(tree.label(), children)


shallow = shallow_tree(treebank.parsed_sents()[0])
print(f"Shallow tree : {shallow}\n")
shallow.draw()

# Converting tree labels


def convert_tree_labels(tree, tag_map):
    """
        Convert labels on a tree, based on the given mapping
    """
    children = []
    for branch in tree:
        # If it is a tree -> recursive convert the tree
        if isinstance(branch, Tree):
            children.append(convert_tree_labels(branch, tag_map))
        # Simple tuple -> no action required
        else:
            children.append(branch)

    # Make the new tree with the converted children and a (posible) new label
    label = tag_map.get(tree.label(), tree.label())
    return Tree(label, children)


mapping = {
    'NP-SBJ': 'NP',
    'NP-TMP': 'NP'
}

cv_labels = convert_tree_labels(treebank.parsed_sents()[0], mapping)
print(f"Converted labels tree : {cv_labels}\n")
cv_labels.draw()
