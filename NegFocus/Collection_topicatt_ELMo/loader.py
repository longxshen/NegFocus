#!/bin/python
# encoding: utf-8
from __future__ import print_function, division
import os
import re
import codecs
import unicodedata
from utils import create_dico, create_mapping
import model
import string
import random
import numpy as np


def load_sentences(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = line.strip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    return sentences

def pos_mapping(sentences):
    """
    Create a dictionary and a mapping of POS labels, sorted by frequency.
    """
    pos = [[x[1] for x in s] for s in sentences]
    dico = create_dico(pos)
    dico['<UNK>'] = 10000000
    pos_to_id, id_to_pos = create_mapping(dico)
    print("Found %i unique pos (%i in total)" % (
        len(dico), sum(len(x) for x in pos)
    ))
    return dico, pos_to_id, id_to_pos

def conNode_mapping(sentences):
    """
    Create a dictionary and a mapping of chunk labels, sorted by frequency.
    """
    conNode = [[x[2] for x in s] for s in sentences]
    dico = create_dico(conNode)
    dico['<UNK>'] = 10000000
    conNode_to_id, id_to_conNode = create_mapping(dico)
    print("Found %i unique conNode (%i in total)" % (
        len(dico), sum(len(x) for x in conNode)
    ))
    return dico, conNode_to_id, id_to_conNode

def depNode_mapping(sentences):
    """
    Create a dictionary and a mapping of dependency node labels, sorted by frequency.
    """
    depNode = [[x[3] for x in s] for s in sentences]
    dico = create_dico(depNode)
    dico['<UNK>'] = 10000000
    depNode_to_id, id_to_depNode = create_mapping(dico)
    print("Found %i unique depNode (%i in total)" % (
        len(dico), sum(len(x) for x in depNode)
    ))
    return dico, depNode_to_id, id_to_depNode

def semroles_mapping(sentences):
    """
    Create a dictionary and a mapping of semantic roles labels, sorted by frequency.
    """
    semroles = [[x[4] for x in s] for s in sentences]
    dico = create_dico(semroles)
    dico['<UNK>'] = 10000000
    semroles_to_id, id_to_semroles = create_mapping(dico)
    print("Found %i unique semroles (%i in total)" % (
        len(dico), sum(len(x) for x in semroles)
    ))
    return dico, semroles_to_id, id_to_semroles

def loc_mapping(sentences):
    """
    Create a dictionary and a mapping of location labels, sorted by frequency.
    """
    loc = [[x[6] for x in s] for s in sentences]
    dico = create_dico(loc)
    dico['<UNK>'] = 10000000
    loc_to_id, id_to_loc = create_mapping(dico)
    print("Found %i unique location (%i in total)" % (
        len(dico), sum(len(x) for x in loc)
    ))
    return dico, loc_to_id, id_to_loc

def word_mapping(sentences, lower=True):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)

    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word

def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico[model.START_TAG] = -1
    dico[model.STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique focus tags" % len(dico))
    return dico, tag_to_id, id_to_tag

def prepare_dataset(sentences, word_to_id, tag_to_id, pos_to_id, conNode_to_id, depNode_to_id, semroles_to_id, loc_to_id, lower=True ):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - str_words, a sequence of words
        - words, word indexes
        - tags, tag indexes
	...
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        pos = [pos_to_id[w[1]] for w in s]#[VB,CD,NN]->[1,2,3]
        constituency_node = [conNode_to_id[w[2]] for w in s]
        dependency_node = [depNode_to_id[w[3]] for w in s]
        semroles = [semroles_to_id[w[4]] for w in s]
        semroles_word = [w[4] for w in s]
        cue = [word_to_id[f(w[5])] for w in s]
        loc = [loc_to_id[w[6]] for w in s]

        data.append({
            'str_words': str_words,
            'words': words,
            'tags': tags,
            'pos': pos,
            'constituency_node': constituency_node,
            'dependency_node': dependency_node,
            'semroles': semroles,
            'semroles_word': semroles_word,
	    'cue': cue,
	    'loc': loc,
        })
    return data













