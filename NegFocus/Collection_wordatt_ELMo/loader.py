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
    for line in codecs.open(path, 'r', 'utf-8'):#这种方法可以指定一个编码打开文件，使用这个方法打开的文件读取返回的将是unicode。写入时，如果参数 是unicode，则使用open()时指定的编码进行编码后写入；如果是str，则先根据源代码文件声明的字符编码，解码成unicode后再进行前述 操作。相对内置的open()来说，这个方法比较不容易在编码上出现问题
        # line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        line = line.strip()
        # if not line:
        #     if len(sentence) > 0:
        #         if 'DOCSTART' not in sentence[0][0]:
        #             sentences.append(sentence)
        #         sentence = []
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)#3重列表
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    # if len(sentence) > 0:
    #     if 'DOCSTART' not in sentence[0][0]:
    #         sentences.append(sentence)
    return sentences


def pos_mapping(sentences):
    """
    Create a dictionary and a mapping of POS labels, sorted by frequency.
    """
    pos = [[x[1] for x in s] for s in sentences]  # 二重列表，[[VB,NN,VP],[CC,VB]]
    dico = create_dico(pos)  # key:word, value:出现次数
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
    conNode = [[x[2] for x in s] for s in sentences]  # 二重列表，[[B-VP,I-VP,O],[O,B-VP]]
    dico = create_dico(conNode)  # key:word, value:出现次数
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
    depNode = [[x[3] for x in s] for s in sentences]  # 二重列表，[[xcomp,dobj,root],[det,amod]]
    dico = create_dico(depNode)  # key:word, value:出现次数
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
    semroles = [[x[4] for x in s] for s in sentences]  # 二重列表，[[A1,A0,AM-MOD],[V,AM-NEG]]
    dico = create_dico(semroles)  # key:word, value:出现次数
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
    loc = [[x[6] for x in s] for s in sentences]  # 二重列表，[[-5,-4,-3],[-2,-1]]
    dico = create_dico(loc)  # key:word, value:出现次数
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
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]#二重列表，[[I,have,a],[a,dream]]
    context_bef = []
    context_aft = []
    s1 = []
    s2 = []
    for s in sentences:  # 二重列表
        for x in s:
            index = x.index('###')
            for i in range(7,index):
                s1.append(x[i].lower())
            for i in range(index+1,len(x)-1):
                s2.append(x[i].lower())
        context_bef.append(s1)
        context_aft.append(s2)
        s1 = []
        s2 = []
    words.extend(context_bef)
    words.extend(context_aft)

    dico = create_dico(words)#key:word, value:出现次数

    #dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    # dico = {k:v for k,v in dico.items() if v>=3}
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
    #for i in dico:
    #    print(i)
    return dico, tag_to_id, id_to_tag



def prepare_dataset(sentences, word_to_id, tag_to_id, pos_to_id, conNode_to_id, depNode_to_id, semroles_to_id, loc_to_id, lower=True ):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]#[I,have,a]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        tags = [tag_to_id[w[-1]] for w in s]
        pos = [pos_to_id[w[1]] for w in s]#[VB,CD,NN]->[1,2,3]
        constituency_node = [conNode_to_id[w[2]] for w in s]
        dependency_node = [depNode_to_id[w[3]] for w in s]
        semroles = [semroles_to_id[w[4]] for w in s]
        semroles_word = [w[4] for w in s]
	cue = [word_to_id[f(w[5])] for w in s]#记录每个句子中的动词触发词
	loc = [loc_to_id[w[6]] for w in s]
	context_bef = []
        context_aft = []
        s1 = []
        s2 = []
        for x in s:
            index = x.index('###')
            for e in range(7,index):
                s1.append(word_to_id[f(x[e])])#[1,2,3,4,5]
            for e in range(index+1,len(x)-1):
                s2.append(word_to_id[f(x[e])])
            # s1.append(x[7:index])
            # s2.append(x[index + 1:-1])
            context_bef = s1
            context_aft = s2
            break

            #context_bef.append(s1)
            #context_aft.append(s2)
            #s1 = []
            #s2 = []

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
	    'context_bef': context_bef,
            'context_aft': context_aft,
        })
    return data













