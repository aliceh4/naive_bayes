# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020
# Modified by Kiran Ramnath 02/13/2021

"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used when your
code is evaluated, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter, defaultdict
import time
import operator


def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each mail
    example: suppose I had two mails 'like this city' and 'get rich quick' in my training set
    Then train_set := [['like','this','city'], ['get','rich','quick']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two mails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each mail that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-idf value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """
    # training: get # of docs that contain w
    word_freq_doc = Counter()
    for email in train_set:
        used = []
        for word in email:   
            if (word not in used):    
                word_freq_doc[word] += 1
                used.append(word)
            else:
                continue

    # dev
    num_train_docs = len(train_set)
    result = []
    word_freq = Counter()
    for email in dev_set:
        word_freq.clear()
        for word in email:
            word_freq[word] += 1

        max_tf = 0
        for word in email:
            tf_idf = (word_freq[word]) / (len(email)) * math.log(num_train_docs / (1 + word_freq_doc[word]))
            if tf_idf > max_tf:
                new_word = word
                max_tf = tf_idf
        result.append(new_word)    

        
    # return list of words (should return a list, not numpy array or similar)
    return result