# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

from collections import Counter
import math


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    # return predicted labels of development set

    # first get our list of likelihoods
    ham_likelihood, spam_likelihood, ham_freq, spam_freq = calculate_likelihood(train_set, train_labels, smoothing_parameter)
    predicted_labels = development_phase(ham_likelihood, spam_likelihood, ham_freq, spam_freq, dev_set, smoothing_parameter, pos_prior)

    return predicted_labels


def calculate_likelihood(train_set, train_labels, smoothing_parameter):
    """
    We want to calculate the posterior probabilities: P(Ham | Words) = P(Ham) * product of P(word | ham)

    We can think of this as our "training" phase
    """
    # Initialize our counter variables
    ham_freq = Counter()
    spam_freq = Counter()

    # Iterate through train_set and populate frequency in corresponding locations
    for i in range (0, len(train_set)):
        email = train_set[i]
        label = train_labels[i]

        # now iterate through the words in each email
        for word in email:
            if (label == 1):
                ham_freq[word] += 1
            else:
                spam_freq[word] += 1
    
    # Have likelihoods be dict with format: {string: double} = {"word": P(ham), ...}
    ham_likelihood  = {}
    spam_likelihood = {}

    # iterate through each unique word in ham_freq
    # NOTE: likelihood = [count(x) + k] / [N + k|X|]
    for word in list(ham_freq):
        # In future, maybe change |X| to be number of unique words in both spam and ham
        ham_likelihood[word] = float(ham_freq[word] + smoothing_parameter) / float(sum(ham_freq.values()) + smoothing_parameter * len(list(ham_freq)))

    for word in list(spam_freq):
        spam_likelihood[word] = float(spam_freq[word] + smoothing_parameter) / float(sum(spam_freq.values()) + smoothing_parameter * len(list(spam_freq)))
    
    # NOTE: we also return ham_freq and spam_freq here so it is easer for our program to calculate a probability if word is not in our likelihood dicts
    return ham_likelihood, spam_likelihood, ham_freq, spam_freq


def development_phase(ham_likelihood, spam_likelihood, ham_freq, spam_freq, dev_set, smoothing_parameter, pos_prior):
    # Initialize our labels list
    labels = []

    # Iterate through our emails
    for email in dev_set:
        # Take log to prevent underflow issues
        prob_ham = math.log(pos_prior)
        prob_spam = math.log(1.0 - pos_prior)

        # Go through each word in email
        for word in email:
            # Check if word is in our likelihood dict (if word is not present, then likelihood = [0 + k] / [N + k|X|])
            if word in ham_likelihood:
                prob_ham += math.log(ham_likelihood[word])
            else:
                prob_ham += math.log(float(smoothing_parameter) / float(sum(ham_freq.values()) + smoothing_parameter * len(list(ham_freq))))

            if word in spam_likelihood:
                prob_spam += math.log(spam_likelihood[word])
            else:
                prob_spam += math.log(float(smoothing_parameter) / float(sum(spam_freq.values()) + smoothing_parameter * len(list(spam_freq))))

        # compare probabilities and populate our labels list accordingly
        if (prob_ham > prob_spam):
            labels.append(1)
        else:
            labels.append(0)

    return labels

