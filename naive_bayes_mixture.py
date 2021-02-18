# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

from collections import Counter
import math


def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda, unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    # return predicted labels of development set
    ham_likelihood, spam_likelihood, ham_freq, spam_freq, ham_likelihood2, spam_likelihood2, ham_freq2, spam_freq2 = calculate_likelihood(train_set, train_labels, unigram_smoothing_parameter, bigram_smoothing_parameter)
    labels = development_phase(ham_likelihood, spam_likelihood, ham_freq, spam_freq, ham_likelihood2, spam_likelihood2, ham_freq2, spam_freq2, dev_set, unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior, bigram_lambda)


    return labels

def development_phase(ham_likelihood, spam_likelihood, ham_freq, spam_freq, ham_likelihood2, spam_likelihood2, ham_freq2, spam_freq2, dev_set, unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior, bigram_lambda):
    # Initialize our labels list
    labels = []

    for email in dev_set:
        # Take log to prevent underflow issues
        prob_ham = math.log(pos_prior)
        prob_spam = math.log(1.0 - pos_prior)

        prob_ham2 = math.log(pos_prior)
        prob_spam2 = math.log(1.0 - pos_prior)

        # Go through each word in email
        for j in range (len(email)):

            # UNIGRAM
            word = email[j]
            # Check if word is in our likelihood dict (if word is not present, then likelihood = [0 + k] / [N + k|X|])
            if word in ham_likelihood:
                prob_ham += math.log(ham_likelihood[word])
            else:
                prob_ham += math.log(float(unigram_smoothing_parameter) / float(sum(ham_freq.values()) + unigram_smoothing_parameter * len(list(ham_freq))))

            if word in spam_likelihood:
                prob_spam += math.log(spam_likelihood[word])
            else:
                prob_spam += math.log(float(unigram_smoothing_parameter) / float(sum(spam_freq.values()) + unigram_smoothing_parameter * len(list(spam_freq))))
            
            # BIGRAM
            if (j < len(email) - 1):
                bigram_words = email[j] + email[j + 1]
                if bigram_words in ham_likelihood2:
                    prob_ham2 += math.log(ham_likelihood2[bigram_words])
                else:
                    prob_ham2 += math.log(float(bigram_smoothing_parameter) / float(sum(ham_freq.values()) + bigram_smoothing_parameter * len(list(ham_freq))))

                if bigram_words in spam_likelihood:
                    prob_spam2 += math.log(spam_likelihood[bigram_words])
                else:
                    prob_spam2 += math.log(float(bigram_smoothing_parameter) / float(sum(spam_freq.values()) + bigram_smoothing_parameter * len(list(spam_freq))))

        # compare probabilities and populate our labels list accordingly
        ham_prob = (1 - bigram_lambda) * prob_ham + bigram_lambda * prob_ham2
        spam_prob = (1 - bigram_lambda) * prob_spam + bigram_lambda * prob_spam2
        if ((ham_prob) > (spam_prob)):
            labels.append(1)
        else:
            labels.append(0)

    return labels


def calculate_likelihood(train_set, train_labels, unigram_smoothing_parameter, bigram_smoothing_parameter):
    """
    We want to calculate the posterior probabilities: P(Ham | Words) = P(Ham) * product of P(word | ham)

    We can think of this as our "training" phase
    """
    # Initialize our counter variables
    ham_freq = Counter()
    spam_freq = Counter()
    ham_freq2 = Counter()
    spam_freq2 = Counter()

    # Iterate through train_set and populate frequency in corresponding locations
    for i in range (0, len(train_set)):
        email = train_set[i]
        label = train_labels[i]

        # now iterate through the words in each email
        for j in range(0, len(email) - 1):
            # this is a bigram model, so combine pairs
            bigram_words = email[j] + " " + email[j + 1]
            word = email[j]
            if (label == 1):
                ham_freq2[bigram_words] += 1
                ham_freq[word] += 1
            else:
                spam_freq2[bigram_words] += 1
                spam_freq[word] += 1
        # take care of last word for unigram model
        word = email[len(email) - 1]
        if (label == 1):
            ham_freq[word] += 1
        else:
            spam_freq[word] += 1     
    
    # Have likelihoods be dict with format: {string: double} = {"word": P(ham), ...}
    ham_likelihood  = {}
    spam_likelihood = {}
    ham_likelihood2  = {}
    spam_likelihood2 = {}

    # iterate through each unique word in ham_freq
    # NOTE: likelihood = [count(x) + k] / [N + k|X|]

    # UNIGRAM
    for word in list(ham_freq):
        # In future, maybe change |X| to be number of unique words in both spam and ham
        ham_likelihood[word] = float(ham_freq[word] + unigram_smoothing_parameter) / float(sum(ham_freq.values()) + unigram_smoothing_parameter * len(list(ham_freq)))

    for word in list(spam_freq):
        spam_likelihood[word] = float(spam_freq[word] + unigram_smoothing_parameter) / float(sum(spam_freq.values()) + unigram_smoothing_parameter * len(list(spam_freq)))

    # BIGRAM
    for word in list(ham_freq2):
        # In future, maybe change |X| to be number of unique words in both spam and ham
        ham_likelihood2[word] = float(ham_freq2[word] + bigram_smoothing_parameter) / float(sum(ham_freq2.values()) + bigram_smoothing_parameter * len(list(ham_freq2)))

    for word in list(spam_freq2):
        spam_likelihood2[word] = float(spam_freq2[word] + bigram_smoothing_parameter) / float(sum(spam_freq2.values()) + bigram_smoothing_parameter * len(list(spam_freq2)))
    
    return ham_likelihood, spam_likelihood, ham_freq, spam_freq, ham_likelihood2, spam_likelihood2, ham_freq2, spam_freq2