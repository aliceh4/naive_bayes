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
    ham_freq, spam_freq, ham_freq2, spam_freq2 = calculate_likelihood(train_set, train_labels)
    labels = development_phase(ham_freq, spam_freq, ham_freq2, spam_freq2, dev_set, unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior, bigram_lambda)
    return labels

def development_phase(ham_freq, spam_freq, ham_freq2, spam_freq2, dev_set, unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior, bigram_lambda):
    # Initialize our labels list
    labels = []

    total_ham = sum(ham_freq.values())
    total_spam = sum(spam_freq.values())
    ham_len = len(list(ham_freq))
    spam_len = len(list(spam_freq))

    total_ham2 = sum(ham_freq2.values())
    total_spam2 = sum(spam_freq2.values())
    ham_len2 = len(list(ham_freq2))
    spam_len2 = len(list(spam_freq2))

    for email in dev_set:
        # Take log to prevent underflow issues
        prob_ham = math.log10(pos_prior)
        prob_spam = math.log10(1.0 - pos_prior)

        prob_ham2 = math.log10(pos_prior)
        prob_spam2 = math.log10(1.0 - pos_prior)

        # Go through each word in email
        for j in range (len(email)):

            # UNIGRAM
            word = email[j]
            # Check if word is in our likelihood dict (if word is not present, then likelihood = [0 + k] / [N + k|X|])
            if word in ham_freq.keys():
                prob_ham += math.log10(float(ham_freq[word] + unigram_smoothing_parameter) / float(total_ham + unigram_smoothing_parameter * ham_len))
            else:
                prob_ham += math.log10(float(unigram_smoothing_parameter) / float(total_ham + unigram_smoothing_parameter * ham_len))

            if word in spam_freq.keys():
                prob_spam += math.log10(float(spam_freq[word] + unigram_smoothing_parameter) / float(total_spam + unigram_smoothing_parameter * spam_len))
            else:
                prob_spam += math.log10(float(unigram_smoothing_parameter) / float(total_spam + unigram_smoothing_parameter * spam_len))
            
            # BIGRAM
            if (j < len(email) - 1):
                bigram_words = email[j] + email[j + 1]
                if bigram_words in ham_freq2.keys():
                    prob_ham2 += math.log10(float(ham_freq2[word] + bigram_smoothing_parameter) / float(total_ham2 + bigram_smoothing_parameter * ham_len2))
                else:
                    prob_ham2 += math.log10(float(bigram_smoothing_parameter) / float(total_ham2 + bigram_smoothing_parameter * ham_len2))

                if bigram_words in spam_freq2.keys():
                    prob_spam2 += math.log10(float(spam_freq2[word] + bigram_smoothing_parameter) / float(total_spam2 + bigram_smoothing_parameter * spam_len2))
                else:
                    prob_spam2 += math.log10(float(bigram_smoothing_parameter) / float(total_spam2 + bigram_smoothing_parameter * spam_len2))

        # compare probabilities and populate our labels list accordingly
        ham_prob = (1 - bigram_lambda) * prob_ham + bigram_lambda * prob_ham2
        spam_prob = (1 - bigram_lambda) * prob_spam + bigram_lambda * prob_spam2
        if ((ham_prob) > (spam_prob)):
            labels.append(1)
        else:
            labels.append(0)
    return labels


def calculate_likelihood(train_set, train_labels):
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
        for j in range(0, len(email)):
            word = email[j]
            if (label == 1):
                ham_freq[word] += 1
            else:
                spam_freq[word] += 1
            
            if j < len(email) - 1:
                bigram_words = email[j] + email[j + 1]
                if label == 1:
                     ham_freq2[bigram_words] += 1
                else:
                    spam_freq2[bigram_words] += 1
    return ham_freq, spam_freq, ham_freq2, spam_freq2