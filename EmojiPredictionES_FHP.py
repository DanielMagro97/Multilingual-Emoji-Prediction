from typing import List # for type annotation
import collections      # for defaultdict
import numpy as np      # for numpy arrays

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# For Building the Model
from keras import models
from keras import layers
import matplotlib.pyplot as plt # for plotting train / dev training curves

best_hyper_parameters = [False, False, 64, True, 64, 'LSTM', False, 32, 0.05, 3, 64]
best_test_error = np.inf

# Load the Training Data
# load the training tweets
with open('Semeval2018-Task2-EmojiPrediction/train/crawler/data/es/tweet_by_ID_04_2_2019__03_18_24.txt.text', 'r', encoding="utf-8") as f:
    train_tweets = [tweet.strip().split() for tweet in f]
# load the training labels
with open('Semeval2018-Task2-EmojiPrediction/train/crawler/data/es/tweet_by_ID_04_2_2019__03_18_24.txt.labels', 'r', encoding="utf-8") as f:
    train_labels = [int(label.strip()) for label in f]

# Load the Validation Data
# load the validation/development tweets
with open('Semeval2018-Task2-EmojiPrediction/trial/es_trial.text', 'r', encoding="utf-8") as f:
    val_tweets = [tweet.strip().split() for tweet in f]
# load the validation/development labels
with open('Semeval2018-Task2-EmojiPrediction/trial/es_trial.labels', 'r', encoding="utf-8") as f:
    val_labels = [int(label.strip()) for label in f]


# -*- coding: utf-8 -*-
from codecs import open
import sys

# This script evaluates the systems on the SemEval 2018 task on Emoji Prediction.
# It takes the gold standard and system's output file as input and prints the results in terms of macro and micro average F-Scores (0-100).

def f1(precision, recall):
    return (2.0 * precision * recall) / (precision + recall)
def main(path_goldstandard, path_outputfile):
    truth_dict = {}
    output_dict_correct = {}
    output_dict_attempted = {}
    # truth_file_lines = open(path_goldstandard, encoding='utf8').readlines()
    # submission_file_lines = open(path_outputfile, encoding='utf8').readlines()
    truth_file_lines = path_goldstandard
    submission_file_lines = path_outputfile
    if len(submission_file_lines) != len(truth_file_lines): sys.exit(
        'ERROR: Number of lines in gold and output files differ')
    for i in range(len(submission_file_lines)):
        line = submission_file_lines[i]
        emoji_code_gold = truth_file_lines[i]#.replace("\n", "")
        if emoji_code_gold not in truth_dict:
            truth_dict[emoji_code_gold] = 1
        else:
            truth_dict[emoji_code_gold] += 1
        emoji_code_output = submission_file_lines[i]#.replace("\n", "")
        if emoji_code_output == emoji_code_gold:
            if emoji_code_output not in output_dict_correct:
                output_dict_correct[emoji_code_gold] = 1
            else:
                output_dict_correct[emoji_code_output] += 1
        if emoji_code_output not in output_dict_attempted:
            output_dict_attempted[emoji_code_output] = 1
        else:
            output_dict_attempted[emoji_code_output] += 1
    precision_total = 0
    recall_total = 0
    num_emojis = len(truth_dict)
    attempted_total = 0
    correct_total = 0
    gold_occurrences_total = 0
    f1_total = 0
    for emoji_code in truth_dict:
        gold_occurrences = truth_dict[emoji_code]
        if emoji_code in output_dict_attempted:
            attempted = output_dict_attempted[emoji_code]
        else:
            attempted = 0
        if emoji_code in output_dict_correct:
            correct = output_dict_correct[emoji_code]
        else:
            correct = 0
        if attempted != 0:
            precision = (correct * 1.0) / attempted
            recall = (correct * 1.0) / gold_occurrences
            if precision != 0.0 or recall != 0.0: f1_total += f1(precision, recall)
        attempted_total += attempted
        correct_total += correct
        gold_occurrences_total += gold_occurrences
    macrof1 = f1_total / (num_emojis * 1.0)
    precision_total_micro = (correct_total * 1.0) / attempted_total
    recall_total_micro = (correct_total * 1.0) / gold_occurrences_total
    if precision_total_micro != 0.0 or recall_total_micro != 0.0:
        microf1 = f1(precision_total_micro, recall_total_micro)
    else:
        microf1 = 0.0
    # print("Macro F-Score (official): " + str(round(macrof1 * 100, 3)))
    # print("-----")
    # print("Micro F-Score: " + str(round(microf1 * 100, 3)))
    # print("Precision: " + str(round(precision_total_micro * 100, 3)))
    # print("Recall: " + str(round(recall_total_micro * 100, 3)))
    return 1 / round(macrof1 * 100, 3)



def test_model(hyper_parameters) -> float:
    print()
    print(hyper_parameters)

    # Clean the Tweets
    # Add an EDGE token at the beginning of each tweet
    # Change tokens to Lower Case
    # Remove Stop Words (Hyper Parameter)
    # Stem Words (Hyper Parameter)
    def clean_data(tweets: List[List[str]]) -> List[List[str]]:
        stop_words = set(stopwords.words('spanish'))
        stemmer = nltk.stem.SnowballStemmer('spanish')

        clean_tweets: List[List[str]] = []
        for tweet in tweets:
            clean_tweet: List[str] = ['EDGE']
            for token in tweet:
                # change the token to lower case
                token = token.lower()
                # if stop words are being removed (hyper parameter)
                if hyper_parameters[0]:
                    #  and the token is a stop word, skip/ignore it
                    if token in stop_words:
                        continue
                # if tokens are being stemmed (hyper parameter)
                if hyper_parameters[1]:
                    # add the stemmed token to the list for this tweet
                    clean_tweet.append(stemmer.stem(token))
                else:
                    # if not, add the 'unstemmed' token to the list for this tweet
                    clean_tweet.append(token)
            # if a bidirectional RNN will be used, add an EDGE token to the end of the tweet too
            if hyper_parameters[3]:
                clean_tweet.append('EDGE')
            # add the list of tokens for the current tweet to the list of clean tweets
            clean_tweets.append(clean_tweet)

        return clean_tweets

    train_tweets_clean: List[List[str]] = clean_data(train_tweets)
    val_tweets_clean: List[List[str]] = clean_data(val_tweets)


    # Define a Function for Extracting a Vocabulary from Data
    def get_vocab(tweets: List[List[str]], min_freq: int):
        # get token frequencies from data
        token_freqs = collections.defaultdict(lambda: 0)
        for tweet in tweets:
            for token in tweet:
                token_freqs[token] += 1

        # sort tokens by their frequency in descending order
        vocab = sorted(token_freqs.keys(), key=token_freqs.get, reverse=True)

        # remove low frequency words from the end of the vocabulary until the minimum frequency is encountered
        while token_freqs[vocab[-1]] < min_freq:
            vocab.pop()

        # remove 'EDGE' from the vocab as it will be manually added in at index 1 (to avoid duplicates)
        vocab.remove('EDGE')

        # return the extracted vocab, plus 3 keyword terms
        return ['PAD', 'EDGE', 'UNKNOWN'] + vocab

    # Extract the Vocabulary from the Training Data
    vocab = get_vocab(train_tweets_clean, 3)


    # Create a Dictionary which, for a given token ('word'), returns its index.
    # A default dictionary is used so that for a token not in the vocabulary, 2 is returned (the index for UNKNOWN),
    # instead of throwing an error.
    token2index = collections.defaultdict(lambda: 2)
    for (index, token) in enumerate(vocab):
        token2index[token] = index
    # index2token = { index: token for (index, token) in enumerate(vocab) }


    # Define a Function for converting the tweets from:
    # A List of Lists, where each inner List contains strings (words), to:
    # A List of Lists, where each inner List contains ints (indexes)
    def index_data(tweets: List[List[str]]) -> List[List[int]]:
        # create a List (of lists) which will store all the individual tweets (stored as lists of ints)
        # iterate through all the tweets
        # create a list which will store the indexes of all the tokens in this tweet (list of ints)
        # iterate through all the tokens in the tweet
        # add the index of the current token to the end of the list for the current tweets
        # once the list of indexes for the current tweet has been generated, add it to the end of the list of tweets
        return [ [ token2index[token] for token in tweet ] for tweet in tweets ]

    # Index the train_tweets and the val_tweets
    train_tweets_index: List[List[int]] = index_data(train_tweets_clean)
    val_tweets_index: List[List[int]] = index_data(val_tweets_clean)
    # print(' '.join(index2token[i] for i in train_tweets_index[0]))


    # Define a function for converting a given List of Tweets into a Numpy Array with Padding
    def pad_data(tweets: List[List[int]]) -> np.ndarray:
        # find how long the longest tweet is (how many words)
        max_tweet_length = max(len(tweet) for tweet in tweets)
        # create a numpy array with as many rows as there are tweets,
        # and as many columns as there are words in the longest tweet
        tweets_padded = np.zeros([len(tweets), max_tweet_length], np.int32)
        # populate the numpy array by iterating over every tweet in the List of Tweets
        for (i, tweet) in enumerate(tweets):
            # let n be the number of words in a tweet (its length) (a tweet is a List of ints)
            # set the first n elements in the numpy array for that tweet's row to its values in the list.
            # the remaining, untouched elements will remain as 0, which means PAD
            tweets_padded[i, :len(tweet)] = tweet

        return tweets_padded

    # Pad the train_tweets and val_tweets (indexed)
    train_tweets_padded: np.ndarray = pad_data(train_tweets_index)
    val_tweets_padded: np.ndarray = pad_data(val_tweets_index)


    # Build the Neural Net
    def build_neural_net(hyperparameters):
        # the size of the embedding vector
        embedding_vector_size: int = hyperparameters[2]
        # whether to use a bidirectional RNN over the embedding
        use_bidirectional: bool = hyperparameters[3]
        # the vector size representing the input sequence
        input_vector_representation_size: int = hyperparameters[4]
        # whether to use a SimpleRNN, LSTM or GRU
        use_SRNN_LSTM_GRU: str = hyperparameters[5]
        # whether to use a second RNN after the first
        use_second_RNN: bool = hyperparameters[6]
        # the vector size of the second RNN
        intermediate_vector_representation_size: int = hyperparameters[7]
        # rate of dropout to use - float between 0 and 1
        dropout: float = hyperparameters[8]

        model = models.Sequential()

        # Add an Embedding layer to the model, with as many inputs as terms in the vocab,
        # and as many nodes as defined by the embedding_vector_size hyper parameter
        model.add(layers.Embedding(len(vocab), embedding_vector_size, input_length=None, mask_zero=True))

        # Add the first RNN Layer. If the use_bidirectional hyper parameter is set to True,
        # then use a bidirectional implementation
        if use_bidirectional:
            # Add the first RNN Layer as a Simple RNN, LSTM or GRU depending on the use_SRNN_LSTM_GRU hyper parameter
            # also use dropoput according to the hyper parameter
            # and return sequences of the first layer depending on whether a second Recursive Layer will be used
            if use_SRNN_LSTM_GRU == 'SRNN':
                model.add(layers.Bidirectional(layers.SimpleRNN(input_vector_representation_size, dropout=dropout,
                                                                return_sequences=use_second_RNN)))
            elif use_SRNN_LSTM_GRU == 'LSTM':
                model.add(layers.Bidirectional(layers.LSTM(input_vector_representation_size, dropout=dropout,
                                                           return_sequences=use_second_RNN)))
            elif use_SRNN_LSTM_GRU == 'GRU':
                model.add(layers.Bidirectional(layers.GRU(input_vector_representation_size, dropout=dropout,
                                                          return_sequences=use_second_RNN)))
        else:
            if use_SRNN_LSTM_GRU == 'SRNN':
                model.add(layers.SimpleRNN(input_vector_representation_size, dropout=dropout,
                                           return_sequences=use_second_RNN))
            elif use_SRNN_LSTM_GRU == 'LSTM':
                model.add(layers.LSTM(input_vector_representation_size, dropout=dropout,
                                      return_sequences=use_second_RNN))
            elif use_SRNN_LSTM_GRU == 'GRU':
                model.add(layers.GRU(input_vector_representation_size, dropout=dropout,
                                     return_sequences=use_second_RNN))

        if use_second_RNN:
            if use_SRNN_LSTM_GRU == 'SRNN':
                model.add(layers.SimpleRNN(intermediate_vector_representation_size, dropout=dropout))
            elif use_SRNN_LSTM_GRU == 'LSTM':
                model.add(layers.LSTM(intermediate_vector_representation_size, dropout=dropout))
            elif use_SRNN_LSTM_GRU == 'GRU':
                model.add(layers.GRU(intermediate_vector_representation_size, dropout=dropout))

        # softmax layer
        model.add(layers.Dense(19, activation='softmax'))

        return model

    rnn_model = build_neural_net(hyper_parameters)

    rnn_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy', #when the output is a softmax layer, use this loss function to measure the error
                        metrics=['acc'])

    history = rnn_model.fit(train_tweets_padded, train_labels,
                              epochs=hyper_parameters[9],
                              batch_size=hyper_parameters[10],
                              validation_data=(val_tweets_padded, val_labels))

    train_losses = history.history['loss']
    dev_losses = history.history['val_loss']
    epochs = range(1, len(history.history['loss']) + 1)
    # plt.plot(epochs, train_losses, 'b-', label='Train loss')
    # plt.plot(epochs, dev_losses, 'r-', label='Dev loss')
    # plt.title('Train and dev loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.ylim(0, 6)
    # plt.grid()
    # plt.legend()
    # plt.show()


    # Evaluation - Predicting unseen data

    # load the test/evaluation texts
    with open('Semeval2018-Task2-EmojiPrediction/test/es_test.text', 'r', encoding="utf-8") as f:
        test_tweets = [tweet.strip().split() for tweet in f]
    # load the test/evaluation labels
    with open('Semeval2018-Task2-EmojiPrediction/test/es_test.labels', 'r', encoding="utf-8") as f:
        test_labels = [int(label.strip()) for label in f]

    test_tweets_clean: List[List[str]] = clean_data(test_tweets)
    test_tweets_index: List[List[int]] = index_data(test_tweets_clean)
    test_tweets_padded: np.ndarray = pad_data(test_tweets_index)

    output = rnn_model.predict(test_tweets_padded)

    output_labels: List[int] = []
    for predicted_emoji in output:
        # find which label obtained the highest score
        best_label: int = np.argmax(predicted_emoji)
        output_labels.append(best_label)

    inv_score: float = main(test_labels, output_labels)
    print(str(1/inv_score))

    global best_test_error
    global best_hyper_parameters
    if inv_score < best_test_error:
        best_test_error = inv_score
        best_hyper_parameters = hyper_parameters

        print('\n\n')
        print(hyper_parameters)
        print(str(inv_score))
        print(str(1/inv_score))

        print(best_hyper_parameters)
        print(best_test_error)
        print('\n\n')

    return inv_score

from skopt import forest_minimize
import skopt.space

forest_minimize(test_model, [skopt.space.Categorical([False, True]),
                             skopt.space.Categorical([False, True]),
                             skopt.space.Categorical([4,8,16,32,64,128,256]),
                             skopt.space.Categorical([False, True]),
                             skopt.space.Categorical([4,8,16,32,64,128,256]),
                             skopt.space.Categorical(['SRNN', 'LSTM', 'GRU']),
                             skopt.space.Categorical([False, True]),
                             skopt.space.Categorical([4,8,16,32,64,128,256]),
                             skopt.space.Real(1e-5, 10, "log-uniform"),
                             skopt.space.Integer(1, 5),
                             skopt.space.Categorical([32,64,128,256,512])],
                base_estimator="RF", n_calls=1000,
                n_random_starts=10, acq_func="EI",
                x0=[False, False, 64, True, 64, 'LSTM', False, 32, 0.05, 3, 64])
