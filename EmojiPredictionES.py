from typing import List # for type annotation
import collections      # for defaultdict
import numpy as np      # for numpy arrays

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# For Building the Model
from keras import models
from keras import layers
import pickle                       # for saving vocab to disk
# from keras.models import save_model # for saving the model to disk
import matplotlib.pyplot as plt     # for plotting train / dev training curves

# Hyper Parameters
hyper_parameters = [False, False, 64, True, 64, 'LSTM', False, 32, 0.05, 3, 64]
# 0 - whether to remove stop words
# 1 - whether to stem words
# 2 - the size of the embedding vector
# 3 - whether to use a bidirectional RNN over the embedding
# 4 - the vector size representing the input sequence
# 5 - whether to use a SimpleRNN, LSTM or GRU (accepted values: SRNN, LSTM or GRU)
# 6 - whether to use a second RNN after the first
# 7 - the vector size of the second RNN
# 8 - rate of dropout to use - float between 0 and 1
# 9 - number of epochs to train for
# 10 - batch size to use while training

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


# Clean the Tweets
# Add an EDGE token at the beginning of each tweet
# Change tokens to Lower Case
# Remove Stop Words (Hyper Parameter)
# Stem Words (Hyper Parameter)
def clean_data(tweets: List[List[str]]) -> List[List[str]]:

    remove_stopwords: bool = hyper_parameters[0]
    stem_tokens: bool = hyper_parameters[1]

    stop_words = set(stopwords.words('spanish'))
    stemmer = nltk.stem.SnowballStemmer('spanish')

    clean_tweets: List[List[str]] = []
    for tweet in tweets:
        clean_tweet: List[str] = ['EDGE']
        for token in tweet:
            # change the token to lower case
            token = token.lower()
            # if stop words are being removed (hyper parameter)
            if remove_stopwords:
                #  and the token is a stop word, skip/ignore it
                if token in stop_words:
                    continue
            # if tokens are being stemmed (hyper parameter)
            if stem_tokens:
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

# Save vocab to disk for use in prediction when loading a trained model
with open('models/es_vocab.pkl', 'wb+') as f:
    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
print('Vocab saved to disk')


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
plt.plot(epochs, train_losses, 'b-', label='Train loss')
plt.plot(epochs, dev_losses, 'r-', label='Dev loss')
plt.title('Train and dev loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 6)
plt.grid()
plt.legend()
plt.show()


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

# use the trained model to predict unseen data. Output will be a List of 19 scores
softmax_output: List[List[float]] = rnn_model.predict(test_tweets_padded)
# a list that will store which label the model scored the highest
output_labels: List[int] = []
for scores in softmax_output:
    # choose the output to be the label which obtained the highest score
    highest_score_label: int = np.argmax(scores)
    output_labels.append(highest_score_label)

import Evaluation
Evaluation.main(test_labels, output_labels)

# Save the trained RNN to disk
rnn_model.save('models/es_rnn_model.h5')
print("Saved ES RNN to disk")
