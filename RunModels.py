import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import Evaluation
import collections  # for default dict
import pickle       # for loading vocab from disk
from typing import List # for type annotation
import numpy as np
from keras.models import load_model # for loading models from disk

def clean_data(tweets: List[List[str]], language: str) -> List[List[str]]:
    hyper_parameters = [False, False, 64, True, 64, 'LSTM', False, 32, 0.05, 3, 64]

    remove_stopwords: bool = hyper_parameters[0]
    stem_tokens: bool = hyper_parameters[1]

    if language == 'us':
        stop_words = set(stopwords.words('spanish'))
        stemmer = nltk.stem.SnowballStemmer('spanish')
    elif language == 'es':
        stop_words = set(stopwords.words('spanish'))
        stemmer = nltk.stem.SnowballStemmer('spanish')
    else:
        print('invalid arguments, must be us or es')

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

def index_data(tweets: List[List[str]], token2index) -> List[List[int]]:
    # create a List (of lists) which will store all the individual tweets (stored as lists of ints)
    # iterate through all the tweets
    # create a list which will store the indexes of all the tokens in this tweet (list of ints)
    # iterate through all the tokens in the tweet
    # add the index of the current token to the end of the list for the current tweets
    # once the list of indexes for the current tweet has been generated, add it to the end of the list of tweets
    return [ [ token2index[token] for token in tweet ] for tweet in tweets ]

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


# load the US test/evaluation texts
with open('Semeval2018-Task2-EmojiPrediction/test/us_test.text', 'r', encoding="utf-8") as f:
    us_test_tweets = [tweet.strip().split() for tweet in f]
# load the US test/evaluation labels
with open('Semeval2018-Task2-EmojiPrediction/test/us_test.labels', 'r', encoding="utf-8") as f:
    us_test_labels = [int(label.strip()) for label in f]

# pre process the test data
us_test_tweets_clean: List[List[str]] = clean_data(us_test_tweets, 'us')
with open('models/us_vocab.pkl', 'rb') as f:
    us_vocab = pickle.load(f)
us_token2index = collections.defaultdict(lambda: 2)
for (index, token) in enumerate(us_vocab):
    us_token2index[token] = index
us_test_tweets_index: List[List[int]] = index_data(us_test_tweets_clean, us_token2index)
us_test_tweets_padded: np.ndarray = pad_data(us_test_tweets_index)

# load the trained us model
us_rnn_model = load_model('models/us_rnn_model.h5')
print("Loaded US RNN from disk")

# use the trained model to predict unseen data. Output will be a List of 20 scores
us_softmax_output: List[List[float]] = us_rnn_model.predict(us_test_tweets_padded)
# a list that will store which label the model scored the highest
us_output_labels: List[int] = []
for scores in us_softmax_output:
    # choose the output to be the label which obtained the highest score
    highest_score_label: int = np.argmax(scores)
    us_output_labels.append(highest_score_label)

print('US Model Results:')
Evaluation.main(us_test_labels, us_output_labels)

print()

# load the ES test/evaluation texts
with open('Semeval2018-Task2-EmojiPrediction/test/es_test.text', 'r', encoding="utf-8") as f:
    es_test_tweets = [tweet.strip().split() for tweet in f]
# load the ES test/evaluation labels
with open('Semeval2018-Task2-EmojiPrediction/test/es_test.labels', 'r', encoding="utf-8") as f:
    es_test_labels = [int(label.strip()) for label in f]

# pre process the test data
es_test_tweets_clean: List[List[str]] = clean_data(es_test_tweets, 'es')
with open('models/es_vocab.pkl', 'rb') as f:
    es_vocab = pickle.load(f)
es_token2index = collections.defaultdict(lambda: 2)
for (index, token) in enumerate(es_vocab):
    es_token2index[token] = index
es_test_tweets_index: List[List[int]] = index_data(es_test_tweets_clean, es_token2index)
es_test_tweets_padded: np.ndarray = pad_data(es_test_tweets_index)

# load the trained es model
es_rnn_model = load_model('models/es_rnn_model.h5')
print("Loaded ES RNN from disk")

# use the trained model to predict unseen data. Output will be a List of 19 scores
es_softmax_output: List[List[float]] = es_rnn_model.predict(es_test_tweets_padded)
# a list that will store which label the model scored the highest
es_output_labels: List[int] = []
for scores in es_softmax_output:
    # choose the output to be the label which obtained the highest score
    highest_score_label: int = np.argmax(scores)
    es_output_labels.append(highest_score_label)

print('ES Model Results:')
Evaluation.main(es_test_labels, es_output_labels)


display_outputs: str = input("would you like to see a sample of the outputs? 'y' for yes, 'n' for no\n")
if display_outputs == 'y':

    # load the emoji mappings (label number -> emoji image matrix)
    with open('Semeval2018-Task2-EmojiPrediction/mapping/us_mapping.txt', 'r', encoding="utf-8") as f:
        us_emoji_mapping = [mapping.strip().split() for mapping in f]

    # load the emoji mappings (label number -> emoji image matrix)
    with open('Semeval2018-Task2-EmojiPrediction/mapping/es_mapping.txt', 'r', encoding="utf-8") as f:
        es_emoji_mapping = [mapping.strip().split() for mapping in f]

    count_correct: int = 0
    count_wrong: int = 0
    print('US Tweets')
    for i, tweet in enumerate(us_test_tweets):
        if us_output_labels[i] == us_test_labels[i] and count_correct < 3:
            print(tweet)
            print('correctly classified as:')
            print(us_emoji_mapping[us_output_labels[i]][1] + ' ' + us_emoji_mapping[us_output_labels[i]][2])
            print()
            count_correct += 1

        if us_output_labels[i] != us_test_labels[i] and count_correct < 5:
            print(tweet)
            print('incorrectly classified as:')
            print(us_emoji_mapping[us_output_labels[i]][1] + ' ' + us_emoji_mapping[us_output_labels[i]][2])
            print('should be labeled labelled as')
            print(us_emoji_mapping[us_test_labels[i]][1] + ' ' + us_emoji_mapping[us_test_labels[i]][2])
            print()
            count_wrong += 1


    count_correct: int = 0
    count_wrong: int = 0
    print('ES Tweets')
    for i, tweet in enumerate(es_test_tweets):
        if es_output_labels[i] == es_test_labels[i] and count_correct < 3:
            print(tweet)
            print('correctly classified as:')
            print(es_emoji_mapping[es_output_labels[i]][1] + ' ' + es_emoji_mapping[es_output_labels[i]][2])
            print()
            count_correct += 1

        if es_output_labels[i] != es_test_labels[i] and count_correct < 5:
            print(tweet)
            print('incorrectly classified as:')
            print(es_emoji_mapping[es_output_labels[i]][1] + ' ' + es_emoji_mapping[es_output_labels[i]][2])
            print('should be labeled labelled as')
            print(es_emoji_mapping[es_test_labels[i]][1] + ' ' + es_emoji_mapping[es_test_labels[i]][2])
            print()
            count_wrong += 1
