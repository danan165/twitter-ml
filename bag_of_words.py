import pandas as pd
import spacy
import numpy as np

from string import punctuation
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
from sklearn.svm import LinearSVC


nlp = spacy.load("en_core_web_sm")


def clean_tweet(tweet):
    '''
    Removes stopwords/punctuation/numbers. Lemmatizes remaining words (ex: 'going' --> 'go')
    Returns the clean tweet as a list of tokens.
    '''
    # cast to lowercase + remove punctuation
    tweet = tweet.lower()
    for char in punctuation:
        tweet = tweet.replace(char, ' ')

    # remove stopwords/numbers
    doc = nlp(tweet)

    clean_tokens = []
    for token in doc:
        if token.is_stop == False and str(token).isalpha() == True:
            clean_tokens.append(token)

    # lemmatize clean_tokens
    lookups = Lookups()
    lookups.add_table("lemma_rules", {"noun": [["s", ""]]})
    lemmatizer = Lemmatizer(lookups)
    for i in range(len(clean_tokens)):
        token = clean_tokens[i]
        lemmas = lemmatizer(str(token), str(token.pos_))
        clean_tokens[i] = lemmas[0]

    return clean_tokens


def create_dictionary(tweets):
    '''
    Given list of all tweets, creates a dictionary mapping each distinct
    word in the corpus to a unique index. 
    '''
    tweet_dictionary = dict()
    idx = 0

    for tweet in tweets:
        clean_tokens = clean_tweet(tweet)
        for token in clean_tokens:
            str_token = str(token)
            if str_token not in tweet_dictionary:
                tweet_dictionary[str_token] = idx
                idx += 1

    return tweet_dictionary


def generate_feature_matrix(tweets, tweet_dictionary):
    '''
    For each tweet, generate a vector the same length as tweet_dictionary. Put a '1' 
    at the vector index matching the tweet_dictionary index if that word appears in the tweet.
    Put a '0' otherwise. Return the matrix as a 2D numpy array.
    '''
    num_tweets = len(tweets)
    num_words = len(tweet_dictionary)
    feature_matrix = np.zeros((num_tweets, num_words))

    tweets = list(tweets)

    for i in range(num_tweets):
        clean_tokens = clean_tweet(tweets[i])
        for token in clean_tokens:
            token = str(token)
            if token in tweet_dictionary:
                feature_matrix[i][tweet_dictionary[token]] = 1

    return feature_matrix


def main():
    # read data from pickle files
    aoc_df = pd.read_pickle('data/aoc_2500tweets_2019-01-01.pkl')
    aoc_df = aoc_df[['text', 'username']]
    trump_df = pd.read_pickle('data/trump_2500tweets_2019-01-01.pkl')
    trump_df = trump_df[['text', 'username']]
    df = pd.concat([aoc_df, trump_df])

    # create dictionary of distinct words from list of tweets (to use for bag of words)
    tweet_dictionary = create_dictionary(df['text'])

    print(tweet_dictionary)
    print('length of dictionary: ', len(tweet_dictionary))

    # generate the 0-1 feature matrix for bag of words
    feature_matrix = generate_feature_matrix(df['text'], tweet_dictionary)

    print(feature_matrix)

    # TODO: use feature matrix to train a model

    X = np.array(feature_matrix)
    Y = np.array(df['username'])

    Xtrain = np.concatenate([X[0:1250], X[2500:3750]])
    Ytrain = np.concatenate([Y[0:1250], Y[2500:3750]])
    Xtest = np.concatenate([X[1250:2500],X[3750:]]) 
    Ytest = np.concatenate([Y[1250:2500], Y[3750:]])

    model = LinearSVC()
    model.fit(Xtrain, Ytrain)
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Test accuracy:", model.score(Xtest, Ytest))


if __name__=="__main__":
    main()