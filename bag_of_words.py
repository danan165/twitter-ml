import pandas as pd
import spacy
import numpy as np
import pickle

from string import punctuation
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold


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


def cv_performance(clf, X, y, k=5, metric='accuracy'):
    '''
    returns cross validation performance of clf on given data.
    '''
    # performance of model on each fold
    scores = []

    skf = StratifiedKFold(n_splits=k)
    skf.get_n_splits(X, y)
    # StratifiedKFold(n_splits=k, random_state=None, shuffle=False)

    # calculate accuracy performance of model in each fold
    for train_index, test_index in skf.split(X, y):
        print('***new SK Fold***')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('fitting...')
        clf = clf.fit(X_train, y_train)
        print('predicting...')
        y_pred = clf.predict(X_test)

        score = metrics.accuracy_score(y_test, y_pred)
        scores.append(score)
        print('score: ', score)
       
    # return the average performance across all fold splits.
    return np.array(scores).mean()


def select_SVC_param_linear(X, y, c_vals=[]):
    '''
    use cross validation to select the best param val for 'C'.
    '''
    # store avg performance for each C value
    avg_performances = []

    for c_val in c_vals:
        print('c val: ', c_val)
        clf = SVC(C=c_val, kernel='linear')
        avg_performance = cv_performance(clf, X, y)
        avg_performances.append(avg_performance)
        print('performance: ', avg_performance, '\n')
    
    print('best avg performance: ', np.max(avg_performances))
    return c_vals[np.argmax(avg_performances)]


def save_most_important_words(X_train, Y_train, tweet_dictionary):
    '''
    Find top ~4000 most important words used to classify tweets with the SVM.
    Save these words as a dictionary with the format {'word': unique_index}, where 
    unique_index is a unique integer. 
    '''

    clf = SVC(C=0.1, kernel='linear')
    clf.fit(X_train, Y_train)
    theta = clf.coef_
    max_coefs = []
    min_coefs = []
    for i in range(2000):
        max_coefs.append(np.max(theta))
        theta = np.delete(theta, np.argmax(theta))
        min_coefs.append(np.min(theta))
        theta = np.delete(theta, np.argmin(theta))

    pos_words = []
    for max_coef in max_coefs:
        idx = np.where(clf.coef_ == max_coef)[1][0]
        for key, value in tweet_dictionary.items():
            if idx == value:
                pos_words.append(key)

    neg_words = []
    for min_coef in min_coefs:
        idx = np.where(clf.coef_ == min_coef)[1][0]
        for key, value in tweet_dictionary.items():
            if idx == value:
                neg_words.append(key)

    top_words = np.concatenate((pos_words, neg_words), axis=0)

    top_words_dict = dict()
    unique_idx = 0

    for word in top_words:
        if word not in top_words_dict:
            top_words_dict[word] = unique_idx
            unique_idx += 1

    with open('top_4000_words_trump_AOC.pkl', 'wb') as f:
        pickle.dump(top_words_dict, f, pickle.HIGHEST_PROTOCOL)


def main():
    # read data from pickle files
    print('loading data...')
    aoc_df = pd.read_pickle('data/aoc_2500tweets_2019-01-01.pkl')
    aoc_df = aoc_df[['text', 'username']]
    trump_df = pd.read_pickle('data/trump_2500tweets_2019-01-01.pkl')
    trump_df = trump_df[['text', 'username']]
    df = pd.concat([aoc_df, trump_df])

    # randomly shuffle the rows in the data
    print('shuffling data...')
    df = df.sample(frac=1).reset_index(drop=True)

    # create dictionary of distinct words from list of tweets (to use for bag of words)
    print('creating dictionary...')
    # tweet_dictionary = create_dictionary(df['text'])

    with open('top_4000_words_trump_AOC.pkl', 'rb') as f:
        tweet_dictionary = pickle.load(f)

    # generate the 0-1 feature matrix for bag of words
    print('generating feature matrix...')
    feature_matrix = generate_feature_matrix(df['text'], tweet_dictionary)

    # split data into 75/25 with balanced labels (train/test)
    print('splitting train/test data...')
    X = feature_matrix
    y = np.array(df['username'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    # code used to generate the top 4000 words file:
    # save_most_important_words(X_train, y_train, tweet_dictionary)
    # exit(1)

    # SVM hyperparameter selection
    print('selecting hyperparameters for SVM...')
    c_vals = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
    c_val = select_SVC_param_linear(X_train, y_train, c_vals=c_vals)
    print('selected C=', c_val)

    # train SVM using selected hyperparam
    print('training SVM...')
    model = SVC(C=c_val, kernel='linear')
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    print("Linear SVM Accuracy:", score) # 86.2% using all 10000+ words with C=0.1


if __name__=="__main__":
    main()