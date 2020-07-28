 # Dana and Jacob's Twitter ML Project!!

**Goal: classify the sentiment (mood) of tweets scraped from twitter**

## General TODO List

1. get familiar with twitter scraper library in python

2. gather data!! (we need LOTS of tweets)

3. Figure out how we will label the data

4. identify NLP libs to use in order to determine which features of these tweets we want to use to train/test our model. (this will require playing around with some NLP libraries!)

5. try out different classification models that have been trained on our data

6. choose the best classification model based on success metrics (accuracy, precision, sensitivity, etc.)

## Twitter Scraper in Python

We need a library that will allow us to pull thousands and thousands of tweets. We need a lot of data to train an accurate model! Here are some resources to help us figure that out, but we need to do some more research:

[Tweepy vs GetOldTweets3](https://towardsdatascience.com/how-to-scrape-tweets-from-twitter-59287e20f0f1)

[twitter-scraper](https://pypi.org/project/twitter-scraper/)

[GetOldTweets3 documentation](https://github.com/Mottl/GetOldTweets3)

`tweet_scraper.py` contains the code used for creating our datasets, which we store as pickle files in the `data/` folder.

## Predicting author of tweet using Bag of Words (WIP)

see `bag_of_words.py` file. Be sure to run

`python -m spacy download en_core_web_sm`

in your terminal before running this file.

## Predicting author of a given tweet (WIP)

See `train_textcat.py`, a copy of an example [text categorizer](https://spacy.io/usage/training#textcat) from the spaCy documentation. Look into spaCy's [models](https://spacy.io/usage/training) that can be trained/saved/loaded for our task.

To run `train_textcat.py`:

1. `python -m spacy download en_core_web_sm`

2. `python train_textcat.py -m en_core_web_sm`

Another idea: we could create [word vectors](https://spacy.io/usage/vectors-similarity) and use a more traditional classifier?? (such as decision tree, SVM, bagging/boosting, etc).


## Accuracy results

86.2% using all 10000+ words with C=0.1
87.1% using most important 4000 words with C=0.1
87.4% using most important 2000 words with C=0.1
89.3% using most important 1000 words with C=0.1