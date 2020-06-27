import pandas as pd
import spacy

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from string import punctuation
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups


def clean_tweet(tweet):
    '''
    Removes stopwords/punctuation/numbers.
    Returns the clean tweet as a list of tokens.
    '''
    # remove punctuation + cast to lowercase
    tweet = tweet.lower()
    for c in punctuation:
        tweet = tweet.replace(c, "")

    # remove stopwords/numbers
    nlp = English()
    doc = nlp(tweet)

    clean_tokens = []
    for token in doc:
        if token.is_stop == False and str(token).isalpha() == True:
            clean_tokens.append(token)

    print(clean_tokens)


def main():
    # read data from pickle files
    aoc_df = pd.read_pickle('data/aoc_2500tweets_2019-01-01.pkl')
    trump_df = pd.read_pickle('data/trump_2500tweets_2019-01-01.pkl')

    # nlp = spacy.load("en_core_web_sm")
    # for doc in nlp.pipe(aoc_df['text']):
    #     print(doc)
    #     print([(ent.text, ent.label_) for ent in doc.ents])
    #     exit(1)

    for index, row in trump_df.iterrows():
        print(row['username'])
        exit(1)


if __name__=="__main__":
    main()