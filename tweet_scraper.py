'''
pick two twitter users, AOC and Donald Trump, and classify whether 
a tweet belongs to either of them
'''
import GetOldTweets3 as got
import pandas as pd


def get_tweets(username, date_since, max_num_tweets):
    '''
    Given username, date string (YYYY-MM-DD) and maximimum number of tweets,
    returns pandas dataframe of tweets.
    '''
    tweetCriteria = got.manager.TweetCriteria().setUsername(username)\
                                            .setSince(date_since)\
                                            .setMaxTweets(max_num_tweets)\
                                            .setEmoji("unicode")
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    df_cols = ['id', 'permalink', 'username', 'to', 'text', 'date',
                'retweets', 'favorites', 'mentions', 'hashtags', 'geo']
    df = pd.DataFrame(columns=df_cols)

    for tweet in tweets:
        tweet_dict = {'id': tweet.id, 'permalink': tweet.permalink, 'username': tweet.username, 'to': tweet.to, 
                        'text': tweet.text, 'date': tweet.date, 'retweets': tweet.retweets, 'favorites': tweet.favorites, 
                        'mentions': tweet.mentions, 'hashtags': tweet.hashtags, 'geo': tweet.geo}
        df = df.append(tweet_dict, ignore_index=True)

    return df


def main():
    trump_tweets = get_tweets('realDonaldTrump', '2019-01-01', 2500)
    aoc_tweets = get_tweets('AOC', '2019-01-01', 2500)

    print('--------------------------------TRUMP TWEETS PREVIEW---------------------------------')
    print(trump_tweets.head(10))

    print('\n\n')

    print('--------------------------------AOC TWEETS PREVIEW---------------------------------')
    print(aoc_tweets.head(10))


    print('num trump tweets: ', trump_tweets.shape[0])
    print('num AOC tweets: ', aoc_tweets.shape[0])


if __name__ == '__main__':
    main()

