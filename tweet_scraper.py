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
    date_since = '2019-01-01'
    num_tweets = 2500

    trump_tweets = get_tweets('realDonaldTrump', date_since, num_tweets)

    print('--------------------------------TRUMP TWEETS PREVIEW---------------------------------')
    print(trump_tweets.head(10))

    print('\n\n')

    aoc_tweets = get_tweets('AOC', date_since, num_tweets)

    print('--------------------------------AOC TWEETS PREVIEW---------------------------------')
    print(aoc_tweets.head(10))


    print('num trump tweets: ', trump_tweets.shape[0])
    print('num AOC tweets: ', aoc_tweets.shape[0])

    print('saving dataframes to pkl files...')
    trump_df_filename = 'data/trump_' + str(num_tweets) + 'tweets_' + date_since + '.pkl'
    trump_tweets.to_pickle(trump_df_filename)

    aoc_df_filename = 'data/aoc_' + str(num_tweets) + 'tweets_' + date_since + '.pkl'
    aoc_tweets.to_pickle(aoc_df_filename)
    print('done')



if __name__ == '__main__':
    main()

