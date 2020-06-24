'''
Take ten twitter users, five liberal and five conservative, and classify whether 
a tweet is more liberal or conservative
'''

from tweet_scraper import get_tweets
import twitter_scraper
import GetOldTweets3 as got

def main():
    
    num_tweets = 500
    date_since = '2019-01-01'
    # Popular liberal twitter accounts
    warren_tweets = get_tweets('SenWarren', date_since, num_tweets)
    king_tweets = get_tweets('shaunking', date_since, num_tweets)
    simon_tweets = get_tweets('AoDespair', date_since, num_tweets)
    volsky_tweets = get_tweets('igorvolsky', date_since, num_tweets)
    rosen_tweets = get_tweets('jayrosen_nyu', date_since, num_tweets)

    # Popular conservative twitter accounts
    malkin_tweets = get_tweets('michellemalkin', date_since, num_tweets)
    rove_tweets = get_tweets('KarlRove', date_since, num_tweets)
    beck_tweets = get_tweets('glennbeck', date_since, num_tweets)
    gingrich_tweets = get_tweets('newtgingrich', date_since, num_tweets)
    johns_tweets = get_tweets('michaeljohns', date_since, num_tweets)


    '''
    print('-------------------------------- Elizabeth Warren Tweets Preview---------------------------------')
    print(warren_tweets.head(10))
    
    print('--------------------------------Shaun King Tweets Preview---------------------------------')
    print(king_tweets.head(10))

    print('--------------------------------David Simon Tweets Preview---------------------------------')
    print(simon_tweets.head(10))

    print('--------------------------------Igor Volsky Tweets Preview---------------------------------')
    print(volsky_tweets.head(10))

    print('--------------------------------Jay Rosen Tweets Preview---------------------------------')
    print(rosen_tweets.head(10))


    print('--------------------------------Michelle Malkin Tweets Preview---------------------------------')
    print(malkin_tweets.head(10))
    
    print('--------------------------------Karl Rove Tweets Preview---------------------------------')
    print(rove_tweets.head(10))

    print('--------------------------------Glenn Beck Tweets Preview---------------------------------')
    print(beck_tweets.head(10))

    print('--------------------------------Newt Gingrich Tweets Preview---------------------------------')
    print(gingrich_tweets.head(10))

    print('--------------------------------Michael Johns Tweets Preview---------------------------------')
    print(johns_tweets.head(10))

    '''

    print('num warren tweets: ', warren_tweets.shape[0])
    print('num king tweets: ', king_tweets.shape[0])
    print('num simon tweets: ', simon_tweets.shape[0])
    print('num volsky tweets: ', volsky_tweets.shape[0])
    print('num rosen tweets: ', rosen_tweets.shape[0])
    print('num malkin tweets: ', malkin_tweets.shape[0])
    print('num rove tweets: ', rove_tweets.shape[0])
    print('num beck tweets: ', beck_tweets.shape[0])
    print('num gingrich tweets: ', gingrich_tweets.shape[0])
    print('num johns tweets: ', johns_tweets.shape[0])

    print('saving dataframes to pkl files...')
    warren_df_filename = 'data/warren_' + str(num_tweets) + 'tweets_' + date_since + '.pkl'
    warren_tweets.to_pickle(warren_df_filename)
    king_df_filename = 'data/king_' + str(num_tweets) + 'tweets_' + date_since + '.pkl'
    king_tweets.to_pickle(king_df_filename)
    simon_df_filename = 'data/simon_' + str(num_tweets) + 'tweets_' + date_since + '.pkl'
    simon_tweets.to_pickle(simon_df_filename)
    volsky_df_filename = 'data/volsky_' + str(num_tweets) + 'tweets_' + date_since + '.pkl'
    volsky_tweets.to_pickle(volsky_df_filename)
    rosen_df_filename = 'data/rosen_' + str(num_tweets) + 'tweets_' + date_since + '.pkl'
    rosen_tweets.to_pickle(rosen_df_filename)
    malkin_df_filename = 'data/malkin_' + str(num_tweets) + 'tweets_' + date_since + '.pkl'
    malkin_tweets.to_pickle(malkin_df_filename)
    rove_df_filename = 'data/rove_' + str(num_tweets) + 'tweets_' + date_since + '.pkl'
    rove_tweets.to_pickle(rove_df_filename)
    beck_df_filename = 'data/beck_' + str(num_tweets) + 'tweets_' + date_since + '.pkl'
    beck_tweets.to_pickle(beck_df_filename)
    gingrich_df_filename = 'data/gingrich_' + str(num_tweets) + 'tweets_' + date_since + '.pkl'
    gingrich_tweets.to_pickle(gingrich_df_filename)
    johns_df_filename = 'data/johns_' + str(num_tweets) + 'tweets_' + date_since + '.pkl'
    johns_tweets.to_pickle(johns_df_filename)
    print('done')


if __name__ == '__main__':
    main()