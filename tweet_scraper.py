# ------------GET OLD TWEETS 3 -----------------------

import GetOldTweets3 as got

# can get 10000 tweets using query search! could do this iteratively
# a lot of different query params...customizable!!!

tweetCriteria = got.manager.TweetCriteria().setQuerySearch('europe refugees')\
                                           .setMaxTweets(10000)
tweets = got.manager.TweetManager.getTweets(tweetCriteria)
print(len(tweets))

exit(1)





# ----------TWITTER SCRAPER----------------------


from twitter_scraper import get_tweets

tweets = set()
total_num_tweets = 0

# must query by username or hashtag...may not be sufficiently random set of tweets
for tweet in get_tweets('#happy', pages=25):
    tweets.add(tweet['tweetId'])
    total_num_tweets += 1

# for tweet in tweets:
#     print('------------------------')
#     print('****TEXT****')
#     print(tweet['text'])
#     print('\n')

#     print('****TIME****')
#     print(tweet['time'])
#     print('\n')

#     print('****USERNAME****')
#     print(tweet['username'])
#     print('\n')

#     print('------------------------')

print('total num tweets: ', total_num_tweets)
print('num unique tweets: ', len(tweets))




