# ------------GET OLD TWEETS 3 -----------------------

import GetOldTweets3 as got

tweetCriteria = got.manager.TweetCriteria().setUsername("barackobama whitehouse")\
                                           .setMaxTweets(2)
tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
print(tweet.text)




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




