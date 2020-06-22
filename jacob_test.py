import twitter_scraper
import langdetect
import GetOldTweets3 as got

for tweet in twitter_scraper.get_tweets('#animalcrossing', pages=1):
    english = True
    for hashtag in tweet['entries']['hashtags']:
        if langdetect.detect(hashtag) != 'tl' and langdetect.detect(hashtag) != 'en':
            english = False
            break
    #if langdetect.detect(tweet['text']) == 'en' and english:
        #print(tweet['text'])

tweet_criteria = got.manager.TweetCriteria().setQuerySearch('#stanloona').setTopTweets(True).setMaxTweets(10).setEmoji('unicode')
tweets = got.manager.TweetManager.getTweets(tweet_criteria)
for tweet in tweets: 
    print(tweet.text)