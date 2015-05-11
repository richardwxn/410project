__author__ = 'newuser'

from TwitterSearch import *
import time
file2=open('/Users/newuser/Desktop/CS410/input3.txt','a')
try:
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
    tso2=TwitterSearchOrder()
    tso.set_keywords(['stock'])
    tso2.set_keywords(['economy'])
    tso.set_language('en')# let's define all words we would like to have a look for # we want to see German tweets only
    # tso.set_result_type('popular')
    tso2.set_language('en')
    tso.set_include_entities(False)
    tso2.set_include_entities(False)# and don't give us all those entity information
    print('sb')
    # it's about time to create a TwitterSearch object with our secret tokens
    ts = TwitterSearch(
                    consumer_key='PbbX32OF5kCVpn42yUzojlUD8',
                      consumer_secret='6MBXkjmglrNa0v3Ew3bada90R428Kxv8FQbItBiAWyauucTTFi',
                      access_token='623527276-dwIszI9Xt54MG5rtz6cgXE8HAjOeHvL5t9eeGx0u',
                      access_token_secret='lszKhbwSXdjsZoLQ4vwYPfzIqE9fyHlWRy5URiTKX0VFw'
     )

     # this is where the fun actually starts :)
    counter = 0 # rate-limit counter
    sleep_at = 122 # enforce delay after receiving 123 tweets
    sleep_for = 60.5
    sleeptime=0
    s=set()
    for tweet in ts.search_tweets_iterable(tso):
        s.add(tweet['text'])
        counter += 1 # increase counter
        if counter >= sleep_at: # it's time to apply the delay
            sleeptime+=1
            if sleeptime>3:
                break
            counter = 0
            time.sleep(sleep_for)
    for settweet in s:
        file2.write( '%s' % ( settweet.strip('\n').encode('utf-8') )+'\t'+'business'+'\n')
        # file2.write('\n')
    time.sleep(sleep_for)
    counter=0
    sleeptime=0
    sb=set()
    for tweet in ts.search_tweets_iterable(tso2):
        sb.add(tweet['text'])
        counter += 1 # increase counter
        if counter >= sleep_at: # it's time to apply the delay
            sleeptime+=1
            if sleeptime>3:
                break
            counter = 0
            time.sleep(sleep_for)
    for tweet in sb:
        file2.write( '%s' % ( tweet.strip('\n').encode('utf-8') )+'\t'+'business'+'\n')
        # file2.write('\n')

except TwitterSearchException as e: # take care of all those ugly errors if there are some
    print(e)
