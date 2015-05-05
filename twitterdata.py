__author__ = 'newuser'

import twitter
api = twitter.Api(consumer_key='fZhoY2TMHIS9ODgw5HjrdppPB',
                      consumer_secret='dABHXhNz2qA551hcMBcRgDZgRSy0PZ9otgW4BjEuDLYWmiwuZX',
                      access_token_key='623527276-cstgcL5iD8TSrTW7i9MtcAmaTggZBQHzE2F3Rc51',
                      access_token_secret='NiyoFAYg5YioOrMwemhEaE7LN6UhrSm96JovuHwv7cyep')
print(api.VerifyCredentials())
# statuses = api.GetUserTimeline(screen_name="Azuremagazine")
# print [s.text for s in statuses]
sb=api.GetUser(screen_name="Azuremagazine")
print(sb.id)

file=open("",'rb')
