__author__ = 'newuser'
import twitter
api = twitter.Api(consumer_key='fZhoY2TMHIS9ODgw5HjrdppPB',
                      consumer_secret='dABHXhNz2qA551hcMBcRgDZgRSy0PZ9otgW4BjEuDLYWmiwuZX',
                      access_token_key='623527276-cstgcL5iD8TSrTW7i9MtcAmaTggZBQHzE2F3Rc51',
                      access_token_secret='NiyoFAYg5YioOrMwemhEaE7LN6UhrSm96JovuHwv7cyep')
print(api.VerifyCredentials())
# statuses = api.GetUserTimeline(screen_name="Azuremagazine")
# print [s.text for s in statuses]

def split(line):
    "Helper function to split a line by tabs"
    return line.rstrip('\n').split('\t')

# read the tsv vile
rows = []
with open("/Users/newuser/Downloads/data.1.4.tsv") as tsv:
    # line = tsv.readline()
    # headers = split(line)

    for line in tsv.readlines():
        row = split(line)
        # assert len(row) == len(headers)
        rows.append(row)

# create a dataframe object. pandas is a python
# library for data analysis, kind of like R.
import pandas as pd
data = pd.DataFrame(data=rows)
file2=open('/Users/newuser/Desktop/CS410/input2.txt','w')
# print(data[4][1])

for index in range(len(data[1])):

            if index==0:
                continue
            try:
                nima=api.GetUser(screen_name=data[2][index])
            except twitter.TwitterError:
                continue
            file2.write(str(nima.id)+"\t")
            file2.write(data[1][index]+"\n")


    # bleh = str(inst)
    # if bleh == "[{u'message': u'Sorry, that page does not exist', u'code': 34}]":
    #     print 'invalid username'
    #     checkedList.append(n)
    #     continue
    # elif bleh == 'Not authorized':
    #     print 'locked account'
    #     checkedList.append(n)
    #     continue
        # print(index)

# part1 = data[:len(data)/4]
# part2 = data[len(data)/4:len(data)/2]
# part3 = data[len(data)/2:len(data)/4*3]
# part4 = data[len(data)/4*3:]
# # make sure we didn't forget any rows
# assert len(part1) + len(part2)+len(part3)+len(part4)== len(data)
#
# # save to excel format
# part1.to_csv('/Users/newuser/Downloads/data.1.1.tsv', sep='\t')
# part2.to_csv('/Users/newuser/Downloads/data.1.2.tsv', sep='\t')
# part3.to_csv('/Users/newuser/Downloads/data.1.3.tsv', sep='\t')
# part4.to_csv('/Users/newuser/Downloads/data.1.4.tsv', sep='\t')