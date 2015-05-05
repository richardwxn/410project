__author__ = 'newuser'
import twitter
import numpy as np
import threading
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
import pandas.io.parsers

# global data
# data=pandas.io.parsers.read_csv("/Users/newuser/Downloads/data.1.1.tsv",sep='\t')
# pd.DataFrame.from_csv
data = pd.DataFrame(data=rows)
file2=open('/Users/newuser/Desktop/CS410/input2.txt','w')
# print(data[4][1])
# userid = numpy.array(data[2])
# for index in range(len(data[1])):

def insertdata(index):
            i=index*10
            # nima=api.GetUser(screen_name=','.join(data[3][i:10*(index+1)]))

            # nima=api.GetStatus(id=','.join(data[2][i:10*(index+1)]))
            # print(np.array(data[2][i:10*(index+1)]))
            nima=api.GetStatus(id=data[2][i:10*(index+1)])
            print(str(nima.text))
            # file2.write(str(nima.id)+"\t")
            # file2.write(data[1][index]+"\n")

def calculate():
    for i in xrange(len(data[1])/10+1):
            if i==0:
                continue
            try:
                insertdata(i)
            except twitter.TwitterError as cao:
                print(str(cao))
                continue

calculate()
# class myThread(threading.Thread):
#     def __init__(self, threadID):
#         threading.Thread.__init__(self)
#         self.threadID=threadID
#     def run(self):
#         calculate()
#
# thread1=myThread(1)
# thread2=myThread(2)
# thread1.start()
# thread2.start()
# thread1.join()
# thread2.join()
# t1 = threading.Thread(target=calculate, args=(5,))
# t2 = threading.Thread(target=calculate, args=(8,))
# t1.start()
# t2.start()
# t1.join()
# t2.join()


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
# part1.to_csvc
# part2.to_csv('/Users/newuser/Downloads/data.1.2.tsv', sep='\t')
# part3.to_csv('/Users/newuser/Downloads/data.1.3.tsv', sep='\t')
# part4.to_csv('/Users/newuser/Downloads/data.1.4.tsv', sep='\t')