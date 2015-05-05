__author__ = 'newuser'

def split(line):
    "Helper function to split a line by tabs"
    return line.rstrip('\n').split('\t')

# read the tsv vile
rows = []
with open("/Users/newuser/Desktop/CS410/output.txt") as tsv:
    # line = tsv.readline()
    # headers = split(line)

    for line in tsv.readlines():
        row = split(line)
        # assert len(row) == len(headers)
        rows.append(row)
tsv.close()
import pandas as pd
data = pd.DataFrame(data=rows)

row2s=[]
with open("/Users/newuser/Downloads/ODPtweets-Mar17-29.tsv") as file2:
    for line in file2.readlines():
        row = split(line)
        row2s.append(row)
data2=pd.DataFrame(data=row2s)
file2.close()
# output=open("/Users/newuser/Desktop/CS410/test.txt",'w')
sb=pd.merge(data,data2,how='inner')
sb.to_csv('/Users/newuser/Desktop/CS410/test.txt', sep='\t')

