__author__ = 'newuser'

import nltk
import re

file2=open('/Users/newuser/Desktop/CS410/input3.txt','r')
file3=open('/Users/newuser/Desktop/CS410/processraw.txt','w')

def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                  if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].most_common(3)) for tag in cfd.conditions())

    # 2. Remove non-letters
def cleandata(review_text):
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))

for line in file2.readline():
    newline=cleandata(line)
    word_token=nltk.word_tokenize(newline)
    word_tags=nltk.pos_tag(word_token)
    noun_get=[a[1] for (a,b) in word_tags if b[1]=='NOUN']

    tagdict = findtags('NN', word_tags)
    for tag in sorted(tagdict):
        print(tag, tagdict[tag])