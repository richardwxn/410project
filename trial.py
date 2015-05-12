__author__ = 'newuser'

import html
import pprint
import re
import  time
from time import clock
import Cython
# from html.parser import HTMLParser
from HTMLParser import HTMLParser
from lightning.classification import CDClassifier
from lightning.classification import LinearSVC
from lightning.classification import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
# from pyspark import SparkContext
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from time import time
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier

class ReutersParser(HTMLParser):
    """
    ReutersParser subclasses HTMLParser and is used to open the SGML
    files associated with the Reuters-21578 categorised test collection.

    The parser is a generator and will yield a single document at a time.
    Since the data will be chunked on parsing, it is necessary to keep
    some internal state of when tags have been "entered" and "exited".
    Hence the in_body, in_topics and in_topic_d boolean members.
    """
    def __init__(self, encoding='latin-1'):
        """
        Initialise the superclass (HTMLParser) and reset the parser.
        Sets the encoding of the SGML files by default to latin-1.
        """
        # html.parser.HTMLParser.__init__(self)
        HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def _reset(self):
        """
        This is called only on initialisation of the parser class
        and when a new topic-body tuple has been generated. It
        resets all off the state so that a new tuple can be subsequently
        generated.
        """
        self.in_body = False
        self.in_topics = False
        self.in_topic_d = False
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        """
        parse accepts a file descriptor and loads the data in chunks
        in order to minimise memory usage. It then yields new documents
        as they are parsed.
        """
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_starttag(self, tag, attrs):
        """
        This method is used to determine what to do when the parser
        comes across a particular tag of type "tag". In this instance
        we simply set the internal state booleans to True if that particular
        tag has been found.
        """
        if tag == "reuters":
            pass
        elif tag == "body":
            self.in_body = True
        elif tag == "topics":
            self.in_topics = True
        elif tag == "d":
            self.in_topic_d = True

    def handle_endtag(self, tag):
        """
        This method is used to determine what to do when the parser
        finishes with a particular tag of type "tag".

        If the tag is a  tag, then we remove all
        white-space with a regular expression and then append the
        topic-body tuple.

        If the tag is a  or  tag then we simply set
        the internal state to False for these booleans, respectively.

        If the tag is a  tag (found within a  tag), then we
        append the particular topic to the "topics" list and
        finally reset it.
        """
        if tag == "reuters":
            self.body = re.sub(r'\s+', r' ', self.body)
            self.docs.append( (self.topics, self.body) )
            self._reset()
        elif tag == "body":
            self.in_body = False
        elif tag == "topics":
            self.in_topics = False
        elif tag == "d":
            self.in_topic_d = False
            self.topics.append(self.topic_d)
            self.topic_d = ""

    def handle_data(self, data):
        """
        The data is simply appended to the appropriate member state
        for that particular tag, up until the end closing tag appears.
        """
        if self.in_body:
            self.body += data
        elif self.in_topic_d:
            self.topic_d += data


def obtain_topic_tags():
    """
    Open the topic list file and import all of the topic names
    taking care to strip the trailing "\n" from each word.
    """
    topics = open(
        "/Users/newuser/Downloads/reuters21578/all-topics-strings.lc.txt", "r"
    ).readlines()
    topics = [t.strip() for t in topics]
    return topics

def filter_doc_list_through_topics(topics, docs):
    """
    Reads all of the documents and creates a new list of two-tuples
    that contain a single feature entry and the body text, instead of
    a list of topics. It removes all geographic features and only
    retains those documents which have at least one non-geographic
    topic.
    """
    ref_docs = []
    for d in docs:
        if d[0] == [] or d[0] == "":
            continue
        if d[0] in topics:

                d_tup = (d[0], d[1])
                ref_docs.append(d_tup)
    return ref_docs

def create_tfidf_training_data(docs):
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list.

    The function returns both the class label vector (y) and
    the corpus token/feature matrix (X).
    """
    # Create the training data class labels
    y = [d[0] for d in docs]

    # Create the document corpus list
    corpus = [d[1] for d in docs]

    # Create the TF-IDF vectoriser and transform the corpus
    X=corpus
    # vectorizer = TfidfVectorizer(min_df=1)
    #
    # X = vectorizer.fit_transform(corpus)
    return X, y

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm



if __name__ == "__main__":
    # Create the list of Reuters data and create the parser
    # files = ["/Users/newuser/Downloads/reuters21578/reut2-%03d.sgm" % r for r in range(0, 2)]
    start_time=clock()
    file="/Users/newuser/Downloads/news"
    # parser = ReutersParser()

    # Parse the document and force all generated docs into
    # a list so that it can be printed out to the console
    hehe = []
    files=[]
    # for d in parser.parse(open(file, 'rb')):
    i=1
    d=[]
    with open(file,'rb') as f:
        hehe=[line.strip().decode() for line in f.readlines()]
        # for line in f.readlines():
        #     d.append(line)
        #     if(i%8==0):
        #         hehe.append(d)
        #         d=[]
        #     i=i+1
    # print(hehe.__len__())
    # d=[]
    i=1
    j=6
    docs=[]
    # for sb in hehe:
    while(j<=hehe.__len__()-1):
        docs.append([hehe[j],hehe[i]])
        i=i+8
        j=j+8
    topics=['sport', 'business', 'us', 'health', 'sci&Tech', 'world','entertainment']
    # Obtain the topic tags and filter docs through it
    # topics = obtain_topic_tags()

    ref_docs = filter_doc_list_through_topics(topics, docs)
    # Vectorise and TF-IDF transform the corpus
    X, y = create_tfidf_training_data(ref_docs)
    print(clock()-start_time)
     # Create the training-test split of the data
    # Create and train the Support Vector Machine
    # Make an array of predictions on the test set
    # Output the hit-rate and the confusion matrix for each model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(clock()-start_time)
    # svm = train_svm(X_train, y_train)
    # print(clock()-start_time)
    # pred = svm.predict(X_test)
    # print(svm.score(X_test, y_test))
    # print(confusion_matrix(pred, y_test))

    # forest = RandomForestClassifier(n_estimators = 100)
    # forest.fit(X_train, y_train)
    # pred2=forest.predict(X_test)
    # print(forest.score(X_test, y_test))
    # print(confusion_matrix(pred2, y_test))

    from sklearn.grid_search import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
             'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
    }
    text_clf=Pipeline([('vect', CountVectorizer()),('tfidf',TfidfTransformer()),('clf',SGDClassifier(loss='hinge',penalty='l2'))])
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=3)
    gs_clf = gs_clf.fit(X_train, y_train)

    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))





    # clfs = (CDClassifier(loss="squared_hinge",
    #                      penalty="l2",
    #                      max_iter=20,
    #                      random_state=0),
    #
    #         LinearSVC(max_iter=20,
    #                   random_state=0),
    #
    #         SGDClassifier(learning_rate="constant",
    #                       alpha=1e-3,
    #                       max_iter=20,
    #                       random_state=0))
    #
    # for clf in clfs:
    #     print(clf.__class__.__name__)
    #     clf.fit(X_train, y_train)
    #     print(clf.score(X_test, y_test))



    # print("Performing dimensionality reduction using LSA")
    # # t0 = time()
    # # Vectorizer results are normalized, which makes KMeans behave as
    # # spherical k-means for better results. Since LSA/SVD results are
    # # not normalized, we have to redo the normalization.
    # svd = TruncatedSVD()
    # lsa = make_pipeline(svd, Normalizer(copy=False))
    #
    # X = lsa.fit_transform(X)
    #
    # # print("done in %fs" % (time() - t0))
    #
    # explained_variance = svd.explained_variance_ratio_.sum()
    # print("Explained variance of the SVD step: {}%".format(
    #     int(explained_variance * 100)))
    #
    # print()
    # labels=y
    # true_k = np.unique(labels).shape[0]
    # # .shape[0]
    # print(true_k)
    # km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
    #                      init_size=1000, batch_size=1000, verbose=0)
    #
    # print("Clustering sparse data with %s" % km)
    # t0 = time()
    # km.fit(X)
    # print("done in %0.3fs" % (time() - t0))
    # print()
    #
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    # print("Adjusted Rand-Index: %.3f"
    #       % metrics.adjusted_rand_score(labels, km.labels_))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, km.labels_, sample_size=1000))
    #
    # print()
    #
    # # if not (opts.n_components or opts.use_hashing):
    # print("Top terms per cluster:")
    # order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    # terms = vectorizer.get_feature_names()
    # for i in range(true_k):
    #     print("Cluster %d:" % i, end='')
    #     for ind in order_centroids[i, :10]:
    #         print(' %s' % terms[ind], end='')
    #     print()





    # reduced_data = PCA(n_components=2).fit_transform(X.toarray())
    # kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
    # kmeans.fit(reduced_data)
    #
    # # Step size of the mesh. Decrease to increase the quality of the VQ.

    # h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
    #

    # # Plot the decision boundary. For that, we will assign a color to each





    # reduced_data=km.fit(X)
    # x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
    # y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #
    # # Obtain labels for each point in mesh. Use last trained model.
    # Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
    # #
    # # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(Z, interpolation='nearest',
    #            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #            cmap=plt.cm.Paired,
    #            aspect='auto', origin='lower')
    #
    # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # # Plot the centroids as a white X
    # centroids = km.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1],
    #             marker='x', s=169, linewidths=3,
    #             color='w', zorder=10)
    # plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
    #           'Centroids are marked with white cross')
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()