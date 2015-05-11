# -*- coding: utf-8 -*-

import unittest

import numpy as np

from tfidf.tfidf import *
from tfidf.featsel import *
from tfidf.smoothing import *
from tfidf.preprocessing import *

doc1 = """C'est un trou de verdure où chante une rivière,
Accrochant follement aux herbes des haillons
D'argent ; où le soleil, de la montagne fière,
Luit : c'est un petit val qui mousse de rayons."""

doc2 = """Un soldat jeune, bouche ouverte, tête nue,
Et la nuque baignant dans le frais cresson bleu,
Dort ; il est étendu dans l'herbe, sous la nue,
Pâle dans son lit vert où la lumière pleut."""

doc3 = """Les pieds dans les glaïeuls, il dort. Souriant comme
Sourirait un enfant malade, il fait un somme :
Nature, berce-le chaudement : il a froid."""

doc4 = """Les parfums ne font pas frissonner sa narine ;
Il dort dans le soleil, la main sur sa poitrine,
Tranquille. Il a deux trous rouges au côté droit."""

dataset = map(lambda x: strip_punctuation2(x.lower()), [doc1, doc2, doc3, doc4])
# doc1 and doc2 have class 0, doc3 and doc4 avec class 1
classes = np.array([[1,0],[1,0],[0,1],[0,1]])

class tfidfTest(unittest.TestCase):

    def testTokenize(self):
        self.assertEquals(tokenize("salut les amis"), 
                          ["salut", "les", "amis"])

        self.assertEquals(tokenize("salut  les   amis "), 
                          ["salut", "les", "amis"])

        self.assertEquals(tokenize("Salut  LES   amis !"), 
                          ["Salut", "LES", "amis", "!"])

    def testTermCounts(self):
        term_counts, vocab = tc(dataset)
        self.assertEquals(len(term_counts), len(dataset))
        for i in range(len(dataset)):
            # the len of the documents should be equal to the sum of word counts
            self.assertEquals(len(tokenize(dataset[i])),
                              sum(term_counts[i].values()))

        self.assertEquals(term_counts[0]["la"], 1)
        self.assertEquals(term_counts[1]["la"], 3)
        self.assertRaises(KeyError, term_counts[2].__getitem__, "la")
        self.assertEquals(term_counts[3]["la"], 1)

    def testTermFrequencies(self):
        term_counts, vocab = tc(dataset)
        term_frequencies = tf_from_tc(term_counts)
        for doc in term_frequencies:
            self.assertAlmostEquals(sum(doc.values()), 1.0)

        self.assertTrue(term_frequencies[0]["la"] > 0)
        self.assertTrue(term_frequencies[1]["la"] > 0)
        self.assertRaises(KeyError, term_frequencies[2].__getitem__, "la")
        self.assertTrue(term_frequencies[3]["la"] > 0)

    def testInvertDocumentCounts(self):
        term_counts, vocab = tc(dataset)
        inv_doc_counts = idc_from_tc(term_counts)
        self.assertEquals(len(vocab), len(inv_doc_counts))
        self.assertEquals(inv_doc_counts["la"], 3) 

    def testInvertDocumentFrequencies(self):
        term_counts, vocab = tc(dataset)
        inv_doc_freq = idf_from_tc(term_counts)
        self.assertEquals(len(vocab), len(inv_doc_freq))
        self.assertTrue(inv_doc_freq["la"] > 0)
        to_vector(inv_doc_freq, vocab)

    def testTFIDFDict(self):
        td, v = tfidf(dataset).as_dict()
        self.assertTrue(td[0]["la"] > 0)
        self.assertTrue(td[1]["la"] > 0)
        self.assertRaises(KeyError, td[2].__getitem__, "la")
        self.assertTrue(td[3]["la"] > 0)

    def testTFIDFArray(self):
        td, v = tfidf(dataset).as_array()

class FeatureSelectionTest(unittest.TestCase):

    def setUp(self):
        tfdict, self.v = tc(dataset)
        self.td = to_sparse_matrix(tfdict, self.v).toarray()

    def testGetCounts(self):
        N11,N01,N10,N00 = get_counts(self.td, classes, 0, 0) # t0 = "a"
        # number of documents containing "a" and in class c0
        self.assertEquals(N11,0)
        # number of documents not containing "a" and in class c0
        self.assertEquals(N01,2)
        # number of documents containing "a" and not in class c0
        self.assertEquals(N10,2)
        # number of documents not containing "a" and not in class c0
        self.assertEquals(N00,0)

        N11,N01,N10,N00 = get_counts(self.td, classes, 36, 1) # t36 = "la"
        # number of documents containing "la" and in class c1
        self.assertEquals(N11,1)
        # number of documents not containing "la" and in class c1
        self.assertEquals(N01,1)
        # number of documents containing "la" and not in class c1
        self.assertEquals(N10,2)
        # number of documents not containing "la" and not in class c1
        self.assertEquals(N00,0)

    def testMutualInformation(self):
        mutual_information(self.td, classes)

    def testChi2(self):
        chi2(self.td, classes)

    def testSelectAvg(self):
        A = chi2(self.td, classes)
        newtd1, newvocab = select_avg(self.td, self.v, A, K=10)
        newtd2 = replace_vocab(self.td, self.v, newvocab)

        for newtd in (newtd1, newtd2):
            self.assertEquals(newtd.shape[1], self.td.shape[1])
            self.assertEquals(newtd.shape[0], 10)
            for term in newvocab:
                # the term counts should be the same in the original and the new
                # reduced matrix
                self.assertTrue(np.array_equal(self.td[self.v[term],:], 
                                            newtd[newvocab[term],:]))

    def testSelectMax(self):
        A = chi2(self.td, classes)
        newtd1, newvocab = select_max(self.td, self.v, A, K=10)
        newtd2 = replace_vocab(self.td, self.v, newvocab)

        for newtd in (newtd1, newtd2):
            self.assertEquals(newtd.shape[1], self.td.shape[1])
            self.assertEquals(newtd.shape[0], 10)
            for term in newvocab:
                # the term counts should be the same in the original and the new
                # reduced matrix
                self.assertTrue(np.array_equal(self.td[self.v[term],:], 
                                            newtd[newvocab[term],:]))

class SmoothingTest(unittest.TestCase):

    def setUp(self):
        tfdict, self.v = tc(dataset)
        self.td = to_sparse_matrix(tfdict, self.v).toarray()

    def testLaplace(self):
        newtd = laplace(self.td)
        self.assertTrue(np.array_equal(newtd.sum(axis=0).astype(np.float32),
                                       np.array([1.]*self.td.shape[1]).\
                                           astype(np.float32)))

    def testLidstone(self):
        newtd = lidstone(self.td)
        self.assertTrue(np.array_equal(newtd.sum(axis=0).astype(np.float32),
                                       np.array([1.]*self.td.shape[1]).\
                                           astype(np.float32)))


class PreprocessingTest(unittest.TestCase):

    def testStripNumeric(self):
        self.assertEquals(strip_numeric("salut les amis du 59"),
                          "salut les amis du ")

    def testStripShort(self):
        self.assertEquals(strip_short("salut les amis du 59", 3),
                          "salut les amis")

    def testStripTags(self):
        self.assertEquals(strip_tags("<i>Hello</i> <b>World</b>!"), 
                          "Hello World!")

    def testStripMultipleWhitespaces(self):
        self.assertEquals(strip_multiple_whitespaces("salut  les\r\nloulous!"),
                          "salut les loulous!")

    def testStripNonAlphanum(self):
        self.assertEquals(strip_non_alphanum("toto nf-kappa titi"),
                          "toto nf kappa titi")

    def testSplitAlphanum(self):
        self.assertEquals(split_alphanum("toto diet1 titi"), 
                          "toto diet 1 titi")
        self.assertEquals(split_alphanum("toto 1diet titi"), 
                          "toto 1 diet titi")

    def testStripStopwords(self):
        self.assertEquals(remove_stopwords("the world is square"),
                          "world square")
            

if __name__ == "__main__":
    unittest.main()