# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 16:16:44 2014

Bag of Words or “Bag of n-grams” representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.

Results:

train.vectorized.mat : docs-terms matrix from tokenized train data

test.vectorized.mat : docs-terms matrix from tokenized test data

vectorizer.bin : dumped vectorizer object

*.dat : human-readable texts dumps for analysis

@author: Ning
"""

import os,sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from util import *
from util.log import _logger
from scipy.sparse import *
import cPickle as pickle
from collections import defaultdict
import codecs


def vectorize(tfidf=False,binary=False):
    
    _logger.info("Loding...")
    
    trainX = [r[0] for r in tsv.reader(conv.redirect('train.tokenized.dat'))]
    testX = [r[0] for r in tsv.reader(conv.redirect('test.tokenized.dat'))]
    
    vectorizer = None
    if tfidf:
        vectorizer = TfidfVectorizer
    else:
        vectorizer = CountVectorizer
    
    _logger.info("Fitting and transforming...")
    vectorizer = vectorizer(token_pattern=u'(?u)\\b\\w+\\b',binary=binary, ngram_range = (1, 3))
    trainX = vectorizer.fit_transform(trainX)
    testX = vectorizer.transform(testX)
    
    _logger.info("Dumping binaries...")
    pickle.dump(vectorizer,open("vectorizer.bin",'w'))
    pickle.dump(trainX,open("train.vectorized.mat",'w'))
    pickle.dump(testX,open("test.vectorized.mat",'w'))
    
    schema = vectorizer.get_feature_names()
    codecs.open("schema.dat",'w',encoding='utf-8').write('\n'.join(schema))

    # debug
#    _logger.info("Dumping inversered...")
#    codecs.open("test.vectorized.dat",'w',encoding='utf-8').write( '\n'.join( [(' '.join(i)) for i in vectorizer.inverse_transform(testX)] ) )
#    codecs.open("train.vectorized.dat",'w',encoding='utf-8').write( '\n'.join( [(' '.join(i)) for i in vectorizer.inverse_transform(trainX)] ) )

    trainX = trainX.tocoo(False)
    testX = testX.tocoo(False)
    
    _logger.info("Dumping test.vectorized.dat...")
    with codecs.open("test.vectorized.dat",'w',encoding='utf-8') as fl:
        dc = defaultdict(list)
        for r,c,v in zip(testX.row,testX.col,testX.data):
            dc[r].append( "%s(%s)=%s"%(schema[c],c,v) )
        for i in sorted(dc.keys()):
            fl.write("%s\t%s\n" % (i, " , ".join(list(dc[i])) ))
    
    
    _logger.info("Dumping train.vectorized.dat...")
    with codecs.open("train.vectorized.dat",'w',encoding='utf-8') as fl:
        dc = defaultdict(list)
        for r,c,v in zip(trainX.row,trainX.col,trainX.data):
            dc[r].append( "%s(%s)=%s"%(schema[c],c,v) )
        for i in sorted(dc.keys()):
            fl.write("%s\t%s\n" % (i, " , ".join(list(dc[i])) ))
#    
#    _logger.info("Dumping train.vectorized.dat...")
#    with codecs.open("train.vectorized.dat",'w',encoding='utf-8') as fl:
#        for i,r in enumerate(trainX):
#            fl.write("%s\t" % (i))
#            fl.write(' , '.join( ["%s(%s)=%s"%(schema[c],c,trainX[i,c]) for c in r.nonzero()[1]] ) )
#            fl.write("\n")

if __name__ == "__main__":
    vectorize(True)
