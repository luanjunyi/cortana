# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 17:01:42 2014

1vs1 is almost as same performence as 1vsR. But dramatically more expensive time consuming
due to >= O(N^2) where N is number of trainning samples in libsvm's implementation

@author: Ning
"""

from sklearn.svm import (SVC,LinearSVC)
import cPickle as pickle
from util import *
from util.log import _logger
import codecs
import argparse

def linear_train(modelfile,trainmatfile,vs='1vsR',C=1,regularize='l2',gridsearch=False):
    ""

    _logger.info("linear_train : %s " % (modelfile))
    _logger.info("Loading...")

    trainX = pickle.load(open(conv.redirect(trainmatfile)))
    trainy = [r[1] for r in tsv.reader(conv.redirect("data|train.dat"))]

    # Optimation
    trainX = trainX.tocsr(False)
    
    _logger.info("Training...")
    if vs == '1vsR':
        if regularize == 'l1':
            clf = LinearSVC(loss='l2',penalty='l1',dual=False,C=C)
        else:
            clf = LinearSVC(loss='l1',penalty='l2',dual=True,C=C)
    elif vs == '1vs1':
        clf = SVC(kernel='linear')
    else:
        raise "Not supported"
        
    clf.fit(trainX,trainy)
    
    _logger.info("Dumping to %s" % (modelfile))
    
    pickle.dump(clf,open(modelfile,'w'))
    
    return clf


def test(modelfile,testmatfile,outfile):
    ""
    
    clf = pickle.load(open(modelfile))
    
    testX = pickle.load(open(conv.redirect(testmatfile)))
    testy = [r[1] for r in tsv.reader(conv.redirect("data|test.dat"))]
    
    _logger.info("Testing...")
    predicts = clf.predict(testX)
    
    with codecs.open(outfile,'w',encoding='utf-8') as fl:
        for src,p in zip(tsv.reader(conv.redirect("data|test.dat")),predicts):
            fl.write( "%s\t%s\t%s\n" % (src[0],p, src[1]) )
    
if __name__ == "__main__":

    cmd = argparse.ArgumentParser()
    cmd.add_argument("--input", help="path of the training data",default="bow|train.vectorized.mat")
    cmd.add_argument("--output", help="path of the output",default="linear_1vsR_l1.predicted.dat")
    cmd.add_argument("--test", help="path of the test data",default="bow|test.vectorized.mat")
    cmd.add_argument("--regularize", help="regularition",default="l2")
    cmd.add_argument("--C", help="alpha of discounting", type=float, default=1)
    cmd.add_argument("--vs", help="enable vs", default='1vsR')
    cmd.add_argument("--gridsearch", help="enable gridsearch", type=bool, default=False)
    
    args = cmd.parse_args()

    #linear_train("linear_1vs1_l1.model","bow|train.vectorized.mat",vs='1vs1',regularize='l1')
    #test("linear_1vs1_l1.model","bow|test.vectorized.mat","linear_1vs1_l1.predicted.dat")
    
    linear_train("svm.model",trainmatfile=args.input,vs=args.vs,regularize=args.regularize,gridsearch=args.gridsearch,C=args.C)
    test("svm.model",args.test,args.output)
    
