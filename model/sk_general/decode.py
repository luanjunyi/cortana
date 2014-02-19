import sys, os, math
import argparse
import cPickle as pickle
from collections import defaultdict
import numpy as np
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from util.log import _logger
from util import *
from feat.terms.term_categorize import term_category

from train import CLFs, Vectorizer

def serv(clf):
    vert = clf.named_steps['vert']
    analyzer = vert.build_analyzer()
    svm = clf.named_steps['clf']
    domains = svm.classes_
    while True:
        query = raw_input('Input your query(must be segmented by SPACE), q to quit:\n').decode('utf-8')
        if query == u'q':
            return
        detail = sorted(zip(domains, clf.decision_function([query])[0]),
                        key = lambda x: -x[1])[:4]
        print 'result:', clf.predict([query])[0], '\n'
        top_domains = set()
        for domain, val in detail:
            print domain, val
            top_domains.add(domain)

        print '%40s\t' % "TERM" + '\t'.join(["%8s" % domain for domain in domains if domain in top_domains])

        tokens = analyzer(query)
        for token in tokens:
            if token not in vert.vocabulary_:
                continue
            idx = vert.vocabulary_[token]
            arr = ['%40s' % token]
            arr.extend(["%8.4f" % svm.coef_[di, idx] for di, domain in enumerate(domains) if domain in top_domains])
            print '\t'.join(arr)
        

def test(test_file_path, clf):
    X, y = load_data(test_file_path)
    size = len(y)

    scores = clf.decision_function(X)
    # y_pred = []
    # for i in xrange(size):
    #     score = scores[i]
    #     detail = sorted(zip(clf.named_steps['clf'].classes_,
    #                         score),
    #                     key = lambda x: -x[1])
    #     if detail[0][1] >= 1.1:
    #         y_pred.append(detail[0][0])
    #     else:
    #         y_pred.append(u'web')

    y_pred = clf.predict(X)
    outfile = open("predicted.dat", 'w')
    for i in range(len(y)):
        sentence, pred, gold = X[i], y_pred[i], y[i]
        outfile.write("%s\t%s\t%s\n" % (sentence.encode('utf-8'), pred.encode('utf-8'), gold.encode('utf-8')))
    _logger.info("accuracy: %f, %d records" % (accuracy_score(y, y_pred),
                                               len(y)))


if __name__ == "__main__":

    cmd = argparse.ArgumentParser()
    cmd.add_argument("--path", help = "path to the test data", default=TEST_FILE_PATH)
    cmd.add_argument("--serv", help = "run as server", dest="as_server", action='store_true')
    cmd.add_argument("--model", help = "path to the pickled model", required=True,
                     choices = ["%s.model" % algo for algo in CLFs.keys()])
    args = cmd.parse_args()

    _logger.info("loading model from %s" % args.model)
    clf = pickle.load(open(args.model))

    if args.as_server:
        serv(clf)
        sys.exit(0)

    test(args.path, clf)

