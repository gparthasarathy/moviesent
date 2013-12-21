# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:21:40 2013
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.decomposition import TruncatedSVD as SVD
import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestClassifier as RFC

f1 = open('traindatapos','r')
f2 = open('traindataposrating','r')
f3 = open('traindatapossent','r')

g1 = open('traindataneg','r')
g2 = open('traindatanegrating','r')
g3 = open('traindatanegsent','r')

h1 = open('testdatapos','r')
h2 = open('testdataposrating','r')
h3 = open('testdatapossent','r')

i1 = open('testdataneg','r')
i2 = open('testdatanegrating','r')
i3 = open('testdatanegsent','r')


traindatapos = pickle.load(f1)
traindataposrating = pickle.load(f2)
traindatapossent = pickle.load(f3)

traindataneg = pickle.load(g1)
traindatanegrating = pickle.load(g2)
traindatanegsent = pickle.load(g3)

testdatapos = pickle.load(h1)
testdataposrating = pickle.load(h2)
testdatapossent = pickle.load(h3)

testdataneg = pickle.load(i1)
testdatanegrating = pickle.load(i2)
testdatanegsent = pickle.load(i3)

tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

traindata = traindatapos + traindataneg
testdata = testdatapos + testdataneg
totaldata = traindata + testdata

trainsent = traindatapossent + traindatanegsent
testsent = testdatapossent + testdatanegsent


lentrain = len(traindata)

print "Fitting train data in tfv"

tfv.fit(totaldata)

print "Transforming train data"
totalnew = tfv.transform(totaldata)

#print trainnew.shape

print "Performing SVD on the new data"

svd = SVD(n_components=250)
totalsvd = svd.fit_transform(totalnew)

trainsvd = totalsvd[:lentrain]
testsvd = totalsvd[lentrain:]

#trainnew = totalnew[:lentrain]
#testnew = totalnew[lentrain:]

rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)

#rd = RFC(n_estimators = 100)

print "Fitting the logistic regressor"
rd.fit(trainsvd,trainsent)
#rd.fit(trainnew,trainsent)

print "classifying the test data"
print rd.score(testsvd,testsent)
#print rd.score(testnew,testsent)