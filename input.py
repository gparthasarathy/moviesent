# -*- coding: utf-8 -*-
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

#traindata
posfilestrain = [ join('train/pos/',f) for f in listdir('train/pos/') if isfile(join('train/pos/',f)) ]
negfilestrain = [ join('train/neg/',f) for f in listdir('train/neg/') if isfile(join('train/neg/',f)) ]

#testdata
posfilestest = [ join('test/pos/',f) for f in listdir('test/pos/') if isfile(join('test/pos/',f)) ]
negfilestest = [ join('test/neg/',f) for f in listdir('test/neg/') if isfile(join('test/neg/',f)) ]

print "Lists generated"
#print len(posfilestrain)
#print posfilestrain[9128]
traindatapos = []
traindataposrating = []
traindatapossent = []
for i in range(len(posfilestrain)):
    if i%100 == 0:
        print i
    #print 'a'
    f = open(posfilestrain[i])
    #print 'b'
    traindatapos.append(f.read())
    #print 'c'
    traindataposrating.append(int(posfilestrain[i].split('_')[1].split('.')[0]))
    traindatapossent.append(1)
    f.close()

print "train pos done"

traindataneg = []
traindatanegrating = []
traindatanegsent = []
for i in range(len(negfilestrain)):
    if i%100 == 0:
        print i
    f = open(negfilestrain[i])
    traindataneg.append(f.read())
    traindatanegrating.append(int(negfilestrain[i].split('_')[1].split('.')[0]))
    traindatanegsent.append(0)
    f.close()

print "Train neg done"

testdatapos = []
testdataposrating = []
testdatapossent = []
for i in range(len(posfilestest)):
    if i%100 == 0:
        print i
    f = open(posfilestest[i])
    testdatapos.append(f.read())
    testdataposrating.append(int(posfilestest[i].split('_')[1].split('.')[0]))
    testdatapossent.append(1)
    f.close()

print "Test pos done"

testdataneg = []
testdatanegrating = []
testdatanegsent = []
for i in range(len(negfilestest)):
    if i%100 == 0:
        print i
    f = open(negfilestest[i])
    testdataneg.append(f.read())
    testdatanegrating.append(int(negfilestest[i].split('_')[1].split('.')[0]))
    testdatanegsent.append(0)
    f.close()

print "Test neg done"
    
print len(traindatapos)
print len(traindataneg)
print len(testdatapos)
print len(testdataneg)

f1 = open('traindatapos','w')
f2 = open('traindataposrating','w')
f3 = open('traindatapossent','w')

g1 = open('traindataneg','w')
g2 = open('traindatanegrating','w')
g3 = open('traindatanegsent','w')

h1 = open('testdatapos','w')
h2 = open('testdataposrating','w')
h3 = open('testdatapossent','w')

i1 = open('testdataneg','w')
i2 = open('testdatanegrating','w')
i3 = open('testdatanegsent','w')

pickle.dump(traindatapos,f1)
pickle.dump(traindataposrating,f2)
pickle.dump(traindatapossent,f3)

pickle.dump(traindataneg,g1)
pickle.dump(traindatanegrating,g2)
pickle.dump(traindatanegsent,g3)

pickle.dump(testdatapos,h1)
pickle.dump(testdataposrating,h2)
pickle.dump(testdatapossent,h3)

pickle.dump(testdataneg,i1)
pickle.dump(testdatanegrating,i2)
pickle.dump(testdatanegsent,i3)


#tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
        #analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

#traindata = traindatapos + traindataneg

#print "Fitting train data in tfv"

#tfv.fit(traindata)

#print "Transforming train data"
#trainnew = tfv.transform(traindata)

#print len(trainnew)