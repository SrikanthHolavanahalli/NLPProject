from collections import namedtuple
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from collections import defaultdict

from sklearn import cross_validation
from nltk.corpus import stopwords
import enchant

from nltk.stem.wordnet import WordNetLemmatizer

cachedStopWords = stopwords.words("english")
lmtzr = WordNetLemmatizer()
all_data = []
train_tags = []
sentenceId = {}
d =enchant.Dict("en_US")

DataDoc= namedtuple('DataDoc', 'tag words')
with open('train.tsv') as alldata:
    for line in alldata:
        sentId = line.split()[1]
        if  sentId not in sentenceId:
            label=line.split()[-1]
            train_tags.append(label)
            word_list=line.lower().split()[2:-2]
            all_data.append(DataDoc(label, word_list))
            sentenceId[sentId] = 'true'
train_data = all_data[:500]
#test_data = all_data[500:1000]
#print (len(train_data))

#train_data=train_data[:1000]+train_data[12500:13500]
#test_data=test_data[2000:3000]+test_data[25000:26000]
#print (len(train_data))
#print (len(test_data))

def get_space(train_data):
    word_space=defaultdict(int)
    for doc in train_data:
        for w in doc.words:
            if w not in cachedStopWords:
                w = lmtzr.lemmatize(w)
                #print(type(d.check(w)))
                if(d.check(w) == True):
                    word_space[w]=len(word_space)
    return word_space

word_space=get_space(train_data)
print (len(word_space))
x = 1

def get_sparse_vec(data_point, space):
    # create empty vector
    sparse_vec = np.zeros((len(space)))
    for w in set(data_point.words):
        try:
            if w not in cachedStopWords:
                w = lmtzr.lemmatize(w)
                if (d.check(w) == True):
                    sparse_vec[space[w]]=1
        except:
            continue
    return sparse_vec
train_vecs= [get_sparse_vec(data_point, word_space) for data_point in train_data]
#test_vecs= [get_sparse_vec(data_point, word_space) for data_point in test_data]

#train_tags=[ 1.0 for i in range(250)] + [ 0.0 for i in range(250)]
#test_tags=[ 1.0 for i in range(250)] + [ 0.0 for i in range(250)]

train_vecs=np.array(train_vecs)
train_tags=np.array(train_tags[:500])
#print (train_vecs.shape)

n_jobs = 2

#train_vecs=np.array(train_vecs)
#train_tags=np.array(train_tags)

#the clf variable is changed from SVC to NNET, Naive Bayes, KNN in line 84 and the code was ran once for every algorithm and parameter
print (type(train_tags))
print (type(train_vecs))
clf = SVC(C=1, kernel = 'linear', gamma=1, verbose= False, probability=False)
clf.fit(train_vecs, train_tags)
print ("\nDone fitting classifier on training data...\n")

#------------------------------------------------------------------------------------------
print ("="*50, "\n")
print ("Results with 5-fold cross validation:\n")
print ("="*50, "\n")
#------------------------------------------------------------------------------------------
predicted = cross_validation.cross_val_predict(clf, train_vecs, train_tags, cv=5)
print ("*"*20)
print ("\t accuracy_score\t", metrics.accuracy_score(train_tags, predicted))
print ("*"*20)
print ("precision_score\t", metrics.precision_score(train_tags, predicted,average='macro'))
print ("recall_score\t", metrics.recall_score(train_tags, predicted,average='macro'))
print ("\nclassification_report:\n\n", metrics.classification_report(train_tags, predicted))
print ("\nconfusion_matrix:\n\n", metrics.confusion_matrix(train_tags, predicted))
