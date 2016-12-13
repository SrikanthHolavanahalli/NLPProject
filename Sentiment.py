from collections import namedtuple
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from collections import defaultdict

from sklearn import cross_validation
from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

cachedStopWords = stopwords.words("english")
lmtzr = WordNetLemmatizer()
all_data = []  
DataDoc= namedtuple('DataDoc', 'tag words')
with open('Tweets.csv') as alldata:
    for line_no, line in enumerate(alldata):
        label=line.split()[0]
        word_list=line.lower().split()[1:]
        all_data.append(DataDoc(label, word_list))
train_data = all_data[:25000]
test_data = all_data[25000:50000]
#print (len(train_data))

train_data=train_data[:1000]+train_data[12500:13500]
test_data=test_data[2000:3000]+test_data[25000:26000]
#print (len(train_data))
#print (len(test_data))

def get_space(train_data):
    word_space=defaultdict(int)
    for doc in train_data:
        for w in doc.words:
            if w not in cachedStopWords:
                w = lmtzr.lemmatize(w)
                word_space[w]=len(word_space)
    return word_space

word_space=get_space(train_data)
#print (len(word_space))

def get_sparse_vec(data_point, space):
    # create empty vector
    sparse_vec = np.zeros((len(space)))
    for w in set(data_point.words):
        try:
            if w not in cachedStopWords:
                w = lmtzr.lemmatize(w)
                sparse_vec[space[w]]=1
        except:
            continue
    return sparse_vec
train_vecs= [get_sparse_vec(data_point, word_space) for data_point in train_data]
test_vecs= [get_sparse_vec(data_point, word_space) for data_point in test_data]

train_tags=[ 1.0 for i in range(1000)] + [ 0.0 for i in range(1000)]
test_tags=[ 1.0 for i in range(1000)] + [ 0.0 for i in range(1000)]

train_vecs=np.array(train_vecs)
train_tags=np.array(train_tags)
#print (train_vecs.shape)

#n_jobs = 2

train_vecs=np.array(train_vecs)
train_tags=np.array(train_tags)

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
print ("precision_score\t", metrics.precision_score(train_tags, predicted))
print ("recall_score\t", metrics.recall_score(train_tags, predicted))
print ("\nclassification_report:\n\n", metrics.classification_report(train_tags, predicted))
print ("\nconfusion_matrix:\n\n", metrics.confusion_matrix(train_tags, predicted))
