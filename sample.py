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
sentenceId = {}
DataDoc= namedtuple('DataDoc', 'tag words')
with open('train.tsv') as alldata:
    for line in alldata:
        if  line.split()[2] not in sentenceId:
            label=line.split()[-1]
            word_list=line.lower().split()[2:-2]
            all_data.append(DataDoc(label, word_list))
            sentenceId[line.split()[2]] = 'true'
train_data = all_data[:25000]
test_data = all_data[25000:50000]



