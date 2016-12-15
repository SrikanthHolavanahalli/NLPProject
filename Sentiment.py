import re
from collections import Counter
from collections import namedtuple
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from collections import defaultdict
from sklearn import cross_validation
from nltk.corpus import stopwords
import enchant
import nltk
import sys
from nltk.stem.wordnet import WordNetLemmatizer

cachedStopWords = stopwords.words("english")
lmtzr = WordNetLemmatizer()
all_data = []
data_bigrams =[]
train_tags = []
sentenceId = {}
d =enchant.Dict("en_US")
ipStr = ""
predictFlag = 0


################################### spell correcter functions ###############################
def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N

def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

################################################################################################



##### CLI input checker ###########
if(len(sys.argv) < 3):
    print("Usage: python3 sentiment.py classifier model [optional text]")
    sys.exit(-1)
elif(len(sys.argv) > 3):
    inputStr = sys.argv[3:]
    for ip in sys.argv[3:]:
        ipStr += ip + " "
    print(ipStr)
    predictFlag = 1

classifier = sys.argv[1]
feature_model = sys.argv[2]
#######################################


##### reading movie reviews ###################
DataDoc= namedtuple('DataDoc', 'tag words')
BiDoc= namedtuple('BiDoc', 'tag bigrams')
with open('train.tsv') as alldata:
    for line in alldata:
        sentId = line.split()[1]
        if  sentId not in sentenceId:
            label=line.split()[-1]
            train_tags.append(label)
            word_list=line.lower().split()[2:-2]
            bigram_list = list(nltk.ngrams(word_list, 2))
            all_data.append(DataDoc(label, word_list))
            data_bigrams.append(BiDoc(label, bigram_list))
            sentenceId[sentId] = 'true'
train_data = all_data[:100] + all_data[1000:1100] + all_data[2000:2100] + all_data[3000:3100] + all_data[4000:4100]
train_bi_data = data_bigrams[:100] + data_bigrams[1000:1100] + data_bigrams[2000:2100] + data_bigrams[3000:3100] + data_bigrams[4000:4100]
################################################



###### building workspace ( unique words )  #######
def get_space(train_data):
    word_space=defaultdict(int)
    for doc in train_data:
        for w in doc.words:
            if w not in cachedStopWords:
                w = lmtzr.lemmatize(w)
                #print(type(d.check(w)))

                if(d.check(w) == True):
                    word_space[w]=len(word_space)
                else:
                    w = correction(w)
                    word_space[w] = len(word_space)
    return word_space

word_space=get_space(train_data)
print (len(word_space))
#####################################################


############## building workspace (bi gram) #####################################
def get_bi_space(train_bi_data):
    bigram_space=defaultdict(int)
    for doc in train_bi_data:
        for pairs in doc.bigrams:
            if pairs[0] not in cachedStopWords and pairs[1] not in cachedStopWords:
                w0 = lmtzr.lemmatize(pairs[0])
                w1 = lmtzr.lemmatize(pairs[1])
                #print(type(d.check(w)))
                '''if(d.check(w) == True):
                    word_space[w]=len(word_space)'''
                w = w0 + ','+ w1
                bigram_space[w]=len(bigram_space)
    return bigram_space

bigram_space=get_bi_space(train_bi_data)
print (len(bigram_space))
##################################################################################


###########  Unigram model  ###################
def get_sparse_vec(data_point, space):
    # create empty vector
    sparse_vec = np.zeros((len(space)))
    for w in set(data_point.words):
        try:
            if w not in cachedStopWords:
                w = lmtzr.lemmatize(w)
                if (d.check(w) == True):
                    sparse_vec[space[w]]=1
                else:
                    w = correction(w)
                    word_space[w] = len(word_space)
        except:
            continue
    return sparse_vec
###################################################



########### Bigram Model #####################################################
def get_bigram_vec(data_point, space):
    # create empty vector
    bigram_vec = np.zeros((len(space)))
    for pairs in set(data_point.bigrams):
        try:
            if pairs[0] not in cachedStopWords and pairs[1] not in cachedStopWords:
                w0 = lmtzr.lemmatize(pairs[0])
                w1 = lmtzr.lemmatize(pairs[1])
                w = w0 + ','+ w1
                '''if (d.check(w) == True):
                    sparse_vec[space[w]]=1'''
                bigram_vec[space[w]]=1 #is the order of bigram important
        except:
            continue
    return bigram_vec
##############################################################################



########### feature model selection ########################################################
if feature_model == 'uni':
    train_vecs = [get_sparse_vec(data_point, word_space) for data_point in train_data]
    train_vecs = np.array(train_vecs)
elif feature_model == 'bi':
    train_bi_vecs = [get_bigram_vec(data_point, bigram_space) for data_point in train_bi_data]
    train_vecs = np.array(train_bi_vecs)
else:
    print("Invalid feature model")
    sys.exit(-1)

train_tags=np.array(train_tags[:100] + train_tags[1000:1100] + train_tags[2000:2100] + train_tags[3000:3100] + train_tags[4000:4100])
##########################################################################################




#################### classifiers ########################################################
print("model" + " " +sys.argv[2])
print("classifier " + sys.argv[1])

if(classifier == 'svm'):
    clf = SVC(C=1, kernel = 'linear', gamma=1, verbose= False, probability=False)
elif(classifier == 'knn'):
    clf = KNeighborsClassifier(n_neighbors=5)
elif(classifier == 'nb'):
    clf = MultinomialNB()
elif(classifier == 'nnet'):
    clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
else:
    print("invalid classifier")
    sys.exit(-1)

clf.fit(train_vecs, train_tags)
print ("\nDone fitting classifier on training data...\n")
###########################################################################################


if(predictFlag == 1):
    print(clf.predict(np.array(sys.argv[3:])))


########### post classification analysis ########################################################
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
##################################################################################################
