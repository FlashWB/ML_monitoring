#coding=utf-8
import numpy as np
import math
import sys
from sys import argv
import time
from sklearn.naive_bayes import GaussianNB  
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from gensim import models

'''
construct vocab
'''

class config(object):
    """docstring for config"""
    def __init__(self):
        super(config, self).__init__()
        self._vectorize = "onehot"# choose onehot, wf, tf, tfidf, word2vec
        #choose GaussianNB,MultinomialNB,BernoulliNB or SVM_SVC_linear SVM_SVC_rbf
        self._model = "SVM_SVC_linear" 
        
        self.dict_length = 0 
        self.batch_size = 2000
        self.train_data_num = 90000 #could be None, means read all data
        self.test_data_num = 10000

        self.epoch = 10

        self.vocab_path = './FEATURE_ORDERED.txt'
        self.train_data_path = './train'
        self.test_data_path = "./test"
        # self.word2vec_path = "./word2vec_model"



def createVocab(path, length):
    vocab = []
#    with open(path, 'r', encoding='utf-8') as f:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines()[:length]:
            vocab.append(line.strip()[0])
    word_dict = dict(zip(vocab, range(len(vocab))))  # zip 对应位置打包，('8',0) index_init = 0
    return word_dict

'''
read  data 
create vector
'''

class Vectorize(object):
    def __init__(self,word_dict):
        super(Vectorize, self).__init__()
        self.word_dict = word_dict
        self.word2vec_dict = models.Word2Vec.load('./word2vec_model2')
    '''
    read data 
    X = ['dsfakjl','adfkj']
    Y = [1,0]

    '''
    def readData(self,path,length):
        X = []
        Y = []
        with open(path, 'r', encoding='utf-8') as t:
            if length == None:
                length = len(t.readlines())
            
            for line in t.readlines()[:length]:
                try:
                    sentence, label = line.strip().split('\t')
                except:
                    continue
                X.append(sentence)
                Y.append(int(label))
        return X,Y
    
    '''
    X,Y = readData(path,length)
    iter = getMinibatch(X,Y,1000,'zeroone')
    for X,Y in iter:
        encode.partial_fit(X,Y)
    '''
    def getMinibatch(self, inputData_X,inputData_Y,batch_size,encode):
        X_batch = []
        Y_batch = []

        for x,y in zip(inputData_X,inputData_Y):
            
            X_batch.append(x)
            Y_batch.append(y)

            if len(X_batch) == batch_size:
                if encode =='onehot':
                    X,Y = self.oneHot(X_batch,Y_batch)
                elif encode == 'wf':
                    X,Y = self.wf(X_batch,Y_batch)
                elif encode == 'tf':
                    X,Y = self.tf(X_batch,Y_batch)
                elif encode == 'tfidf':
                    X,Y = self.tfIdf(X_batch,Y_batch)
                elif encode == 'word2vec':
                    X,Y = self.word2vec(X_batch,Y_batch)
                else:
                    print ("input right embed way")
                    exit()
                yield X,Y
                X_batch = []
                Y_batch = []

        if X_batch!=[]:
            if encode =='onehot':
                X,Y = self.oneHot(X_batch,Y_batch)
            elif encode == 'wf':
                X,Y = self.wf(X_batch,Y_batch)
            elif encode == 'tf':
                X,Y = self.tf(X_batch,Y_batch)
            elif encode == 'tfidf':
                X,Y = self.tfIdf(X_batch,Y_batch)
            elif encode == 'word2vec':
                X,Y = self.word2vec(X_batch,Y_batch)
                pass
            else:
                print ("input right embed way")
                exit()
            yield X,Y
            X_batch = []
            Y_batch = []
    '''
    vectorize: oneHot  tfIdf
    '''
    def oneHot(self, X, Y):
        x = np.zeros((len(X),len(self.word_dict)))
        y = np.array(Y)
        for index, sentence in enumerate(X):
            sentence = sentence.strip().split()
            for word in sentence:
                try:
                    word_index_in_vocab = self.word_dict[word]
                    x[index, word_index_in_vocab] = 1
                except:
                    continue
        return x,y
    def wf(self, X, Y):
        x = np.zeros((len(X),len(self.word_dict)))
        y = np.array(Y)
        for index, sentence in enumerate(X):
            sentence = sentence.strip().split()
            for word in sentence:
                try:
                    word_index_in_vocab = self.word_dict[word]
                    x[index,word_index_in_vocab] += 1
                except:
                    continue
        return x,y 
    def tf(self, X, Y):
        x = np.zeros((len(X),len(self.word_dict)))
        y = np.array(Y)
        for index, sentence in enumerate(X):
            sentence = sentence.strip().split()
            for word in sentence:
                try:
                    word_index_in_vocab = self.word_dict[word]
                    x[index, word_index_in_vocab] += float(1/len(sentence))
                except:
                    continue
        # print('tf')
        print(x.shape)
        return x,y
    def tfIdf(self, X, Y):
        x = np.zeros((len(X),len(self.word_dict)))
        y = np.array(Y)
        sum_word = np.ones((len(self.word_dict)))
        logX = float(math.log(len(X)))
        idf = []
        for index, sentence in enumerate(X):
            sentence = sentence.strip().split()
            # print(sentence)
            for word in set(sentence):
                try:
                    word_index_in_vocab = self.word_dict[word]
                    sum_word[word_index_in_vocab] += 1
                    x[index,word_index_in_vocab] += float(sentence.count(word)/len(sentence))
                    # print(x)
                except:
                    continue
        for i in sum_word:
            a = logX - math.log(float(i))
            idf.append(a)
        idfvec = np.array(idf)
        for index,sentence in enumerate(X):
            sentence = sentence.strip().split()
            for word in set(sentence):
                try:
                    word_index_in_vocab = self.word_dict[word]
                    x[index,word_index_in_vocab] *= idfvec[word_index_in_vocab]
                except:
                    continue
        print("train_data")
        # print (x)
        print(x.shape)
        return x,y 
    def word2vec(self, X, Y):
        # word2vec_dict = models.Word2Vec(X,size=len(self.word_dict))
        x = np.zeros((len(X),len(self.word_dict)))
        y = np.array(Y)
        for index, sentence in enumerate(X):
            sentence = sentence.strip().split()
            for word in sentence:
                if word in self.word2vec_dict:
                    x[index,] += self.word2vec_dict[word][:len(self.word_dict)]
                else:
                    continue
        # print(x.shape)
        return x,y
        

    '''
    x,y = getABatch(path,length)
    vectorize all data
    '''
    def getABatch(self, path, length,encode):

        X,Y = self.readData(path,length)
        if encode == 'onehot':
            x,y = self.oneHot(X,Y)
        elif encode == 'wf':
            x,y = self.wf(X,Y)
        elif encode == 'tf':
            x,y = self.tf(X,Y)
        elif encode == 'tfidf':
            x,y = self.tfIdf(X,Y)
        elif encode == 'word2vec':
            x,y = self.word2vec(X,Y)
        else:
            print("input right embed way")
            exit()
        return x,y
    # def trainNB(self, train_data_path,test_data_path,
    #      train_length, test_length, batch_size, encode, _model):
    #     X_train_text,Y_train_text = self.readData(train_data_path,train_length)
    #     train_Iter = self.getMinibatch(X_train_text,Y_train_text,batch_size,encode)
    #     X_test, Y_test = self.getABatch(test_data_path,test_length,encode)
    #     print("-"*50)
    #     print("Train model")
    #     if _model == "GaussianNB":
    #         clf = GaussianNB()
    #     elif _model == "MultinomialNB":
    #         clf = MultinomialNB()
    #     elif _model == "BernoulliNB":
    #         clf = BernoulliNB()
    #     else:
    #         print("Please input right model")
    #     for X_train, Y_train in train_Iter:
    #         clf.partial_fit(X_train, Y_train, classes = np.array([0,1]))
    #     return X_test, Y_test
    '''
    cost
    '''
    def acc(self, Y_test, Y_pre):
        length = len(Y_pre)
        right = 0
        for i in range(length):
            if Y_test[i] == Y_pre[i]:
                right += 1
        return right/float(length)


if __name__ == '__main__':


    cf =  config()
    cf._vectorize = argv[1]  # onehot, wf, tf, tfidf, word2vec
    cf._model = argv[2] 
    #choose GaussianNB,MultinomialNB,BernoulliNB or SVM_SVC_linear SVM_SVC_rbf
    cf.dict_length = int(argv[3])
    cf.test_data_num = int(argv[4])
    cf.train_data_num = int(argv[5])
    # cf._vectorize = 'word2vec'  # onehot, wf, tf, tfidf, word2vec
    # cf._model = 'BernoulliNB' 
    # #choose GaussianNB MultinomialNB,BernoulliNB or SVM_SVC_linear SVM_SVC_rbf
    # cf.dict_length= 1000
    # cf.test_data_num = 1000
    # cf.train_data_num = 9000
    # cf.word2vec_maxnum = 4000
    start_time = time.time()
    '''
    get the data
    '''
    print ("load vocab")
    word_dict = createVocab(cf.vocab_path,cf.dict_length)
    
    print ("-"*50)

    print ("load & Deal train and test data")
    vec = Vectorize(word_dict)
    '''
    train and predict
    '''
    # print ("Train model")
    if cf._model == "GaussianNB":
        # X_test, Y_test = vec.trainNB(cf.train_data_path,cf.test_data_path,cf.train_data_num,cf.test_data_num,
        #     cf.batch_size, cf._vectorize,cf._model)
        X_train_text,Y_train_text = vec.readData(cf.train_data_path,cf.train_data_num)
        vec.word2vec_dict = models.Word2Vec(X_train_text, size=cf.dict_length)
        # X_train,Y_train = vec.getABatch(cf.test_data_path,cf.train_data_num,clf=cf._vectorize)
        train_Iter =  vec.getMinibatch(X_train_text,Y_train_text,cf.batch_size,cf._vectorize)
        X_test,Y_test = vec.getABatch(cf.test_data_path,cf.test_data_num,cf._vectorize)
        print ("-"*50)
        
        print ("Train model")
        clf = GaussianNB()
        # clf.fit(X_train,Y_train,classes=np.array([0,1]))
        for X_train,Y_train in train_Iter:
            clf.partial_fit(X_train,Y_train,classes=np.array([0,1]))
    
    elif cf._model =="MultinomialNB":
        X_train_text,Y_train_text = vec.readData(cf.train_data_path,cf.train_data_num)
        train_Iter =  vec.getMinibatch(X_train_text,Y_train_text,cf.batch_size,cf._vectorize)
        X_test,Y_test = vec.getABatch(cf.test_data_path,cf.test_data_num,cf._vectorize)
        print ("-"*50)

        print ("Train model")
        clf = MultinomialNB()
        for X_train,Y_train in train_Iter:
            clf.partial_fit(X_train,Y_train,classes=np.array([0,1]))
    
    elif cf._model == "BernoulliNB":
        X_train_text,Y_train_text = vec.readData(cf.train_data_path,cf.train_data_num)
        train_Iter =  vec.getMinibatch(X_train_text,Y_train_text,cf.batch_size,cf._vectorize)
        X_test,Y_test = vec.getABatch(cf.test_data_path,cf.test_data_num,cf._vectorize)
        print ("-"*50)

        print ("Train model")
        clf = BernoulliNB()
        for X_train,Y_train in train_Iter:
            clf.partial_fit(X_train,Y_train,classes=np.array([0,1]))
    
    elif cf._model == "SVM_SVC_linear":
        X_train, Y_train = vec.getABatch(cf.train_data_path, cf.train_data_num,cf._vectorize)
        X_test, Y_test = vec.getABatch(cf.test_data_path,cf.test_data_num, cf._vectorize)
        print ("-"*50)
        
        print ("Train model")
        clf = SVC(kernel='linear', cache_size=500, C=1.0, gamma='auto', coef0=0)
        clf.fit(X_train, Y_train)
    
    elif cf._model == "SVM_SVC_rbf":
        X_train, Y_train = vec.getABatch(cf.train_data_path, cf.train_data_num,cf._vectorize)
        X_test, Y_test = vec.getABatch(cf.test_data_path,cf.test_data_num, cf._vectorize)
        print ("-"*50)
        
        print ("Train model")
        clf = SVC(kernel='rbf', gamma=4, coef0=0)
        clf.fit(X_train, Y_train)
    
    elif cf._model == "GBDT":
        X_train, Y_train = vec.getABatch(cf.train_data_path, cf.train_data_num,cf._vectorize)
        X_test, Y_test = vec.getABatch(cf.test_data_path,cf.test_data_num, cf._vectorize)
        print ("-"*50)

        print ("Train model") 
        clf = GradientBoostingClassifier()
        clf.fit(X_train,Y_train)
    else:
        print ("Please input right model")
        exit()

    print ("-"*50)

    print ("Test")
    print ("-"*50)

    Y_pre = clf.predict(X_test)

    print ("model:%s\nembed:%s\ndict_length:%d\ntrain_data_num:%d\ntest_data_num:%s"
        %(cf._model,cf._vectorize,cf.dict_length,cf.train_data_num,cf.test_data_num))

    print("Results:")
    print("\tY_pre:", Y_pre.shape)
    print("\tY_test:", Y_test.shape)
    print("\tcost:{0:.3f}".format(vec.acc(Y_pre,Y_test)))
    print("time:{0:.3f}s".format((time.time()-start_time)))


 





