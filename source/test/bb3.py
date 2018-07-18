import numpy as np
import math
from sklearn.naive_bayes import GaussianNB  
from sklearn.naive_bayes import MultinomialNB


'''
construct vocab
'''


class config(object):
    """docstring for config"""
    def __init__(self):
        super(config, self).__init__()
        self._vectorize = "zeroone"# choose zeroone of tfidf
        self._model = "NB" #choose NB or SVM
        
        self.dict_length = 1000
        self.batch_size = 1000
        self.train_data_num = 10000 #could be None, means read all data
        self.test_data_num = 10000 

        self.vocab_path = './FEATURE_ORDERED.txt'
        self.train_data_path = './train'
        self.test_data_path = "./test"



def createVocab(path, length):
    vocab = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines()[:length]:
            vocab.append(line.strip()[0])
    word_dict = dict(zip(vocab, range(len(vocab))))  # zip 对应位置打包，
    # print(word_dict)
    return word_dict

'''
read  data 
create vector
'''

class Vectorize(object):
    def __init__(self,word_dict):
        super(Vectorize, self).__init__()
        self.word_dict = word_dict

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
    
    def getMinibatch(self, inputData_X,inputData_Y,batch_size,clf=None):
        X_batch = []
        Y_batch = []

        for x,y in zip(inputData_X,inputData_Y):
            X_batch.append(x)
            Y_batch.append(y)

            if len(X_batch) == batch_size:
                if clf =='zeroone':
                    X,Y = self.oneHot(X_batch,Y_batch)
                elif clf == 'tfidf':
                    X,Y = self.tf(X_batch,Y_batch)
                else:
                    print ('input right embed way')
                exit()
                yield X,Y
                X_batch = []
                Y_batch = []

        if X_batch!=[]:
            if clf =='zeroone':
                X,Y = self.oneHot(X_batch,Y_batch)
            elif clf == 'tfidf':
                X,Y = self.tf(X_batch,Y_batch)
            else:
                print ('input right embed way')
                exit()
            yield X,Y
            X_batch = []
            Y_batch = []

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
    def tf(self, X, Y):
        x = np.zeros((len(X),len(self.word_dict)))
        y = np.array(Y)
        for index, sentence in enumerate(X):
            sentence = sentence.strip().split()
            for word in sentence:
                try:
                    word_index_in_vocab = self.word_dict[word]
                    x[index, word_index_in_vocab] += 1
                except:
                    continue
        return x,y
 
    # def clf(self, path, length, batch_size, clf=None):

    #     X,Y = self.readData(path,length)
    #     if batch_size == None:
    #         x,y = self.oneHot(X,Y)
    #     else:
    #         data_batch = self.getMinibatch(X,Y,batch_size)
    #         if clf =='zeroone':
    #             x,y = self.oneHot(X_batch,Y_batch)
    #         elif clf == 'tfidf':
    #             x,y = self.tf(X_batch,Y_batch)
    #         else:
    #             x,y = None, None
    #     return x,y


    def cost(self, Y_test, Y_pre):
        length = len(Y_pre)
        right = 0
        for i in range(length):
            if Y_test[i] == Y_pre[i]:
                right += 1
        return right/length


if __name__ == '__main__':

    cf =  config()

    '''
    get the data
    '''
    print ("load vocab")
    word_dict = createVocab(cf.vocab_path,cf.dict_length)
    print ("-"*50)

    print ("load & Deal train and test data")
    
    vec = Vectorize(word_dict)

    X_train_text,Y_train_text = vec.readData(cf.train_data_path,cf.train_data_num)
    X_train, Y_train =  vec.getMinibatch(X_train_text,Y_train_text,cf.batch_size,cf._vectorize)
    X_test_text,Y_test_text = vec.readData(cf.test_data_path,cf.test_data_num)
    X_test, Y_test =  vec.getMinibatch(X_test_text,Y_test_text,cf.batch_size,cf._vectorize)

    print ("-"*50)
    
    '''
    train and predict
    '''
    # GaussianNB  zeroone

    print ("Train model")
    if cf._model == "NB":
        clf = GaussianNB()
    elif cf._model == "SVM":
        #clf = 
        pass
    else:
        print ("Please input right model")
        exit()

    clf.partial_fit(X_train,Y_train,classes=np.array([0,1]))
    print ("-"*50)

    Y_pre = clf.predict(X_test)

    print ("model:%s\nembed:%s\ndict_length:%d\ntrain_data_num:%d\ntest_data_num:%s"%(cf._model,cf._vectorize,cf.dict_length,cf.train_data_num,cf.test_data_num))

    print("Results:")
    print("\tY_pre:", Y_pre)
    print("\tY_test:", Y_test)
    print("\tcost:{0:.3f}".format(vec.cost(Y_pre,Y_test)))


 





