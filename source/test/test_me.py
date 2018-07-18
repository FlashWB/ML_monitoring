import numpy as np
import math
from sklearn.naive_bayes import GaussianNB  
from sklearn.naive_bayes import MultinomialNB


'''
construct vocab
'''
dict_length = 100

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
    
    def getMinibatch(self, inputData, batch_size):
        data_batch = []
        it = iter(inputData)
        try:
            for _ in range(batch_size):
                out = next(it)
                data_batch.append(out)
        except StopIteration:
            return None
        # print(len(data_batch))
        return data_batch

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
 
    def clf(self, path, length, batch_size, clf=None):

        X,Y = self.readData(path,length)
        if batch_size == None:
            x,y = self.oneHot(X,Y)
        else:
            X_batch = self.getMinibatch(X,batch_size)
            Y_batch = self.getMinibatch(Y,batch_size)
            if clf =='zeroone':
                x,y = self.oneHot(X_batch,Y_batch)
            elif clf == 'tfidf':
                x,y = self.tf(X_batch,Y_batch)
            else:
                x,y = None, None
        return x,y


    def cost(self, Y_test, Y_pre):
        length = len(Y_pre)
        right = 0
        for i in range(length):
            if Y_test[i] == Y_pre[i]:
                right += 1
        return right/length


if __name__ == '__main__':
    '''
    get the data
    '''
    word_dict = createVocab('./FEATURE_ORDERED.txt',1000)

    vec = Vectorize(word_dict)
    
    # zeroone
    X_train01, Y_train01 = vec.clf('./train',10000,1000,'zeroone')

    # tfidf
    X_traintf, Y_traintf = vec.clf('./train',10000,1000,'tfidf')
    print(X_traintf[3])
    X_test, Y_test = vec.clf('./train',100,None)
    '''
    train and predict
    '''
    # GaussianNB  zeroone
    clf = GaussianNB()
    clf.partial_fit(X_train01,Y_train01,classes=np.array([0,1]))
    Y_pre01 = clf.predict(X_test)

    # print("Y_pre_01:", Y_pre01)
    # print("Y_test:", Y_test)
    # print("cost_01:{0:.3f}".format(vec.cost(Y_pre01,Y_test)))
    
    # GaussianNB tfidf
    clf = GaussianNB()
    clf.partial_fit(X_traintf,Y_traintf,classes=np.array([0,1]))
    Y_pretf = clf.predict(X_test)

    # print("Y_pre_tfidf:", Y_pretf)
    # print("Y_test:", Y_test)
    # print("cost_tfidf:{0:.3f}".format(vec.cost(Y_pretf,Y_test)))
 





