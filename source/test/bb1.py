import numpy as np
from sklearn.naive_bayes import GaussianNB

'''
construct vocab
'''
def createVocab(path,length):
    vocab = []
    with open(path,'r',encoding = 'utf-8') as f:
    #with open(path,'r') as f:
        for line in f.readlines()[:length+1]:
            vocab.append(line.strip()[0])
    word_dict = dict(zip(vocab,range(len(vocab))))
    return word_dict


'''
read  data 
create vector
'''
class Vectorize(object):
    """docstring for Vectorize"""
    def __init__(self,word_dict):
        super(Vectorize, self).__init__()
        self.word_dict = word_dict

        

        
    def readData(self,path,length):
        X = []
        Y = []
        with open(path,'r',encoding='utf-8') as t:
        #with open(path,'r') as t:
            if length == None:
                length = len(t.readlines())
            
            for line in t.readlines()[:length]:
                try:
                    sentence,label = line.strip().split('\t')
                except:
                    continue
                X.append(sentence)
                Y.append(int(label))

        return X,Y

    def ZeroOne(self,path,length):
        
        X,Y = self.readData(path,length)

        x = np.zeros((len(X),len(self.word_dict)))
        y = np.array(Y)

        for index,sentence in enumerate(X):
            sentence = sentence.strip().split()
            for word in sentence:
                try:
                    word_index_in_vocab = self.word_dict[word]
                    x[index,word_index_in_vocab] = 1
                except:
                    continue
        return x,y

    def TFIDF(self,path,length):
        pass


if __name__ == '__main__':
    
    '''
    get the data
    '''
    word_dict = createVocab('./FEATURE_ORDERED.txt',100)    

    Vec = Vectorize(word_dict)  

    X_train,Y_train = Vec.ZeroOne('./train',100) 

    X_test,Y_test = Vec.ZeroOne('./test',10)    

    print (X_train.shape)
    print (Y_train.shape)

    '''
    train and prediction
    '''
    clf = GaussianNB()
    clf.fit(X_train,Y_train)
    # score = clf.score(X_train,Y_train)
    Y_pre = clf.predict(X_test)
    # print("Score:", score)
    print("Y_pre:", Y_pre)
    print("Y_test:", Y_test)







