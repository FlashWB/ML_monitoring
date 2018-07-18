import numpy as np
from gensim.models import Word2Vec

def read_data(path,length):
    X = []
    Y = []
    with open(path,'r',encoding='utf-8') as f:
        if length == None:
            length = len(f.readlines())
        for line in f.readlines()[:length]:
            try:
                sentence, label = line.strip().split('\t')
            except:
                continue
            X.append(sentence)
            Y.append(label)
    return X, Y

X_train,Y_train = read_data('./train',90000)
model = Word2Vec(X_train, min_count=10, size=4000) # size 
model.save('./word2vec_model2')
# print(X_train)
print(model['2'][0])










