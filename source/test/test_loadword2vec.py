import numpy as np
import sys
from sys import argv

from sklearn.naive_bayes import GaussianNB
from gensim import models

dict = models.Word2Vec.load('./word2vec_model2')
print (dict['2'][:100])
# word = ['北', 1, 2,'京','3']

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
            Y.append(int(label))
    return X,Y

X,Y = read_data('./train',5)
# print(X,Y)
# # print(word)
inputx = np.zeros((len(X),3000))
for index, sentence in enumerate(X):
    sentence = sentence.strip().split()
    for word in sentence:
        print(word)
        if word in dict:
            inputx[index,] += dict[word][:3000]
        else:
            continue
    print(inputx[index])
x = dict["2"]
# # y = dict["1"]
# print(x[1])
# print(y[1])
# print((x+y)[1])
# print (len(x + y))




