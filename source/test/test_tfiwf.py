import numpy as np
import sys
from sys import argv
import math
from sklearn.naive_bayes import GaussianNB

def createVocab(path, length):
    vocab = []
#    with open(path, 'r', encoding='utf-8') as f:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines()[:length]:
            vocab.append(line.strip()[0])
    word_dict = dict(zip(vocab, range(len(vocab))))  # zip 对应位置打包，('8',0) index_init = 0
    return word_dict


def read_data(path, length):
    X = []
    Y = []
    with open(path, 'r', encoding='utf-8') as f:
        if length == None:
            length = len(f.readlines())
        for line in f.readlines()[:length]:
            try:
                sentence, label = line.strip().split('\t')
            except:
                continue
            X.append(sentence)
            Y.append(int(label))
    return X, Y

def tfIwf(self, X, Y):
    x = np.zeros((len(X), len(self.word_dict)))
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
                x[index, word_index_in_vocab] += float(
                    sentence.count(word) / len(sentence))
                    # print(x)
            except:
                continue
    for i in sum_word:
        a = logX - math.log(float(i))
        idf.append(a)
    idfvec = np.array(idf)
    for index, sentence in enumerate(X):
        sentence = sentence.strip().split()
        for word in set(sentence):
            try:
                word_index_in_vocab = self.word_dict[word]
                x[index, word_index_in_vocab] *= idfvec[
                    word_index_in_vocab]
            except:
                continue
    print("train_data")
        # print (x)
    print(x.shape)
    return x, y



word_dict = createVocab('./FEATURE_ORDERED.txt',200)
X, Y = read_data('./train', 200)
