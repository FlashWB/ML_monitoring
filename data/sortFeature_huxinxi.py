# -*- coding: utf-8 -*-

from math import log
import operator
import sys
import numpy as np
from collections import Counter
from nltk.util import ngrams
#import treePlotter

class config(object):
    """docstring for config"""
    def __init__(self):
        super(config, self).__init__()
        self.feature_num = 20
        self.data_path = "../../data/train" #char or word
        self.ngram = 2 #1:unigram;2:bigram
        self.method = "CrossEntroy"#KLIC:信息增益；KaFang:卡方检验；PMI：互信息；CrossEntroy:交叉熵.


def calcShannonEnt(dataSet):
    """
    输入：数据集
    输出：数据集的香农熵
    描述：计算给定数据集的香农熵；熵越大，数据集的混乱程度越大
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt



def splitDataSet(dataSet, axis, value):
    """
    输入：数据集，选择维度，选择值
    输出：划分数据集
    描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]+featVec[axis+1:]
            retDataSet.append(reduceFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    输入：数据集
    输出：最好的划分维度
    描述：选择最好的数据集划分维度
    """
    numFeatures = len(dataSet[0])  - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRatio = 0.0
    bestFeature = -1
    fea_info_dict = {}
    for i in range(numFeatures):
        sys.stderr.write("%c%d"%(13,i))
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        splitInfo = 0.0
        #print "\nuniqueVals:",uniqueVals
        for value in uniqueVals:
            #print "split dataset"
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            #print "calc ShannonEnt"
            newEntropy += prob * calcShannonEnt(subDataSet)
            splitInfo += -prob * log(prob, 2)
        #print "end"
        infoGain = baseEntropy - newEntropy
        if (splitInfo == 0): # fix the overflow bug
            fea_info_dict[i] = 0
            continue
        infoGainRatio = infoGain / splitInfo
        fea_info_dict[i] = infoGainRatio
        """
        if (infoGainRatio > bestInfoGainRatio):
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
        """
    sys.stderr.write("\n")
    return fea_info_dict

def majorityCnt(classList):
    """
    输入：分类类别列表
    输出：子节点的分类
    描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
          采用多数判决的方法决定该子节点的分类
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reversed=True)
    return sortedClassCount[0][0]

def KLIC(dataSet, vocab):
    """
    输入：数据集，词表
    """
    fea_list = []

    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataSet[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    fea_len = len(vocab)
    #for i in range(fea_len):
        #sys.stderr.write("%c%d/%d"%(13,i,fea_len))
    fea_info_dict = chooseBestFeatureToSplit(dataSet)
    #print(len(fea_info_dict))
    fea_info_list = sorted(fea_info_dict.items(),key = lambda x:x[1],reverse = True)
    #print(len(fea_info_list))
    #print fea_info_list
    for (fea_index,info) in fea_info_list:
        feature = vocab[fea_index]
        print feature,"\t",info

    return fea_info_list


def KaFang(dataSet,vocab):
    '''
    输入:数据集,词表
    输出：排序好的特征
    '''
    numFeatures = len(dataSet[0])  - 1
    N = len(dataSet)
    fea_info_dict = {}
    for i in range(numFeatures):
        A = 0
        B = 0
        C = 0
        D = 0
        for j in range(N): 
            label = dataSet[j][-1]
            data = dataSet[j]
            if label == 'Y' and data[i] == 1:
                A +=1
            elif label == 'Y' and data[i] == 0:
                C+=1
            elif label == 'N' and data[i] == 1:
                B+=1
            else:
                D+=1
        x = N*(A*B-C*D)*(A*B-C*D)/float((A+C)*(A+B)*(B+D)*(C+D))
        fea_info_dict[i] = x

    fea_info_list = sorted(fea_info_dict.items(),key = lambda x:x[1],reverse = True)
    
    for (fea_index,info) in fea_info_list:
        feature = vocab[fea_index]
        print feature,"\t",info

    return fea_info_list  


def CrossEntroy(dataSet,vocab):


    numFeatures = len(dataSet[0])  - 1
    N = len(dataSet)
    fea_info_dict = {}
    for i in range(numFeatures):
        A = 1
        B = 1
        C = 1
        D = 1
        for j in range(N): 
            label = dataSet[j][-1]
            data = dataSet[j]
            if label == 'Y' and data[i] == 1:
                A +=1
            elif label == 'Y' and data[i] == 0:
                C+=1
            elif label == 'N' and data[i] == 1:
                B+=1
            else:
                D+=1
        Pt = (A+B)/float(N+4)
        PY = (A+C)/float(N+4)
        PN = (B+D)/float(N+4)

        PYt = A/float(A+B)
        PNt = B/float(A+B)

        CD_Y = A/float(A+B)
        CD_N = B/float(A+B)

        DD_Y = A/float(A+C)
        DD_N = B/float(B+D)

        CD_ECE = (PYt*log(PYt/PY)*CD_Y*DD_Y + PNt*log(PNt/PN)*CD_N*DD_N)*Pt
        fea_info_dict[i] = CD_ECE
    fea_info_list = sorted(fea_info_dict.items(),key = lambda x:x[1],reverse = True)
    for (fea_index,info) in fea_info_list:
        feature = vocab[fea_index]
        print feature,"\t",info

    return fea_info_list  


def PMI(dataSet,vocab):

    numFeatures = len(dataSet[0])  - 1
    N = len(dataSet)
    fea_info_dict = {}
    for i in range(numFeatures):
        N11 = 1
        N10 = 1
        N01 = 1
        N00 = 1
        for j in range(N): 
            label = dataSet[j][-1]
            data = dataSet[j]
            if label == 'Y' and data[i] == 1:
                N11 +=1
            elif label == 'Y' and data[i] == 0:
                N01+=1
            elif label == 'N' and data[i] == 1:
                N10+=1
            else:
                N00+=1

        I = N11/float(N+4)*log((N+4)*N11/float((N11+N10)*(N01+N11))) \
            + N01/float(N+4)*log((N+4)*N01/float((N01+N00)*(N01+N11)))\
            + N10/float(N+4)*log((N+4)*N10/float((N11+N10)*(N00+N10)))\
            + N00/float(N+4)*log((N+4)*N00/float((N01+N00)*(N10+N00)))
        fea_info_dict[i] = I
    fea_info_list = sorted(fea_info_dict.items(),key = lambda x:x[1],reverse = True)
    for (fea_index,info) in fea_info_list:
        feature = vocab[fea_index]
        print feature,"\t",info

    return fea_info_list      


def get_vocab_1gram(file_path,fea_num):
    data = (file(file_path).read()).decode("utf-8")
    vocab = []
    data = data.strip().split("\n")
    for sentence in data:
        sentence = sentence.split("\t")[0]
        vocab+=sentence.strip().split()

    vocab_count = Counter(vocab)
    vocab_ordered = (sorted(vocab_count.items(),key = lambda x:x[1],reverse = True))
    #print vocab_ordered[1000]
    vocab = []
    for i in range(fea_num):
        vocab.append(vocab_ordered[i][0])
    print len(vocab)
    #print " ".join(vocab)
    return vocab,dict(zip(vocab,range(len(vocab))))

def get_vocab_2gram(file_path,fea_num):
    data = (file(file_path).read()).decode("utf-8").strip().split("\n")

    vocab = []
    for sentence in data:
        sentence = sentence.strip().split("\t")[0].split()
        sentence = list(ngrams(sentence,2))
        for bigram in sentence:
            vocab.append(" ".join(list(bigram)))

    vocab_count = Counter(vocab)
    vocab_ordered = (sorted(vocab_count.items(),key = lambda x:x[1],reverse = True))
    #print vocab_ordered[1000]
    vocab = []
    for i in range(fea_num):
        vocab.append(vocab_ordered[i][0])
    print len(vocab)
    #print " ".join(vocab)
    return vocab,dict(zip(vocab,range(len(vocab))))

def TextToFeature1gram(text,rev_vocab):
    fea = np.zeros((len(rev_vocab)))
    for char in text:
        try:
            fea[rev_vocab[char]]=1
        except:
            pass
    return list(fea)

def TextToFeature2gram(text,rev_vocab):
    fea = np.zeros((len(rev_vocab)))
    for char in text:
        try:
            fea[rev_vocab[" ".join(char)]]=1
        except:
            pass
    return list(fea)

def createDataSet_1gram(file_path,fea_num):
    vocab,rev_vocab = get_vocab_1gram(file_path,fea_num)
    
    dataSet = []
    #i = 0
    for line in file(file_path):
    #    i+=1
    #    if i>10:
    #        break
        try:
            
            line,label = line.strip().decode("utf-8").split("\t")
            line = line.strip().split()
            fea = TextToFeature1gram(line,rev_vocab)
        except:
            continue
        if label.strip() == "1":
            dataSet.append(fea+["Y"])
        else:
            dataSet.append(fea+["N"])

    #print dataSet
    
    return dataSet,vocab

def createDataSet_2gram(file_path,fea_num):
    vocab,rev_vocab = get_vocab_2gram(file_path,fea_num)
    
    dataSet = []
    #i = 0
    for line in file(file_path):
    #    i+=1
    #    if i>10:
    #        break
        try:
            line,label = line.strip().decode("utf-8").split("\t")
            line = line.strip().split()
            line = ngrams(line,2)
            fea = TextToFeature2gram(line,rev_vocab)
        except:
            continue
        if label.strip() == "1":
            dataSet.append(fea+["Y"])
        else:
            dataSet.append(fea+["N"])

    #print dataSet
    
    return dataSet,vocab

def createDataSet_ori():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true 
    """
    dataSet = [[0, 0, 0, 0, 'N'], 
               [0, 0, 0, 1, 'N'], 
               [1, 0, 0, 0, 'Y'], 
               [2, 1, 0, 0, 'Y'], 
               [2, 2, 1, 0, 'Y'], 
               [2, 2, 1, 1, 'N'], 
               [1, 2, 1, 1, 'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return dataSet, labels

def createTestSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true 
    """
    testSet = [[0, 1, 0, 0], 
               [0, 2, 1, 0], 
               [2, 1, 1, 0], 
               [0, 1, 1, 1], 
               [1, 1, 0, 1], 
               [1, 0, 1, 0], 
               [2, 1, 0, 1]]
    return testSet

def main():
    con = config()


    if con.ngram == 1:
        dataSet, vocab = createDataSet_1gram(con.data_path,con.feature_num)
    else:
        dataSet, vocab = createDataSet_2gram(con.data_path,con.feature_num)


    vocab_tmp = vocab[:] # 拷贝，createTree会改变vocab
    print(len(vocab_tmp))
    print(len(dataSet))
    #exit()
    #exit()
    if con.method == "KaFang":
        fea_list = KaFang(dataSet, vocab_tmp)
    elif con.method == "KLIC":
        fea_list = KLIC(dataSet, vocab_tmp)
    elif con.method == "CrossEntroy":
        fea_list = CrossEntroy(dataSet, vocab_tmp)
    elif con.method == "PMI":
        fea_list = PMI(dataSet, vocab_tmp)



if __name__ == '__main__':
    main()