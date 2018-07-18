# 基于char level  使用信息增益方法 
# train 训练集


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

def createTree(dataSet, labels):
    """
    输入：数据集，特征标签
    输出：决策树
    描述：递归构建决策树，利用上述的函数
    """
    fea_list = []

    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataSet[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    fea_len = len(labels)
    #for i in range(fea_len):
        #sys.stderr.write("%c%d/%d"%(13,i,fea_len))
    fea_info_dict = chooseBestFeatureToSplit(dataSet)
    #print(len(fea_info_dict))
    fea_info_list = sorted(fea_info_dict.items(),key = lambda x:x[1],reverse = True)
    #print(len(fea_info_list))
    #print fea_info_list
    for (fea_index,info) in fea_info_list:
        feature = labels[fea_index]
        print feature,"\t",info


    #myTree = {bestFeatLabel:{}}
    #del(labels[bestFeat])
    #dataSet = splitDataSet(dataSet, bestFeat, "")
    #sys.stderr.write("\n")
    # 得到列表包括节点所有的属性值
    #featValues = [example[bestFeat] for example in dataSet]
    #uniqueVals = set(featValues)
    #for value in uniqueVals:
    #    subLabels = labels[:]
    #    myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    #return myTree

    return fea_list

def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def classifyAll(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll

def storeTree(inputTree, filename):
    """
    输入：决策树，保存文件路径
    输出：
    描述：保存决策树到文件
    """
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    """
    输入：文件路径名
    输出：决策树
    描述：从文件读取决策树
    """
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

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
        dataSet, labels = createDataSet_1gram(con.data_path,con.feature_num)
    else:
        dataSet, labels = createDataSet_2gram(con.data_path,con.feature_num)


    labels_tmp = labels[:] # 拷贝，createTree会改变labels
    print(len(labels_tmp))
    print(len(dataSet))
    #exit()
    #exit()
    desicionTree = createTree(dataSet, labels_tmp)
    #storeTree(desicionTree, 'classifierStorage.txt')
    #desicionTree = grabTree('classifierStorage.txt')
    #print('desicionTree:\n',"\n".join(desicionTree))
   #treePlotter.createPlot(desicionTree)
    #testSet = createTestSet()
    #print('classifyResult:\n', classifyAll(desicionTree, labels, testSet))

if __name__ == '__main__':
    main()