#coding:utf-8

import math
import operator

def createData():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    label=['no surfacing', 'filter']
    return dataSet, label

# 计算香浓熵
def calcShannonEnt(dataSet):
    size_data = len(dataSet)
    labelCounts = {}
    for data in dataSet:
        curLabel = data[-1]
        if curLabel not in labelCounts.keys():
            labelCounts[curLabel] = 1
        else:
            labelCounts[curLabel] += 1
    infoEntropy = 0.0
    for key in labelCounts:
        p = 1.0* labelCounts[key] / size_data
        infoEntropy -= p * math.log(p, 2)

    return infoEntropy


# 对带划分的数据集dataSet在给定的数据集特征axis上按给定的特征值value进行划分
def splitDataSet(dataSet, aixs, value):
    resData = []
    for dat in dataSet:
        if dat[aixs] == value:
            tmpData = dat[:aixs]
            tmpData.extend(dat[aixs+1:])
            resData.append(tmpData)
    return resData


# 选出信息增益最大的划分方式
def chooseBestFeatureToSplit(dataSet):

    size_data = float(len(dataSet))
    num_feature = len(dataSet[0])-1
    # 划分之前的信息熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1

    for i in range(num_feature):
        feature_list = [label[i] for label in dataSet]
        # 转换位set类型除去重复数据
        uniqVal = set(feature_list)
        # 按属性i划分之后得到的新的信息熵
        newEntropy = 0.0
        for value in uniqVal:
            subDataSet = splitDataSet(dataSet, i, value)
            p = len(subDataSet) / size_data
            newEntropy += p * calcShannonEnt(subDataSet)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def getMostClassID(classList):
    classCount = {}
    for label in classList:
        if label not in classCount.keys(): classCount[label] = 0
        classCount[label] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 递归创建决策树
def createTree(dataSet, labels):
    classList = [item[-1] for item in dataSet]

    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet) == 1:
        return getMostClassID(classList)

    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del (labels[bestFeature])

    featureValues = [item[bestFeature] for item in dataSet]
    uniqValues = set(featureValues)
    for value in uniqValues:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)

    return myTree
