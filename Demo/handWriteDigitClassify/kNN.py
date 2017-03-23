#coding:utf-8

import operator
import numpy as np


# 归一化函数
def autoNorm():

    minValues = group.min(0)
    maxValues = group.max(0)
    ranges = maxValues - minValues

    normData = zeros(group.shape)

    normData = group - np.tile(minValues, (group.shape[0], 1))
    normData = normData*1.0 / np.tile(ranges, (group.shape[0], 1))

    return normData, ranges, minValues

# 标签数据向量化
def label2Vector(filename):
    f = open(filename, 'rb')
    line = f.read().strip()
    labelVect = np.zeros((1,2000))
    for i in range(2000):
        labelVect[0][i] = int(line[i])

    f.close()
    return labelVect[0]

# 图像数据标签化
def img2Vector(filename):
    f = open(filename, "rb")
    lines = f.readlines()
    num_lines = len(lines)

    dataVect = np.zeros((num_lines/28,28*28))

    count = 0
    for i in np.arange(1, num_lines, 28):
        for x in range(28):
            row_data = lines[x+count*28].strip()
            for y in range(28):
                dataVect[i/28,x*28+y] = int(row_data[y])
        count += 1

    f.close()

    return dataVect



def classify(Input, group, labels, k):

    # 计算欧式距离
    dataSize = group.shape[0]
    diff = group - np.tile(Input, (dataSize, 1))
    diff = diff**2
    sqSum = diff.sum(axis=1)
    sqSum = sqSum**0.5
    # 对欧式距离从小到大进行索引排序
    sqRes = sqSum.argsort()

    # 统计k个类别中各自的频率
    classCount = {}
    for i in range(k):
        label = labels[sqRes[i]]
        classCount[label] = classCount.get(label, 0)+1

    # 最多的即为Input的类别
    sortedClassCount = sorted(classCount.items(), lambda x,y:cmp(x[1],y[1]), reverse=True)
    return sortedClassCount[0][0]

