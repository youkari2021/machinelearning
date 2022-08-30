import random

from numpy import shape, ones


def createVocabList(dataSet):   # 创建一个包含在所有文档中出现的不重复的词的列表(顺序随机)。
    vocabSet = set([])  # create empty set
    for document in dataSet:    # 当入参为二重列表（向量集合）时用这个
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def bagOfWords2VecMN(vocabList, inputSet):  # 获得文档向量，向量中的数值代表词汇表中的某个单词在一篇文档中的出现次数
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def sigmoid(x):
    if x > 0:
        return 1
    else:
        return 0


def GradAscent(dataMat, labelMat, numIter=150):   # 第三个参数为迭代次数
    m, n = shape(dataMat)
    weight = ones(n)    # 初始化权值w
    for i in range(numIter):
        print('第' + str(i) + '次迭代开始')
        dataInd = list(range(m))  # 剩余数据下标
        for j in range(m):
            study_rate = 4/(1.0 + i + j) + 0.01 # 加一个常数使学习率不会趋于0（新数据永远对权值有影响）,当迭代次数较小时，j占主导，有可能随机跳出局部最优
            randInd = int(random.uniform(0, len(dataInd)))
            del(dataInd[randInd])
            pred = sigmoid(sum(dataMat[randInd] * weight))
            error = labelMat[randInd] - pred    # error有正有负，影响下一行的移动方向
            weight = weight + study_rate * float(error) * dataMat[randInd]
        print('第'+str(i)+'次迭代结束')
    return weight
