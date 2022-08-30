"""
Created on Mar 1, 2018
"""
import numpy as np
import re
import pickle


def textParse(bigString):  # input is big string, #output is word list
    """
		接受一个大字符串并将其解析为字符串列表。该函数去掉少于两个字符的字符串，并将所有字符串转换为小写。
	"""

    listOfTokens = re.split(r'\W', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def createVocabList(dataSet):
    """
		创建一个包含在所有文档中出现的不重复的词的列表。
	"""

    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def bagOfWords2VecMN(vocabList, inputSet):
    """
		获得文档向量，向量中的数值代表词汇表中的某个单词在一篇文档中的出现次数
	"""

    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def spamtest():
    dataset = []  # 文档向量的集合
    classlist = []  # 类标签列表
    errorindex = []
    for i in range(1, 26):  # i取值1-25，是因为每个文件夹的文件名为1-25
        # 交叉放入向量集合
        listoftokens = textParse(open('./ham/{}.txt'.format(str(i)), 'r').read())  # 23.txt中®符号无法被gbk编码读取，直接用空格代替
        dataset.append(listoftokens)
        classlist.append(1)
        listoftokens = textParse(open('./spam/{}.txt'.format(str(i)), 'r').read())
        dataset.append(listoftokens)
        classlist.append(0)
    vocablist = createVocabList(dataset)
    # print(vocablist)
    file = open('vocablist.txt', 'wb')
    pickle.dump(vocablist, file)
    file.close()
    train_indexset = list(range(50))
    test_indexset = []
    for i in range(10):  # 从训练集里随机删掉10个放到测试集
        ranindex = np.random.randint(0, len(train_indexset))
        test_indexset.append(train_indexset[ranindex])
        del train_indexset[ranindex]
    train_mat = []
    train_class = []
    for index in train_indexset:
        train_mat.append(bagOfWords2VecMN(vocablist, dataset[index]))
        train_class.append(classlist[index])
    px_c0, px_c1, pc1 = train(train_mat, train_class)
    file = open('threerate.txt', 'wb')
    pickle.dump([px_c0, px_c1, pc1], file)
    file.close()

    errorcount = 0
    for testindex in test_indexset:
        test_mat = bagOfWords2VecMN(vocablist, dataset[testindex])
        if classify(test_mat, px_c0, px_c1, pc1) != classlist[testindex]:
            errorcount += 1
            errorindex.append(testindex)

    return errorcount/len(test_indexset), set(errorindex)


def train(train_mat, train_class):
    textnum = len(train_mat)
    wordnum = len(train_mat[0])
    # print(train_class)
    pc1 = sum(train_class) / textnum
    # 初始化时，分子分母都已经有了1和2，这样做是拉普拉斯修正的方法，防止缺失值的影响
    p0num = np.ones(wordnum)
    p1num = np.ones(wordnum)
    p0demo = 2
    p1demo = 2
    for i in range(textnum):
        if train_class[i] == 1:
            p1num += train_mat[i]
            p1demo += sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0demo += sum(train_mat[i])
    # 条件概率的计算，用log防止溢出
    px_c1 = np.log(p1num/p1demo)
    px_c0 = np.log(p0num/p0demo)

    return px_c0, px_c1, pc1


def classify(testset, px_c0, px_c1, pc1):
    # 这里计算用testset * px_c 的原因是，testset是一个向量，这个向量为单词出现的次数，如果单词没有出现则为0
    p1 = sum(testset * px_c1) + np.log(pc1)
    p0 = sum(testset * px_c0) + np.log(1-pc1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    rate = [0] * 100
    errorindex = []
    for i in range(0, 100):
        rate[i], test_error = spamtest()
        errorindex = list(set(errorindex) | set(test_error))
    rate_average = sum(rate)/100
    print('100次重复测试的错误率为{}'.format(rate_average))
    errorspam, errorham = [], []
    for i in errorindex:
        if i % 2 == 1:
            errorham.append((i+1)//2)
        else:
            errorspam.append(i//2)
    print('errorham:', errorham)
    print('errorspam', errorspam)

