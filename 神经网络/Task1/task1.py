from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import copy


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


def GradAscent(dataMat, labelMat, numIter=300):   # 第三个参数为迭代次数
    m, n = dataMat.shape
    weight = np.ones(n)    # 初始化权值w
    for i in range(numIter):
        print('第' + str(i) + '次迭代开始')
        dataInd = list(range(m))  # 剩余数据下标
        for j in range(m):
            # study_rate = 4/(1.0 + i + j) + 0.01  # 加一个常数使学习率不会趋于0（新数据永远对权值有影响）,当迭代次数较小时，j占主导，有可能随机跳出局部最优
            study_rate = 0.01
            randInd = int(np.random.uniform(0, len(dataInd)))
            del(dataInd[randInd])
            pred = sigmoid(sum(dataMat[randInd] * weight))
            error = labelMat[randInd] - pred    # error有正有负，影响下一行的移动方向
            weight = weight + study_rate * float(error) * dataMat[randInd]
        print('第'+str(i)+'次迭代结束')
    return weight


def gradient(x, xt, y):
    m, n = x.shape
    y = y.reshape(-1, 1)
    alpha = 0.001
    deta = 0.00001
    maxcycle = 200
    beta = np.ones((n, 1))
    for i in range(maxcycle):
        h = sigmoid(np.matmul(x, beta))
        error = y - h
        tem = beta
        beta = beta + alpha * np.matmul(xt, error)
        if abs(sum(beta - tem)) < deta:
            break
        print(i)
    return beta


def classify(data, beta):
    aa = data * beta
    a = sum(aa)
    y = sigmoid(a)
    if y > 0.5:
        return 1
    return 0


def predict(beta, test_set, test_lables):
    error = 0
    num, _ = test_set.shape
    for i in range(num):
        # label = classify(test_set[i].reshape(-1, 1), beta)  # gradient用
        label = classify(test_set[i], beta)  # GradAscent用
        if label != test_lables[i]:
            error += 1
    print('错误数{}, 总预测数{}, 错误率{}'.format(error, num, error/num))


def colicSklearn(train_set, train_lables, test_set, test_lables):

    classifier = LogisticRegression(solver='liblinear', max_iter=10).fit(train_set, train_lables)
    test_accurcy = classifier.score(test_set, test_lables)
    error_rate = 1 - test_accurcy
    print('错误率', error_rate)


def data_preprocess():
    train_data = np.loadtxt('horseColicTraining.txt', delimiter="\t")
    train_set = train_data[:, :-1]
    train_lables = train_data[:, -1:]

    test_data = np.loadtxt('horseColicTest.txt', delimiter="\t")
    test_set = test_data[:, :-1]
    test_lables = test_data[:, -1:]

    return train_set, train_lables, test_set, test_lables


if __name__ == '__main__':
    train_set, train_lables, test_set, test_lables = data_preprocess()
    colicSklearn(train_set, train_lables, test_set, test_lables)
    beta = GradAscent(train_set, train_lables)
    # beta = gradient(train_set, copy.deepcopy(train_set.T), train_lables)
    predict(beta, test_set, test_lables)

