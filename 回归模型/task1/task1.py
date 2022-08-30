import numpy as np
import copy
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def data_process(datas):
    x, y = [], []
    l = len(datas[0])
    for da in datas:
        tem = []
        for i in range(l-1):
            tem.append(float(da[i]))
        tem.append(1)
        x.append(tem)
        y.append(int(da[l-1]))
    xdata = np.array(x)
    ydata = np.array(y)
    return xdata, ydata


def init_data():
    f = open(r'钞票训练集.txt', 'r')
    datas = [da.split(',') for da in f.read().splitlines(False)]
    x, y = data_process(datas)
    xt = copy.deepcopy(x.T)

    return x, xt, y


def test_data():
    f = open(r'钞票测试集.txt', 'r')
    datas = [da.split(',') for da in f.read().splitlines(False)]
    x = []
    for da in datas:
        tem = []
        for i in da:
            tem.append(float(i))
        tem.append(1)
        x.append(tem)
    f.close()
    return x


def gradient(x, xt, y):
    m, n = x.shape
    y = y.reshape(-1, 1)
    alpha = 0.001
    deta = 0.000001
    maxcycle = 500
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


def pridict(beta):
    test_datax = test_data()
    testdata = []
    i = 0
    for data in test_datax:
        i += 1
        label = classify(np.array(data), beta)
        data[-1] = label
        data.insert(0, i)
        testdata.append(data)

    data_pd = pd.DataFrame(testdata, columns=['样本序号', '变量名1', '变量名2', '变量名3', '变量名4', '真钞or假钞'])
    data_pd.to_csv('莫茗程_U201913602_电信1904_测试集结果.csv', encoding='utf-8', index_label='样本序号', index=False)


def classify(data, beta):
    # data = data.reshape(-1, 1)
    y = sigmoid(sum(np.matmul(beta.T, data.T)))
    if y > 0.5:
        return 1
    return 0


if __name__ == "__main__":
    x, xt, y = init_data()
    beta = gradient(x, xt, y)
    pridict(beta)
    print(beta)
