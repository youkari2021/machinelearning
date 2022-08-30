import numpy as np
import pickle
import copy


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def stoc_grad_ascent(dataMatIn, classLabels):
    m, n = np.shape(dataMatIn)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatIn[i] * weights))  # 数值计算
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatIn[i]
        print(i)
    return weights


def dataprocess(datas):
    processed_data = []
    for data in datas:
        res = [0] * 10000
        for da in data:
            da = int(da)
            res[da] += 1
        processed_data.append(res)
    return np.array(processed_data)


def classify(data, beta):
    y = sigmoid(sum(data * beta))
    if y > 0.5:
        return 1
    return 0


def pridict(beta, test_datax):
    f = open(r'test/test_labels1.txt', 'w')
    for data in test_datax:
        label = classify(np.array(data), beta)
        label = str(label) + '\n'
        f.write(label)
        print(label)
    f.close()


if __name__ == '__main__':

    # f = open(r'train/train_data.txt', 'r')
    # traindatas = [da.split(' ') for da in f.read().splitlines(False)]
    # processed_traindata = dataprocess(traindatas)
    # f.close()
    # file = open('processed_traindata', 'wb')
    # pickle.dump(processed_traindata, file)
    # file.close()
    file = open('processed_traindata', 'rb')
    processed_traindata = pickle.load(file)
    file.close()

    # f = open(r'train/train_labels.txt', 'r')
    # trainlabels = f.read().splitlines(False)
    # f.close()
    # file = open('trainlabels', 'wb')
    # pickle.dump(trainlabels, file)
    # file.close()
    file = open('trainlabels', 'rb')
    trainlabels = pickle.load(file)
    file.close()
    # #
    # f = open(r'test/test_data.txt', 'r')
    # testdatas = [da.split(' ') for da in f.read().splitlines(False)]
    # processed_testdata = dataprocess(testdatas)
    # f.close()
    # file = open('processed_testdata', 'wb')
    # pickle.dump(processed_testdata, file)
    # file.close()
    file = open('processed_testdata', 'rb')
    processed_testdata = pickle.load(file)
    file.close()

    for i in range(len(trainlabels)):
        trainlabels[i] = int(trainlabels[i])
    trainlabels = np.array(trainlabels)
    processed_traindata = np.insert(processed_traindata, -1, 1, axis=1)
    processed_traindata_T = copy.deepcopy(processed_traindata.T)
    beta = stoc_grad_ascent(processed_traindata, trainlabels)
    processed_testdata = np.insert(processed_testdata, -1, 1, axis=1)
    pridict(beta, processed_testdata)

