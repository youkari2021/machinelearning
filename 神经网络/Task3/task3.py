import numpy as np
import pickle
import copy
from sklearn.neural_network import MLPClassifier
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


def dataprocess(datas):
    processed_data = []
    for data in datas:
        res = [0] * 10000
        for da in data:
            da = int(da)
            res[da] += 1
        processed_data.append(res)
    return np.array(processed_data)


def data_init():
    f = open(r'train/train_data.txt', 'r')
    traindatas = [da.split(' ') for da in f.read().splitlines(False)]
    features_train = dataprocess(traindatas)
    f.close()
    f = open(r'train/train_labels.txt', 'r')
    train_labels = f.read().splitlines(False)
    f.close()
    f = open(r'test/test_data.txt', 'r')
    testdatas = [da.split(' ') for da in f.read().splitlines(False)]
    features_test = dataprocess(testdatas)
    f.close()
    print('数据初始化完成')
    return features_train, features_test, train_labels


def mpl_classfier(features_train, features_test, train_labels):
    x_train, x_test, y_train, y_test = train_test_split(features_train, train_labels, test_size=0.2)
    clf = MLPClassifier().fit(x_train, y_train)
    print('模型训练完成')
    train_accuracy = clf.score(x_test, y_test)
    train_error_rate = 1 - train_accuracy
    print('训练集精度{},错误率{}'.format(train_accuracy, train_error_rate))
    predict = clf.predict(features_test)
    print(predict)
    f = open('test/test_labels.txt', 'w')
    for i in predict:
        s = str(i) + '\n'
        f.write(s)
    f.close()
    pass


if __name__ == '__main__':
    features_train, features_test, train_labels = data_init()
    mpl_classfier(features_train, features_test, train_labels)


