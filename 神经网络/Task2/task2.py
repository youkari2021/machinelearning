from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


def data_init():
    # 读取dat文件
    f = open('test/test_texts.dat', 'rb')
    test_texts = pickle.load(f)
    f.close()
    f = open('train/train_texts.dat', 'rb')
    train_texts = pickle.load(f)
    f.close()
    # 数据预处理
    vectorizer = TfidfVectorizer(max_features=10000)
    vectors_test = vectorizer.fit_transform(test_texts)
    vectors_train = vectorizer.fit_transform(train_texts)
    features_train = csr_matrix(vectors_train).toarray()
    features_test = csr_matrix(vectors_test).toarray()
    # 读取数据标签
    f = open('train/train_labels.txt', 'r')
    train_labels = np.loadtxt('train/train_labels.txt', delimiter='\n')
    train_labels = list(train_labels)
    f.close()
    print('数据处理完成')
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
