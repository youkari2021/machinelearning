import pandas as pd
import numpy as np
import re
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
import sklearn.tree as tree
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression


def data_init():
    # 读取数据
    train_data = pd.read_csv('train.tsv', delimiter='\t', header=None)
    test_data = pd.read_csv('test.tsv', delimiter='\t', header=None)

    train_arg1, train_arg2, train_label = train_data.iloc[:, 2], train_data.iloc[:, 3], train_data.iloc[:, 1]
    test_arg1, test_arg2 = test_data.iloc[:, 2], test_data.iloc[:, 3]

    # 对类别标签编码
    class_dict = {'Comparison': 0, 'Contingency': 1, 'Expansion': 2, 'Temporal': 3}
    train_label = np.array([class_dict[label] for label in train_label])

    # 加载词向量文件
    word_vec = pickle.load(open('glove_300.pickle', 'rb'))

    train_arg1_feature = get_feature(train_arg1, word_vec)  # 提取训练集中所有论元1的特征
    train_arg2_feature = get_feature(train_arg2, word_vec)
    test_arg1_feature = get_feature(test_arg1, word_vec)
    test_arg2_feature = get_feature(test_arg2, word_vec)
    print('特征提取完成')
    train_feature = np.concatenate((train_arg1_feature, train_arg2_feature), axis=1)  # 将论元1和论元2的特征拼接
    test_feature = np.concatenate((test_arg1_feature, test_arg2_feature), axis=1)
    print('特征拼接完成')

    return train_feature, train_label, test_feature


# 提取论元特征
def get_feature(data, word_vec):
    feature = np.zeros((len(data), 300), dtype=np.float32)  # 初始化所有论元的特征向量
    for i, arg in enumerate(data):
        arg = re.sub(r'[^A-Za-z0-9 ]+', '', arg)  # 删去论元中除 A-Z,a-z,0-9,空格 之外的字符
        arg = arg.split(' ')                      # 根据空格将论元划分为list
        vector = np.zeros(300, dtype=np.float32)  # 初始化当前论元的特征向量
        for word in arg:
            vector += word_vec[word]              # 对论元中所有词的词向量求和
        feature[i] = vector / len(arg)            # 取平均，作为当前论元的特征向量

    return feature


def aggrerate(clfs, test_feature, weights):
    # 计算测试集预测结果并保存
    pred = []
    for i in range(len(weights)):
        test_pred = clfs[i].predict(test_feature)
        pred.append(test_pred)
    test_pred = []
    for j in range(len(pred[0])):
        dirc = {}
        for i in range(len(weights)):
            if pred[i][j] not in dirc:
                dirc[pred[i][j]] = pred[i][j] * weights[i]
                # dirc[pred[i][j]] = pred[i][j]
            else:
                dirc[pred[i][j]] += pred[i][j] * weights[i]
                # dirc[pred[i][j]] += pred[i][j]
        predmaxword, predmaxkey = -1, -1
        for key, word in dirc.items():
            if predmaxword < word:
                predmaxword = word
                predmaxkey = key
        test_pred.append(predmaxkey)
    print(test_pred)
    return test_pred


# train_feature, train_label, test_feature = data_init()
# x_train, x_test, y_train, y_test = train_test_split(train_feature, train_label, test_size=0.2)
#
# clfs = []
# # # SVM分类
# clf = svm.SVC(decision_function_shape='ovo')
# clf.fit(x_train, y_train)
# clfs.append(clf)
# print('支持向量机完成')
# #
# # # 决策树分类
# clf = tree.DecisionTreeClassifier(max_depth=4)
# clf.fit(x_train, y_train)
# clfs.append(clf)
# print('决策树完成')
# #
# # # 神经网络
# clf = MLPClassifier()
# clf.fit(x_train, y_train)
# clfs.append(clf)
# print('神经网络完成')
# #
# # # 贝叶斯
# # # clf = GaussianNB()
# clf = BernoulliNB()
# clf.fit(x_train, y_train)
# clfs.append(clf)
# print('贝叶斯完成')
# #
# # # 逻辑回归
# clf = LogisticRegression(max_iter=1000)
# clf.fit(x_train, y_train)
# clfs.append(clf)
# print('逻辑回归完成')
#
# weights = []
# # # 计算训练集上的Acc和F1
# for clf in clfs:
#     x_test_pred = clf.predict(x_test)
#     train_acc = accuracy_score(y_test, x_test_pred)
#     train_f1 = f1_score(y_test, x_test_pred, average='macro')
#     print(f'Train Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')
#     # weights.append((train_acc+train_f1)/2)
#     weights.append(train_f1)
# #
# # test_pred = aggrerate(clfs, test_feature, weights)
# re_x_test_pred = aggrerate(clfs, x_test, weights)
# re_x_train_pred = aggrerate(clfs, x_train, weights)
# # with open('test_pred.txt', 'w') as f:
# #     for label in test_pred:
# #         f.write(str(label) + '\n')
# # f.close()
# # re_x_train_pred = clf.predict(x_train)
# train_acc = accuracy_score(y_train, re_x_train_pred)
# train_f1 = f1_score(y_train, re_x_train_pred, average='macro')
# print(f'Train Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')
# # re_x_test_pred = clf.predict(x_test)
# train_acc = accuracy_score(y_test, re_x_test_pred)
# train_f1 = f1_score(y_test, re_x_test_pred, average='macro')
# print(f'Test Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')

