# coding=gbk
import sklearn.tree as tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 只处理 Pclass Sex Age


def dataprocess(data):
    res = []
    for index, row in data.iterrows():
        res.append(list(row))
        # print(res)
    return res


def dataclasschange(data, datanum):
    for i in range(datanum):
        if data.loc[i, 'Age'] < 18:  # 用loc方法可以修改单个值
            data.loc[i, 'Age'] = 1
        elif data.loc[i, 'Age'] < 45:
            data.loc[i, 'Age'] = 2
        elif data.loc[i, 'Age'] < 65:
            data.loc[i, 'Age'] = 3
        elif data.loc[i, 'Age'] < 100:
            data.loc[i, 'Age'] = 4
        elif data.loc[i, 'Age'] == 100:
            data.loc[i, 'Age'] = 0
        if data.loc[i, 'Sex'] == 'male':
            data.loc[i, 'Sex'] = 0
        else:
            data.loc[i, 'Sex'] = 1
    return data


def fareprocess(data):
    fare = data['Fare']
    fare = list(fare)

    for i in range(len(fare)):
        if fare[i] != fare[i]:
            fare[i] = 0
        fare[i] = int(fare[i]) // 20
    fare = pd.DataFrame({'Fare': fare})
    return fare


def familynumprocess(data):
    family = data[['Parch']]

    # num = len(family)
    # for i in range(num):
    #     if family.loc[i, 'SibSp'] > 0:
    #         family.loc[i, 'SibSp'] = 1
    #     if family.loc[i, 'Parch'] > 0:
    #         family.loc[i, 'Parch'] = 1
    return family


if __name__ == "__main__":
    traindata_ori = pd.read_csv('train.csv', encoding='gbk')
    testdata_ori = pd.read_csv('test.csv', encoding='gbk')
    trainfare = fareprocess(traindata_ori)
    testfare = fareprocess(testdata_ori)
    trainfamily = familynumprocess(traindata_ori)
    testfamily = familynumprocess(testdata_ori)

    # print(traindata_ori.info())
    # print(testdata_ori.info())

    traindata_ori['Age'].fillna(-100, inplace=True)
    testdata_ori['Age'].fillna(-100, inplace=True)
    traindata = traindata_ori[['Pclass', 'Sex', 'Age']]
    testdata = testdata_ori[['Pclass', 'Sex', 'Age']]
    # traindata = traindata.join(trainfare)
    # testdata = testdata.join(testfare)
    traindata = traindata.join(trainfamily)
    testdata = testdata.join(testfamily)
    trainsurvive = traindata_ori[['Survived']]
    traindata_num = len(traindata)
    testdata_num = len(testdata)
    traindata = dataclasschange(traindata, traindata_num)
    testdata = dataclasschange(testdata, testdata_num)
    trainx = dataprocess(traindata)

    trainy = dataprocess(trainsurvive)
    testx = dataprocess(testdata)
    # print(trainx)
    # print(testx)
    titanic_tree = tree.DecisionTreeClassifier()
    titanic_tree.fit(trainx, trainy)

    predict = list(titanic_tree.predict(testx))
    order = [(traindata_num+i+1) for i in range(testdata_num)]
    res = {'PassengerId': order, 'Survived': predict}
    res = pd.DataFrame(res)
    testdata_ori.merge(res)[['Pclass', 'Sex', 'Age', 'Parch', 'Survived']].to_csv('mytest.csv', index=False)
    # res.to_csv('gender_submission.csv', index=False)




