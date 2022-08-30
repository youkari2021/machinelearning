# coding=gbk
import sklearn.tree as tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


# 处理原始数据
def dataprocess(fins, bssid):
    x, y = [], []
    for finlabel, v in fins:
        # print(finlabel, len(list(v['BSSIDLabel'])))
        # finlabellist = list(v['BSSIDLabel'])
        # print(v['RSSLabel'])
        finlabellist = list(v['BSSIDLabel'])
        rsslabellist = list(v['RSSLabel'])
        # fi = [0] * bssnums
        fi = [-100] * bssnums
        for bss in bssid:
            if bss in finlabellist:
                # fi[bssid.index(bss)] = 1
                fi[bssid.index(bss)] = rsslabellist[finlabellist.index(bss)]

        x.append(fi)
        roomlabel = list(v['RoomLabel'])
        # print(len(roomlabel))
        y.append(roomlabel[0])
    return x, y


if __name__ == "__main__":
    # 读取与数据预处理
    traindata_ori = pd.read_csv('TrainDT.csv', encoding='gbk')  # utf-8编码读取会报错
    testdata_ori = pd.read_csv('TrainDT.csv', encoding='gbk')
    # print(testdata_ori.info())
    # 感觉ssr的数值没有什么用，判断的时候直接用finLabel进行判断
    # 补全缺失的ssr，设置值为100
    # imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-100)
    # testdata = pd.DataFrame(imp.fit_transform(testdata_ori))
    # traindata = pd.DataFrame(imp.fit_transform(traindata_ori))
    # traindata.columns = traindata_ori.columns
    # testdata.columns = testdata_ori.columns  # 对齐
    traindata = traindata_ori[['BSSIDLabel', 'RSSLabel', 'RoomLabel', 'finLabel']]
    testdata = testdata_ori[['BSSIDLabel', 'RSSLabel', 'RoomLabel', 'finLabel']]
    trainfins = traindata.groupby('finLabel')  # 相同指纹作为一组
    testfins = testdata.groupby('finLabel')
    print(len(testfins))
    bssid = list(set(traindata['BSSIDLabel']))
    bssnums = len(bssid)

    # x为bssid向量组，大小为bssnum*数据量,一条数据大小是bssnum*1。该条数据有存在的bss就为1
    # y为room向量组,x中一条数据对应y的一个数据
    trainx, trainy = dataprocess(trainfins, bssid)
    testx, testy = dataprocess(testfins, bssid)

    # 调库构建决策树并进行预测
    wifitree = tree.DecisionTreeClassifier()
    wifitree.fit(trainx, trainy)
    predict = list(wifitree.predict(testx))
    print('实际结果', np.array(testy))
    print('预测结果', np.array(predict))
    print('精度', accuracy_score(testy, predict))

