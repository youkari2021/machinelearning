# coding=gbk
import sklearn.tree as tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


# ����ԭʼ����
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
    # ��ȡ������Ԥ����
    traindata_ori = pd.read_csv('TrainDT.csv', encoding='gbk')  # utf-8�����ȡ�ᱨ��
    testdata_ori = pd.read_csv('TrainDT.csv', encoding='gbk')
    # print(testdata_ori.info())
    # �о�ssr����ֵû��ʲô�ã��жϵ�ʱ��ֱ����finLabel�����ж�
    # ��ȫȱʧ��ssr������ֵΪ100
    # imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-100)
    # testdata = pd.DataFrame(imp.fit_transform(testdata_ori))
    # traindata = pd.DataFrame(imp.fit_transform(traindata_ori))
    # traindata.columns = traindata_ori.columns
    # testdata.columns = testdata_ori.columns  # ����
    traindata = traindata_ori[['BSSIDLabel', 'RSSLabel', 'RoomLabel', 'finLabel']]
    testdata = testdata_ori[['BSSIDLabel', 'RSSLabel', 'RoomLabel', 'finLabel']]
    trainfins = traindata.groupby('finLabel')  # ��ָͬ����Ϊһ��
    testfins = testdata.groupby('finLabel')
    print(len(testfins))
    bssid = list(set(traindata['BSSIDLabel']))
    bssnums = len(bssid)

    # xΪbssid�����飬��СΪbssnum*������,һ�����ݴ�С��bssnum*1�����������д��ڵ�bss��Ϊ1
    # yΪroom������,x��һ�����ݶ�Ӧy��һ������
    trainx, trainy = dataprocess(trainfins, bssid)
    testx, testy = dataprocess(testfins, bssid)

    # ���⹹��������������Ԥ��
    wifitree = tree.DecisionTreeClassifier()
    wifitree.fit(trainx, trainy)
    predict = list(wifitree.predict(testx))
    print('ʵ�ʽ��', np.array(testy))
    print('Ԥ����', np.array(predict))
    print('����', accuracy_score(testy, predict))

