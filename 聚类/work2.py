#-*- codeing = utf-8 -*-
#@Time : 2022/4/29 16:37
#@Author : Healive
#@File : work2.py
#@Software : PyCharm

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def missingHandling(incompleteData):
    '''缺失值处理'''
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-100)
    completeData = pd.DataFrame(imp.fit_transform(incompleteData))
    completeData.columns = incompleteData.columns
    return completeData


def dataProcess(data, BSSIDs):
    '''数据处理'''
    fdata = data[['BSSIDLabel', 'RSSLabel', 'RoomLabel', 'finLabel']]
    groupedData = fdata.groupby('finLabel')
    processedData = []
    processedClasses = []
    for name, group in groupedData:
        temp = np.array(group['BSSIDLabel'])
        example = len(BSSIDs) * [-133]
        for BSSID in BSSIDs:
            if BSSID in temp:
                pos = list(group['BSSIDLabel']).index(BSSID)
                example[BSSIDs.index(BSSID)] = list(group['RSSLabel'])[pos]
                # example[BSSIDs.index(BSSID)] = 1
        processedData.append(example)
        room = np.array(group['RoomLabel'])
        processedClasses.append(room[1])
    processedData = np.array(processedData)
    return processedData, processedClasses


def modelEva(dataSet, y, y_pred, k):
    score = metrics.davies_bouldin_score(dataSet, y_pred)
    print('当k={}时，模型的DB指数为{}'.format(k,score))
    # temp = 0
    # standlabel = 0
    # for i in range(len(y_pred)):
    #     if y_pred[i] != temp:
    #         temp = y_pred[i]
    #         standlabel += 1
    #     y_pred[i] = standlabel
    # score = accuracy_score(y_pred, y)
    # print('将聚类结果与样本标签对比，相同标签聚到一起的概率为{}%'.format(score*100))


def clusterAnalysis(mod, featureData):
    # 聚成4类数据，统计每个聚类下的数据量，并且求出他们的中心
    r1 = pd.Series(mod.labels_).value_counts()
    r2 = pd.DataFrame(mod.cluster_centers_)
    r = pd.concat([r2, r1], axis=1)
    r.columns = list(featureData.columns) + [u'类别数目']
    # print(r)
    # 给每一条数据标注上被分为哪一类
    r = pd.concat([featureData, pd.Series(mod.labels_, index=featureData.index)], axis=1)
    r.columns = list(featureData.columns) + [u'聚类类别']
    # print(r.head())
    return r


def clusterVisual(r, k):
    # 可视化过程
    ts = TSNE()
    ts.fit_transform(r)
    ts = pd.DataFrame(ts.embedding_, index=r.index)
    tagList = ['r.', 'go', 'b*', 'k+', 'yD']
    for i in range(k):
        a = ts[r[u'聚类类别'] == i]
        plt.plot(a[0], a[1], tagList[i])
    plt.show()


if __name__ == '__main__':
    datafile1 = r'Task2/DataSetKMeans1.csv'
    datafile2 = r'Task2/DataSetKMeans2.csv'
    data = pd.read_csv(datafile1, encoding='gbk')
    data = DataFrame(data)
    # pd.DataFrame.info(data)
    # print(data.head())
    data = missingHandling(data)
    BSSIDs = list(set(data['BSSIDLabel']))
    featureData, labels = dataProcess(data, BSSIDs)
    # print(len(BSSIDs))  # 266
    # print(featureData.shape)    #(328,266)
    # print(len(labels))  #328
    # 聚类
    featureData = DataFrame(featureData)
    featureData.columns = BSSIDs
    kList = [2, 3, 4, 5]
    for k in kList:
        mod = KMeans(n_clusters=k, max_iter=50)
        mod.fit_predict(featureData)
        y_pred = mod.labels_
        modelEva(featureData, labels, y_pred, k)
        r = clusterAnalysis(mod, featureData)
        clusterVisual(r, k)


