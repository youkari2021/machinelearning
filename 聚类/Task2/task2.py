import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pandas import DataFrame


def data_init(path):
    ori = pd.read_csv(path, encoding='gbk')
    bssid_list = list(set(ori['BSSIDLabel']))
    room_list = ori['RoomLabel'].tolist()
    fin_list = ori['finLabel'].tolist()
    rss_list = ori['RSSLabel'].tolist()
    return ori, bssid_list, room_list, fin_list, rss_list


def data_process(path):
    ori, bssid_list, room_list, fin_list, rss_list = data_init(path)
    f_ori = ori.groupby('finLabel')
    bssnums = len(bssid_list)
    x, y = [], []
    for finlabel, v in f_ori:
        finlabellist = list(v['BSSIDLabel'])
        rsslabellist = list(v['RSSLabel'])
        fi = [-100] * bssnums
        for bss in bssid_list:
            if bss in finlabellist:
                fi[bssid_list.index(bss)] = rsslabellist[finlabellist.index(bss)]
        x.append(fi)
        roomlabel = list(v['RoomLabel'])
        y.append(roomlabel[0])
    return x, y, bssid_list


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
    # train_x, train_y, bssid_list = data_process('DataSetKMeans1.csv')
    train_x, train_y, bssid_list = data_process('DataSetKMeans2.csv')

    K = [2, 3, 4, 5]
    for k in K:
        model = KMeans(k)
        model.fit_predict(train_x)
        pred_y = model.labels_
        dbi = davies_bouldin_score(train_x, pred_y)
        print('dbi = {a}, k = {b}'.format(a=dbi, b=k))
        x = DataFrame(train_x)
        x.columns = bssid_list
        r = clusterAnalysis(model, x)
        clusterVisual(r, k)






