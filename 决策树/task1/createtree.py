from math import log
import copy
from drawtree import *


def getent(Data):  # 计算熵
    datanum = len(Data)
    ent = 0
    dic = {}
    for i in range(datanum):
        if Data[i][-1] not in dic:
            dic[Data[i][-1]] = 1
        else:
            dic[Data[i][-1]] += 1
    for key in dic:
        p = dic[key] / datanum
        ent -= p * log(p, 2)
    return ent


def splitdata(Data, i, attr):  # 根据给出的一个属性划分数据集
    res = []
    for da in Data:
        if da[i] == attr:
            res.append(da[0:i]+da[i+1:])
    return res


def getbestattr(Data):
    # 计算maxGain(D,a)
    attrnum = len(Data[0])
    datanum = len(Data)
    ent = getent(Data)
    maxgain = 0
    bestattr = None
    for i in range(attrnum-1):
        attrlist = [da[i] for da in Data]
        dir = {}
        for attr in attrlist:
            if attr not in dir:
                dir[attr] = 1
            else:
                dir[attr] += 1
        attrs = dir.keys()
        entattr = 0
        for attr in attrs:
            subdata = splitdata(Data, i, attr)
            p = dir[attr]/datanum
            entattr += p * getent(subdata)
        gain = ent - entattr
        if gain > maxgain:
            maxgain = gain
            bestattr = i
    return bestattr


def maxnumlist(classlist):  # 选择样本中取值最多的类返回
    dic = {}
    for i in classlist:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] += 1
    a = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)  # 降序排列，选择最大的返回
    return a[0][0]


def treegenerate(Data, attr):
    classlist = [da[-1] for da in Data]
    if classlist.count(classlist[0]) == len(classlist):return classlist[0]  # d中样本全属于同一类别C，返回叶节点
    if not attr or len(Data[0]) == 1:return maxnumlist(classlist)  # 返回叶节点

    splitattr = getbestattr(Data)  # 通过计算最大Gain得到的属性在Data[i]中的序号
    attrname = attr[splitattr]  # 获取属性名字
    tree = {attrname: {}}
    del attr[splitattr]
    attrlist = [da[splitattr] for da in Data]
    attrvs = set(attrlist)
    for attrv in attrvs:
        subdata = splitdata(Data, splitattr, attrv)
        if subdata:tree[attrname][attrv] = treegenerate(subdata, copy.deepcopy(attr))  # 生成分支
        else:tree[attrname][attrv] = maxnumlist(classlist)  # Dv为空则叶节点标记为D中样本最多的类
    return tree


if __name__ == '__main__':
    with open('lenses.txt', 'r') as f:
        # data = f.read().splitlines(False)
        # res = []
        # for da in data:
        #     res.append(da.split('\t'))
        # 准备数据：先把txt文件读取到的字符串转化为二维数组，一个数组存储一行数据
        Data = [da.split('\t') for da in f.read().splitlines(False)]  # 一句话等于上面四句话
        # print(res)
        # print(Data)
        attr = ['age', 'eyetype', 'light', 'tear', 'class']
        tree = treegenerate(Data, attr)
        print(tree)
        draw(tree)


