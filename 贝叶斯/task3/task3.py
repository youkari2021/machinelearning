import numpy as np
import pickle


def dataprocess(datas):
    processed_data = []
    for data in datas:
        res = [0] * 10000
        for da in data:
            da = int(da)
            res[da] += 1
        processed_data.append(res)
    return processed_data


def train(train_mat, train_class):
    textnum = len(train_mat)
    wordnum = len(train_mat[0])
    pc1 = sum(train_class) / textnum
    p0num = np.ones(wordnum)
    p1num = np.ones(wordnum)
    p0demo = 2
    p1demo = 2
    for i in range(textnum):
        if train_class[i] == 1:
            p1num += train_mat[i]
            p1demo += sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0demo += sum(train_mat[i])
    px_c1 = np.log(p1num / p1demo)
    px_c0 = np.log(p0num / p0demo)
    return px_c0, px_c1, pc1


def classify(testset, px_c0, px_c1, pc1):
    # 这里计算用testset * px_c 的原因是，testset是一个向量，这个向量为单词出现的次数，如果单词没有出现则为0
    p1 = sum(testset * px_c1) + np.log(pc1)
    p0 = sum(testset * px_c0) + np.log(1-pc1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':

    f = open(r'train/train_data.txt', 'r')
    traindatas = [da.split(' ') for da in f.read().splitlines(False)]
    processed_traindata = dataprocess(traindatas)
    f.close()
    file = open('processed_traindata', 'wb')
    pickle.dump(processed_traindata, file)
    file.close()
    # file = open('processed_traindata', 'rb')
    # processed_traindata = pickle.load(file)
    # file.close()
    #
    f = open(r'train/train_labels.txt', 'r')
    trainlabels = f.read().splitlines(False)
    f.close()
    file = open('trainlabels', 'wb')
    pickle.dump(trainlabels, file)
    file.close()
    # file = open('trainlabels', 'rb')
    # trainlabels = pickle.load(file)
    # file.close()
    #
    f = open(r'test/test_data.txt', 'r')
    testdatas = [da.split(' ') for da in f.read().splitlines(False)]
    processed_testdata = dataprocess(testdatas)
    f.close()
    file = open('processed_testdata', 'wb')
    pickle.dump(processed_testdata, file)
    file.close()
    # file = open('processed_testdata', 'rb')
    # processed_testdata = pickle.load(file)
    # file.close()

    for i in range(len(trainlabels)):
        trainlabels[i] = int(trainlabels[i])
    trainlabels = np.array(trainlabels)
    px_c0, px_c1, pc1 = train(processed_traindata, trainlabels)

    f = open(r'test/test_labels.txt', 'w')
    for testdata in processed_testdata:
        s = classify(testdata, px_c0, px_c1, pc1)
        s = str(s) + '\n'
        f.write(s)
    f.close()

