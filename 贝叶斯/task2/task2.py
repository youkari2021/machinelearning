import numpy as np
from struct import unpack
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import math

# 配置文件
config = {
    # 训练集文件
    'train_images_idx3_ubyte_file_path': 'data/train-images.idx3-ubyte',
    # 训练集标签文件
    'train_labels_idx1_ubyte_file_path': 'data/train-labels.idx1-ubyte',

    # 测试集文件
    'test_images_idx3_ubyte_file_path': 'data/t10k-images.idx3-ubyte',
    # 测试集标签文件
    'test_labels_idx1_ubyte_file_path': 'data/t10k-labels.idx1-ubyte',

    # 特征提取阙值
    'binarization_limit_value': 0.14,

    # 特征提取后的边长
    'side_length': 14
}


def decode_idx3_ubyte(path):
    '''
    解析idx3-ubyte文件，即解析MNIST图像文件
    '''

    '''
    也可不解压，直接打开.gz文件。path是.gz文件的路径
    import gzip
    with gzip.open(path, 'rb') as f:
    '''
    print('loading %s' % path)
    with open(path, 'rb') as f:
        # 前16位为附加数据，每4位为一个整数，分别为幻数，图片数量，每张图片像素行数，列数。
        magic, num, rows, cols = unpack('>4I', f.read(16))
        print('magic:%d num:%d rows:%d cols:%d' % (magic, num, rows, cols))
        mnistImage = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    print('done')
    return mnistImage


def decode_idx1_ubyte(path):
    '''
    解析idx1-ubyte文件，即解析MNIST标签文件
    '''
    print('loading %s' % path)
    with open(path, 'rb') as f:
        # 前8位为附加数据，每4位为一个整数，分别为幻数，标签数量。
        magic, num = unpack('>2I', f.read(8))
        print('magic:%d num:%d' % (magic, num))
        mnistLabel = np.fromfile(f, dtype=np.uint8)
    print('done')
    return mnistLabel


def normalizeImage(image):
    '''
    将图像的像素值正规化为0.0 ~ 1.0
    '''
    res = image.astype(np.float32) / 255.0
    return res


def load_train_images(path=config['train_images_idx3_ubyte_file_path']):
    return normalizeImage(decode_idx3_ubyte(path))


def load_train_labels(path=config['train_labels_idx1_ubyte_file_path']):
    return decode_idx1_ubyte(path)


def load_test_images(path=config['test_images_idx3_ubyte_file_path']):
    return normalizeImage(decode_idx3_ubyte(path))


def load_test_labels(path=config['test_labels_idx1_ubyte_file_path']):
    return decode_idx1_ubyte(path)


def oneImagesFeatureExtraction(image):
    '''
    对单张图片进行特征提取
    '''
    res = np.empty((config['side_length'], config['side_length']))
    num = 28//config['side_length']
    for i in range(0, config['side_length']):
        for j in range(0, config['side_length']):
            # tempMean = (image[2*i:2*(i+1),2*j:2*(j+1)] != 0).sum()/(2 * 2)
            tempMean = image[num*i:num*(i+1), num*j:num*(j+1)].mean()
            if tempMean > config['binarization_limit_value']:
                res[i, j] = 1
            else:
                res[i, j] = 0
    # plt.plot(res)
    # plt.show()
    return res


def featureExtraction(images):
    res = np.empty((images.shape[0], config['side_length'],
                    config['side_length']), dtype=np.float32)
    for i in range(images.shape[0]):
        res[i] = oneImagesFeatureExtraction(images[i])
        print(i)
    return res


def zero_one_numExtration(feature):
    nums = [0, 0]
    for i in range(config['side_length']):
        for j in range(config['side_length']):
            if feature[i][j] == 0:nums[0] += 1
            else:nums[1] += 1
    return nums


def train(train_mat, trainlabels):

    train_class = trainlabels[:]
    imgnum = len(train_mat)
    matlen = len(train_mat[0])
    pinum = np.ones((10, matlen))
    pidemo = np.ones(10) * 2
    px_c = np.zeros((10, matlen))
    for i in range(imgnum):
        pinum[train_class[i]] += train_mat[i]
        pidemo[train_class[i]] += sum(train_mat[i])
    for i in range(10):
        px_c[i] = np.log(pinum[i]/pidemo[i])
    return px_c


def classify(testset, px_c, log_pc):
    pi = [0] * 10
    for i in range(10):
        pi[i] = sum(testset * px_c[i]) + log_pc[i]

    # pi = list(pi)
    return pi.index(max(pi))


def matprocess(features):
    mats = []
    for feature in features:
        a = np.reshape(feature, -1)
        mats.append(a)
    return mats


if __name__ == '__main__':
    # trainimgs = load_train_images()
    trainlabels = load_train_labels()
    # testimgs = load_test_images()
    testlabels = load_test_labels()
    # trainimgfeatures = featureExtraction(trainimgs)
    # testimgfeatures = featureExtraction(testimgs)
    import pickle
    # file = open('trainimgfeatures', 'wb')
    # pickle.dump(trainimgfeatures, file)
    # file.close()
    # file = open('testimgfeatures', 'wb')
    # pickle.dump(testimgfeatures, file)
    # file.close()
    file = open('trainimgfeatures', 'rb')
    trainimgfeatures = pickle.load(file)
    file.close()
    file = open('testimgfeatures', 'rb')
    testimgfeatures = pickle.load(file)
    file.close()

    labels = [0]*10
    for i in trainlabels:
        labels[i] += 1
    log_pc = labels[:]
    for i in range(10):
        log_pc[i] = np.log(log_pc[i] / 60000)

    train_mat = matprocess(trainimgfeatures)
    test_mat = matprocess(testimgfeatures)
    px_c = train(train_mat, trainlabels)

    error_count = 0
    for i in range(len(test_mat)):
        if classify(test_mat[i], px_c, log_pc) != testlabels[i]:
            error_count += 1
    accute = error_count/len(testlabels)
    print('出错率为', accute)
