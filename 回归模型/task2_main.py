import pickle

from numpy import array

import task2 as t2


# 准备数据
# # f = open('E:/szb/华科生活/2021~2022/机器学习/机器学习第8次作业/2021年春季_第5章_回归模型_编程作业/附加逻辑回归实验/task2/train/train_data.txt')
# # train_data = [r.strip().split(" ") for r in f.readlines()]   # 用列表套列表的方式输入数据
# # f = open('E:/szb/华科生活/2021~2022/机器学习/机器学习第8次作业/2021年春季_第5章_回归模型_编程作业/附加逻辑回归实验/task2/test/test_data.txt')
# # test_data = [r.strip().split(" ") for r in f.readlines()]   # 用列表套列表的方式输入数据
# f = open('E:/szb/华科生活/2021~2022/机器学习/机器学习第8次作业/2021年春季_第5章_回归模型_编程作业/附加逻辑回归实验/task2/train/train_labels.txt')
# train_labels_Mat = [r.strip().split(" ") for r in f.readlines()]   # 用列表套列表的方式输入数据
# train_labels = []
# for i in train_labels_Mat:
#     for j in i:
#         train_labels.append(j)
# count = 0
# for i in train_labels:  # 标签数值化
#     i = int(i)
#     train_labels[count] = i
#     count += 1
#
# file = open('train_Mat', 'rb')      # 读取训练集和测试集特征向量
# train_Mat = pickle.load(file)
# file.close()
# file = open('test_Mat', 'rb')
# test_Mat = pickle.load(file)
# file.close()
# # datas = train_data[:]
# # datas.extend(test_data)
# # vocabList = t2.createVocabList(datas)    # 训练集和测试集的总单词集
# # train_Mat = []  # 存放每个单词出现的数量
# # test_Mat = []   # 存放每个单词出现的数量
# # for i in range(len(train_data)):
# #     wx = t2.bagOfWords2VecMN(vocabList, train_data[i])
# #     wx.append(1.0)  # 添加偏置
# #     train_Mat.append(wx)
# #     print('train_bag' + str(i) + 'done')
# # for i in range(len(test_data)):
# #     wx = t2.bagOfWords2VecMN(vocabList, test_data[i])
# #     wx.append(1.0)  # 添加偏置
# #     test_Mat.append(wx)
# #     print('test_bag' + str(i) + 'done')
# #
# # file_train = open('train_Mat', 'wb')    # 特征向量保存在工程文件夹中
# # pickle.dump(train_Mat, file_train)
# # file_train.close()
# # file_test = open('test_Mat', 'wb')
# # pickle.dump(test_Mat, file_test)
# # file_test.close()
# # 训练
# weight = t2.GradAscent(array(train_Mat), train_labels)
# # 预测
# labelMat = []   # 存放预测值
# for i in range(len(test_Mat)):
#     x = sum(weight * array(test_Mat[i]))
#     res = t2.sigmoid(x)
#     labelMat.append(res)
# # 写文件
# f = open('test_labels', 'w')
# for i in labelMat:
#     f.write(str(i) + '\n')
# f.close()

# 检测和贝叶斯模型预测结果不同的概率
f = open('E:/szb/华科生活/2021~2022/机器学习/机器学习第8次作业/2021年春季_第5章_回归模型_编程作业/附加逻辑回归实验/Logistic_regression/test_labels')
log_res = [r.strip().split("\n") for r in f.readlines()]   # 用列表套列表的方式输入数据
log = []
for i in log_res:
    for j in i:
        log.append(j)
f = open('E:/szb/华科生活/2021~2022/机器学习/机器学习第6次作业/2021年春季_第4章_贝叶斯模型_编程作业/2021年春季_第4章_贝叶斯模型_编程作业/贝叶斯模型/Bayesian_model/test_labels')
bay_res = [r.strip().split("\n") for r in f.readlines()]   # 用列表套列表的方式输入数据
bay = []
for i in bay_res:
    for j in i:
        bay.append(j)
error = 0
for i in range(len(log)):
    if log[i] != bay[i]:
        error = error + 1
print('differce with bay:', error / 25000.0)
