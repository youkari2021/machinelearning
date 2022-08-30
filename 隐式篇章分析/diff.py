# f = open(r'test_pred.txt', 'r')
# labels_huigui = f.read().splitlines(False)
# f.close()
# f = open(r'test_pred_神经网络.txt', 'r')
# labels_beiyesi = f.read().splitlines(False)
# f.close()
# print(len(labels_huigui), len(labels_beiyesi))
# l = len(labels_beiyesi)
# difcount = 0
# for i in range(l):
#     if labels_huigui[i] != labels_beiyesi[i]:
#         difcount += 1
# print('相差率为', difcount/l)
import numpy as np
pred1 = np.loadtxt('test_pred.txt')
pred2 = np.loadtxt('test_pred_支持向量机.txt')
pred3 = np.loadtxt('test_pred_五集成支持向量机2.txt')
l = len(pred1)
a = sum(pred2 == pred1)
print('相同率为', a/l)
