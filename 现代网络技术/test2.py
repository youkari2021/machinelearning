import test
import test3
import random


class Task:
    def __init__(self, ak, wk, dk, sk, num):
        self.ak = ak
        self.wk = wk
        self.dk = dk
        self.sk = sk
        self.num = num


# 前提是任务参数都相同
def generate(m, n, k):  # m为节点个数，n为边的个数，k为任务个数
    X = []
    for j in range(k):
        temp = []
        for i in range(m):
            temp.append(0)
        a = random.randint(0, m-1)
        temp[a] = 1
        X.append(temp)
    print(X)
    pass


if __name__ == '__main__':
    alpha = 0.8
    beta = 0.8
    tasknum = 50
    G, E = test.waxman_graph(15, alpha, beta)
    T = []
    while tasknum <= 400:
        temp = []
        for i in range(tasknum):
            temp.append(Task(random.randint(0, 14), 50, random.uniform(0.1, 5), random.uniform(1, 50), i))
        T.append(temp)
        tasknum += 50
    tacc0 = []
    tacc1 = []
    for t in T:
        a = 0
        b = 0
        for i in range(20):
            Tacc, _, __ = test3.rrv(G, t, E, 0)
            a += len(Tacc)
            Tacc, _, __ = test3.rrv(G, t, E, 1)
            b += len(Tacc)
        tacc0.append(a/20)
        tacc1.append(b/20)
    print(tacc0)
    print(tacc1)



