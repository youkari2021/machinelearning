import test
import test2
import random
import networkx as nx
import matplotlib.pyplot as plt
import copy


f_k_j = 8
b_j = 10  # 约束7,简化不考虑这个情况
h_i = 15  # 约束4
s_i = 600  # 约束2 i上的s_k之和小于s_i
i_run_task_num = {}
i_run_task_storage = {}
# 用dk代表b_ak_vi
# G.nodes(i)ingreed-outgreed=w_k (if x_k_i==1) #约束13
# 假设b_j>=25*w_k,每个节点最多运行10个任务h_i=10
# 约束13、约束7需要对边的属性进行操作
# 任务需要在边上传输时，默认为最短路径传输,每条路径最多传输10个任务


# T为任务集，因为各任务参数相同，故用数字代表其任务序号
# G为有向图
def rrv(G, Task, E, mode):
    Tacc = []
    Tfail = []
    m = len(G)
    T = copy.deepcopy(Task)
    while T:
        k = T[-1]
        X = []
        # a = 0
        for j in range(m):
            X.append(0)
        # X[k.ak] = 1
        if mode == 0:
            a = k.ak
        else:
            a = random.randint(0, 14)
        X[a] = 1
        flag = 'run'
        while flag == 'run':
            flag, E = solution(G, k, Tacc, a, E)
        if flag:
            Tacc.append((k, k.ak))  # t_k add_in_to Tacc
        else:
            Tfail.append((k, k.ak))
        T.pop(-1)
    return Tacc, Tfail, E


# vi为任务运行的节点，k为任务实例，E为边的集合，G为有向拓扑图
def solution(G, k, Tacc, vi, E):
    # # constraint 4
    if vi not in i_run_task_num:
        i_run_task_num[vi] = 1
    elif i_run_task_num[vi] == h_i:
        return False, E
    else:
        i_run_task_num[vi] += 1

    # # constraint 2
    if k.sk > s_i:
        return False, E
    if vi not in i_run_task_storage:
        i_run_task_storage[vi] = k.sk
    elif i_run_task_storage[vi] + k.sk > s_i:
        return False, E
    else:
        i_run_task_storage[vi] += k.sk

    # # constraint 7,13,使用E作为边的属性
    if not nx.has_path(G, k.ak, vi):
        return False, E
    path = nx.shortest_path(G, source=k.ak, target=vi)
    if len(path) == 1:return True, E
    else:
        E_copy = copy.deepcopy(E)
        for node_index in range(len(path)-1):
            for edge_index in range(len(E_copy)):
                (node1, node2), edge_attr = E_copy[edge_index]
                if node1 == path[node_index] and node2 == path[node_index+1]:
                    if edge_attr['task_num'] == b_j:
                        return False, E
                    edge_attr['task_num'] += 1
                    E_copy[edge_index] = ((node1, node2), edge_attr)
                    break
        E = E_copy
    # h = G.in_degree(a) - G.out_degree(a)
    # if h != w_k/f_k_j:  # w_k/f_k_j不为整数的情况需要再考虑
    #     return False
    return True, E


if __name__ == '__main__':
    G, E = test.waxman_graph(15, 1, 1)
    T = []
    for i in range(200):
        T.append(test2.Task(random.randint(0, 14), 50, random.uniform(0.1, 5), 50, i))
    run_task_num = {}
    for i in range(len(G)):
        run_task_num[i] = 0
    mode = 1  # 0-在ak运行，1-随机选择节点运行
    Tacc, Tfail, E = rrv(G, T, E, mode)
    nx.draw(G, with_labels=True)

    # print(E)
    print(len(Tacc))
    # print(len(Tfail))
    # plt.show()

    # for i in range(len(E)):
    #     (a, b), attr = E[i]
    #     if a == 0:
    #         attr['a'] = 1
    #         E[i] = ((a, b), attr)
    #         print(E[i], 'a')
    #     elif b == 0:
    #         attr['b'] = 1
    #         E[i] = ((a, b), attr)
    #         print(E[i], 'b')
        # if 0 in edge:  # 通过这个判断节点连接的边，还可以进一步判断是入还是出
        #     edge[1] = 1
        #     print(edge)
    # if nx.has_path(G, 6, 2):
    #     print(nx.shortest_path(G, source=6, target=2))
    # else:print('no path')