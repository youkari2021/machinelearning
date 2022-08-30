import networkx.generators.geometric as ngx
import networkx as nx
import matplotlib.pyplot as plt
from bisect import bisect_left
from itertools import accumulate, combinations, product
from math import sqrt
import math
import random


def waxman_graph(
    n, beta=0.4, alpha=0.1, L=None, domain=(0, 0, 1, 1), metric=None
):
    nodes = [i for i in range(n)]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    (xmin, ymin, xmax, ymax) = domain
    # Each node gets a uniformly random position in the given rectangle.
    pos = {v: (random.uniform(xmin, xmax), random.uniform(ymin, ymax)) for v in G}
    nx.set_node_attributes(G, pos, "pos")
    # If no distance metric is provided, use Euclidean distance.
    if metric is None:
        metric = euclidean

    if L is None:
        L = max(metric(x, y) for x, y in combinations(pos.values(), 2))
        def dist(u, v):
            return metric(pos[u], pos[v])
    else:
        def dist(u, v):
            return random.random() * L
    # `pair` is the pair of nodes to decide whether to join.
    def should_join(pair):
        return random.random() < beta * math.exp(-dist(*pair) / (alpha * L))

    a =filter(should_join, reverse(combinations(G, 2)))
    G.add_edges_from(a, task={})
    E = []
    for edge in G.edges():
        E.append((edge, {'task_num': 0}))
    return G, E


def euclidean(x, y):
    """Returns the Euclidean distance between the vectors ``x`` and ``y``.

    Each of ``x`` and ``y`` can be any iterable of numbers. The
    iterables must be of the same length.

    """
    return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


def reverse(combination):
    comb1 = []
    for i in combination:
        if random.random() <= 0.5:
            comb1.append((i[1], i[0]))
        else:comb1.append(i)
    return comb1


w_k = 20  # MB =b_av
f_k_j = min(w_k, 30)
b_j = 25*f_k_j  # 约束7
h_i = 10
# G.nodes(i)ingreed-outgreed=w_k (if x_k_i==1) #约束13
# 假设b_j=25*w_k,每个节点最多运行10个任务h_i=10
if __name__ == '__main__':
    H, E = waxman_graph(15, 0.8, 0.8)
    print(H.edges(data=True))
    nx.draw(H, with_labels=True)
    plt.show()
    print(len(H.edges))
