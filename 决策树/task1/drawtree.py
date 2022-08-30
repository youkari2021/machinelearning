import matplotlib.pyplot as plt

decision_node = dict(boxstyle="square", fc="0.75")
leaf_node = dict(boxstyle="circle", fc="0.9")
arrow_args = dict(arrowstyle="<|-", fc='0.5', alpha=0.5)  # alpha:透明度 fc:背景颜色,为0时全黑


def get_leaf_num(tree):
    leaf_num = 0
    first_key = list(tree.keys())[0]
    next_dict = tree[first_key]
    for key in next_dict.keys():
        if type(next_dict[key]).__name__ == "dict":
            leaf_num += get_leaf_num(next_dict[key])
        else:
            leaf_num += 1
    return leaf_num


# 获取树的深度（确定图的高度）
def get_tree_depth(tree):
    depth = 0
    first_key = list(tree.keys())[0]
    next_dict = tree[first_key]
    for key in next_dict.keys():
        if type(next_dict[key]).__name__ == "dict":
            thisdepth = 1 + get_tree_depth(next_dict[key])
        else:
            thisdepth = 1
        if thisdepth > depth: depth = thisdepth
    return depth


# 在当前节点与父节点连线，并填充当前节点的信息，va和ha是设置箭头尾部在节点的横纵位置
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    draw.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                      xytext=centerPt, textcoords='axes fraction',
                      va='bottom', ha="center", bbox=nodeType, arrowprops=arrow_args)
    # draw.ax1.annotate(nodeTxt, xy=parentPt, xytext=centerPt, bbox=nodeType, arrowprops=arrow_args)


# 在父子节点间填充文本信息
def plotMidText(curposition, lastposition, txtString):
    xMid = (lastposition[0] - curposition[0]) / 2.0 + curposition[0]
    yMid = (lastposition[1] - curposition[1]) / 2.0 + curposition[1]
    draw.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(tree, lastposition, nodetext):
    numLeafs = get_leaf_num(tree)
    firstStr = list(tree.keys())[0]
    curposition = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)

    # 画当前节点和内容
    plotMidText(curposition, lastposition, nodetext)
    plotNode(firstStr, curposition, lastposition, decision_node)
    secondDict = tree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    # 给子树分支
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 画子树
            plotTree(secondDict[key], curposition, str(key))
        else:  # 画叶子节点
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), curposition, leaf_node)
            plotMidText((plotTree.xOff, plotTree.yOff), curposition, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def draw(tree):
    fig = plt.figure(1)
    fig.clf()  # 清除plt的xy轴的显示
    axprops = dict(xticks=[], yticks=[])  # 配置坐标轴刻度不显示

    # 子图的三位数分别是xyz,都是1-9，z<=x*y，除非改为plt.subplot(x, y, z)
    # 子图的总数是x*y,画在第z个子图上
    draw.ax1 = plt.subplot(111, frameon=False, **axprops)  # frameon=False配置图中Axes rectangle patch不显示

    plotTree.totalW = float(get_leaf_num(tree))  # 树的宽度
    plotTree.totalD = float(get_tree_depth(tree))  # 树的深度
    # xy偏移量
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(tree, (0.5, 1.0), '')
    plt.show()


def printnodeinfo():
    print(decision_node)
    print(leaf_node)
    print(arrow_args)
