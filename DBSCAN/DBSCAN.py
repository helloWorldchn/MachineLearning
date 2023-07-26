import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# 数据保存在.csv文件中
iris = pd.read_csv("dataset/iris.csv", header=0)  # 鸢尾花数据集 Iris  class=3
wine = pd.read_csv("dataset/wine.csv")  # 葡萄酒数据集 Wine  class=3
seeds = pd.read_csv("dataset/seeds.csv")  # 小麦种子数据集 seeds  class=3
wdbc = pd.read_csv("dataset/wdbc.csv")  # 威斯康星州乳腺癌数据集 Breast Cancer Wisconsin (Diagnostic)  class=2
glass = pd.read_csv("dataset/glass.csv")  # 玻璃辨识数据集 Glass Identification  class=6
jain = pd.read_table("dataset/jain.txt")  # 人工数据集
spiral = pd.read_csv("dataset/spiral.csv")  # 人工数据集
flame = pd.read_csv("dataset/flame.csv")  # 人工数据集
aggregation = pd.read_table("dataset/aggregation.txt")  # 人工数据集
df = spiral  # 设置要读取的数据集
# print(df)
columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
features = columns[:len(columns) - 1]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
dataset = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
attributes = len(df.columns) - 1  # 属性数量（数据集维度）
class_labels = list(df[columns[-1]])  # 原始标签

UNCLASSIFIED = 0
NOISE = -1


# 计算数据点两两之间的距离
def getDistanceMatrix(datas):
    N, D = np.shape(datas)
    dists = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            vi = datas[i, :]
            vj = datas[j, :]
            dists[i, j] = np.sqrt(np.dot((vi - vj), (vi - vj)))
    return dists


#  寻找以点cluster_id 为中心，eps 为半径的圆内的所有点的id
def find_points_in_eps(point_id, eps, dists):
    index = (dists[point_id] <= eps)
    return np.where(index == True)[0].tolist()


# 聚类扩展
# dists ： 所有数据两两之间的距离  N x N
# labs :   所有数据的标签 labs N，
# cluster_id ： 一个簇的标号
# eps ： 密度评估半径
# seeds： 用来进行簇扩展的点
# min_points： 半径内最少的点数
def expand_cluster(dists, labs, cluster_id, seeds, eps, min_points):
    i = 0
    while i < len(seeds):
        # 获取一个临近点
        Pn = seeds[i]
        # 如果该点被标记为NOISE 则重新标记
        if labs[Pn] == NOISE:
            labs[Pn] = cluster_id
        # 如果该点没有被标记过
        elif labs[Pn] == UNCLASSIFIED:
            # 进行标记，并计算它的临近点 new_seeds
            labs[Pn] = cluster_id
            new_seeds = find_points_in_eps(Pn, eps, dists)

            # 如果 new_seeds 足够长则把它加入到seed 队列中
            if len(new_seeds) >= min_points:
                seeds = seeds + new_seeds

        i = i + 1


def dbscan(datas, eps, min_points):
    # 计算 所有点之间的距离
    dists = getDistanceMatrix(datas)

    # 将所有点的标签初始化为UNCLASSIFIED
    n_points = datas.shape[0]
    labs = [UNCLASSIFIED] * n_points

    cluster_id = 0
    # 遍历所有点
    for point_id in range(0, n_points):
        # 如果当前点已经处理过了
        if not (labs[point_id] == UNCLASSIFIED):
            continue

        # 没有处理过则计算临近点
        seeds = find_points_in_eps(point_id, eps, dists)

        # 如果临近点数量过少则标记为 NOISE
        if len(seeds) < min_points:
            labs[point_id] = NOISE
        else:
            # 否则就开启一轮簇的扩张
            cluster_id = cluster_id + 1
            # 标记当前点
            labs[point_id] = cluster_id
            expand_cluster(dists, labs, cluster_id, seeds, eps, min_points)
    return labs, cluster_id


# 绘图    
def draw_cluster(datas, labs, n_cluster):
    plt.cla()

    colors = np.array(["red", "blue", "green", "orange", "purple", "cyan", "magenta", "beige", "hotpink", "#88c999"])
    if attributes > 2:
        datas = PCA(n_components=2).fit_transform(datas)  # 如果属性数量大于2，降维
    # plt.scatter(datas[:, 0], datas[:, 1], s=7., color="black")
    # plt.show()
    for i, lab in enumerate(labs):
        if lab == NOISE:
            plt.scatter(datas[i, 0], datas[i, 1], s=7., color=(0, 0, 0))
        else:
            plt.scatter(datas[i, 0], datas[i, 1], s=7., color=colors[lab - 1])
    plt.show()


if __name__ == "__main__":

    datas = np.array(dataset).astype(np.float32)

    # 数据正则化
    datas = StandardScaler().fit_transform(datas)
    eps = 0.45
    min_points = 5
    label, cluster_id = dbscan(datas, eps=eps, min_points=min_points)
    print("label of my dbscan")
    print(label)

    db = DBSCAN(eps=eps, min_samples=min_points).fit(datas)
    skl_labels = db.labels_
    print("label of sk-DBSCAN")
    print(skl_labels)

    draw_cluster(datas, label, cluster_id)
