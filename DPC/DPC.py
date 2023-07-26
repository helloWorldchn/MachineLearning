import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
df = flame  # 设置要读取的数据集
# print(df)
columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
features = columns[:len(columns) - 1]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
dataset = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
attributes = len(df.columns) - 1  # 属性数量（数据集维度）
class_labels = list(df[columns[-1]])  # 原始标签


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


# 找到密度计算的阈值dc
# 要求平均每个点周围距离小于dc的点的数目占总点数的1%-2%
def select_dc(dists):
    N = np.shape(dists)[0]
    tt = np.reshape(dists, N * N)
    percent = 2.0
    position = int(N * (N - 1) * percent / 100)
    dc = np.sort(tt)[position + N]
    return dc


# 计算每个点的局部密度
def get_density(dists, dc, method=None):
    N = np.shape(dists)[0]
    rho = np.zeros(N)

    for i in range(N):
        if method == None:
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho


# 计算每个数据点的密度距离
# 即对每个点，找到密度比它大的所有点
# 再在这些点中找到距离其最近的点的距离
def get_deltas(dists, rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)
    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            continue

        # 对于其他的点
        # 找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离,并找最小值
        deltas[index] = np.min(dists[index, index_higher_rho])

        # 保存最近邻点的编号
        index_nn = np.argmin(dists[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber


# 通过阈值选取 rho与delta都大的点
# 作为聚类中心
def find_centers_auto(rho, deltas):
    rho_threshold = (np.min(rho) + np.max(rho)) / 2
    delta_threshold = (np.min(deltas) + np.max(deltas)) / 2
    N = np.shape(rho)[0]

    centers = []
    for i in range(N):
        if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
            centers.append(i)
    return np.array(centers)


# 选取 rho与delta乘积较大的点作为
# 聚类中心
def find_centers_K(rho, deltas, K):
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return centers[:K]


def cluster_PD(rho, centers, nearest_neiber):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return

    N = np.shape(rho)[0]
    label = -1 * np.ones(N).astype(int)

    # 首先对几个聚类中进行标号
    for i, center in enumerate(centers):
        label[center] = i

    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 从密度大的点进行标号
        if label[index] == -1:
            # 如果没有被标记过
            # 那么聚类标号与距离其最近且密度比其大
            # 的点的标号相同
            label[index] = label[int(nearest_neiber[index])]
    return label


def draw_decision(rho, deltas):
    plt.cla()
    for i in range(np.shape(datas)[0]):
        plt.scatter(rho[i], deltas[i], s=16., color=(0, 0, 0))
        plt.annotate(str(i), xy=(rho[i], deltas[i]), xytext=(rho[i], deltas[i]))
        plt.xlabel("rho")
        plt.ylabel("deltas")
    # plt.savefig(filename+"_decision.jpg")
    plt.show()


def draw_cluster(datas, label, centers):
    plt.cla()
    K = np.shape(centers)[0]
    colors = np.array(["red", "blue", "green", "orange", "purple", "cyan", "magenta", "beige", "hotpink", "#88c999"])
    if attributes > 2:
        datas = PCA(n_components=2).fit_transform(datas)  # 如果属性数量大于2，降维
    for i in range(K):
        plt.scatter(datas[np.nonzero(label == i), 0], datas[np.nonzero(label == i), 1], c=colors[i], s=7)
        # plt.scatter(datas[centers[i], 0], datas[centers[i], 1], color="k", marker="+", s=200.)  # 聚类中心
    # plt.savefig(file_name + "_cluster.jpg")
    plt.show()


if __name__ == "__main__":
    # 主程序
    datas = np.array(dataset).astype(np.float32)
    # 计算距离矩阵
    dists = getDistanceMatrix(datas)
    # 计算dc
    dc = select_dc(dists)
    print("dc", dc)
    # 计算局部密度
    rho = get_density(dists, dc, method="Gaussion")
    # 计算密度距离
    deltas, nearest_neiber = get_deltas(dists, rho)

    # 绘制密度/距离分布图
    draw_decision(rho, deltas)

    # 获取聚类中心点
    centers = find_centers_K(rho, deltas, 3)
    # centers = find_centers_auto(rho,deltas)
    print("centers", centers)

    label = cluster_PD(rho, centers, nearest_neiber)
    plt.cla()
    K = np.shape(centers)[0]
    draw_cluster(datas, label, centers)
