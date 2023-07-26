from pylab import *
import pandas as pd
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score  # NMI
from sklearn.metrics import rand_score  # RI
from sklearn.metrics import accuracy_score  # ACC
from sklearn.metrics import f1_score  # F-measure

# 数据保存在.csv文件中
iris = pd.read_csv("dataset/iris.csv", header=0)  # 鸢尾花数据集 Iris  class=3
wine = pd.read_csv("dataset/wine.csv")  # 葡萄酒数据集 Wine  class=3
seeds = pd.read_csv("dataset/seeds.csv")  # 小麦种子数据集 seeds  class=3
wdbc = pd.read_csv("dataset/wdbc.csv")  # 威斯康星州乳腺癌数据集 Breast Cancer Wisconsin (Diagnostic)  class=2
glass = pd.read_csv("dataset/glass.csv")  # 玻璃辨识数据集 Glass Identification  class=6
df = iris  # 设置要读取的数据集
# print(df)
columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
features = columns[:len(columns) - 1]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
dataset = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
class_labels = list(df[columns[-1]])  # 原始标签
attributes = len(df.columns) - 1  # 属性数量（数据集维度）
k = 3  # 聚类簇数
MAX_ITER = 20  # 最大迭代数
n = len(dataset)  # 样本数
m = 2.00  # 模糊参数


# 初始化模糊矩阵U
def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x / summation for x in random_num_list]  # 首先归一化
        membership_mat.append(temp_list)
    return membership_mat


# 计算类中心点
def calculateClusterCenter(membership_mat):
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    cluster_mem_val_list = list(cluster_mem_val)
    for j in range(k):
        x = cluster_mem_val_list[j]
        x_raised = [e ** m for e in x]
        denominator = sum(x_raised)
        temp_num = list()
        for i in range(n):
            data_point = list(dataset.iloc[i])
            prod = [x_raised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]  # 每一维都要计算。
        cluster_centers.append(center)
    return cluster_centers


# 更新隶属度
def updateMembershipValue(membership_mat, cluster_centers):
    #    p = float(2/(m-1))
    data = []
    for i in range(n):
        x = list(dataset.iloc[i])  # 取出文件中的每一行数据
        data.append(x)
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j] / distances[c]), 2) for c in range(k)])
            membership_mat[i][j] = float(1 / den)
    return membership_mat, data


# 得到聚类结果
def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzyCMeansClustering():
    # 主程序
    membership_mat = initializeMembershipMatrix()
    curr = 0
    start = time.time()  # 开始时间，计时
    while curr <= MAX_ITER:  # 最大迭代次数
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat, data = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        curr += 1

    print("用时：{0}".format(time.time() - start))
    # print(membership_mat)
    return cluster_labels, cluster_centers, data, membership_mat


labels, centers, data, membership = fuzzyCMeansClustering()


def clustering_indicators(labels_true, labels_pred):
    if type(labels_true[0]) != int:
        labels_true = LabelEncoder().fit_transform(df[columns[len(columns) - 1]])  # 如果标签为文本类型，把文本标签转换为数字标签
    f_measure = f1_score(labels_true, labels_pred, average='macro')  # F值
    accuracy = accuracy_score(labels_true, labels_pred)  # ACC
    normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)  # NMI
    rand_index = rand_score(labels_true, labels_pred)  # RI
    return f_measure, accuracy, normalized_mutual_information, rand_index


F_measure, ACC, NMI, RI = clustering_indicators(class_labels, labels)
print("F_measure:", F_measure, "ACC:", ACC, "NMI", NMI, "RI", RI)
# print(centers)
center_array = array(centers)
label = array(labels)
datas = array(data)
if attributes > 2:
    dataset = PCA(n_components=2).fit_transform(dataset)  # 如果属性数量大于2，降维
# 做散点图
plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c='black', s=7)  # 原图
plt.show()
colors = np.array(["red", "blue", "green", "orange", "purple", "cyan", "magenta", "beige", "hotpink", "#88c999"])
# 循换打印k个簇，每个簇使用不同的颜色
for i in range(k):
    plt.scatter(dataset[nonzero(label == i), 0], dataset[nonzero(label == i), 1], c=colors[i], s=7)
# plt.scatter(center_array[:, 0], center_array[:, 1], marker='x', color='m', s=30)  # 聚类中心
plt.show()
