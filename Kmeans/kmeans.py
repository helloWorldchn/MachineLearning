import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


# 数据保存在.csv文件中
iris = pd.read_csv("dataset/Iris.csv", header=0)  # 鸢尾花数据集 Iris  class=3
wine = pd.read_csv("dataset/wine.csv")  # 葡萄酒数据集 Wine  class=3
seeds = pd.read_csv("dataset/seeds.csv")  # 小麦种子数据集 seeds  class=3
wdbc = pd.read_csv("dataset/wdbc.csv")  # 威斯康星州乳腺癌数据集 Breast Cancer Wisconsin (Diagnostic)  class=2
glass = pd.read_csv("dataset/glass.csv")  # 玻璃辨识数据集 Glass Identification  class=6
df = iris  # 设置要读取的数据集
# print(df)

columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
features = columns[:len(columns) - 1]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
dataset = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
attributes = len(df.columns) - 1  # 属性数量（数据集维度）
class_labels = list(df[columns[-1]])  # 原始标签

k = 3

# 这里已经知道了分3类，其他分类这里的参数需要调试
model = KMeans(n_clusters=k)
# 训练模型
model.fit(dataset)
# 预测全部数据
label = model.predict(dataset)
print(label)


def clustering_indicators(labels_true, labels_pred):
    if type(labels_true[0]) != int:
        labels_true = LabelEncoder().fit_transform(df[columns[len(columns) - 1]])  # 如果数据集的标签为文本类型，把文本标签转换为数字标签
    f_measure = f1_score(labels_true, labels_pred, average='macro')  # F值
    accuracy = accuracy_score(labels_true, labels_pred)  # ACC
    normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)  # NMI
    rand_index = rand_score(labels_true, labels_pred)  # RI
    return f_measure, accuracy, normalized_mutual_information, rand_index


F_measure, ACC, NMI, RI = clustering_indicators(class_labels, label)
print("F_measure:", F_measure, "ACC:", ACC, "NMI", NMI, "RI", RI)

if attributes > 2:
    dataset = PCA(n_components=2).fit_transform(dataset)  # 如果属性数量大于2，降维
# 打印出聚类散点图
plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c='black', s=7)  # 原图
plt.show()
colors = np.array(["red", "blue", "green", "orange", "purple", "cyan", "magenta", "beige", "hotpink", "#88c999"])
# 循换打印k个簇，每个簇使用不同的颜色
for i in range(k):
    plt.scatter(dataset[np.nonzero(label == i), 0], dataset[np.nonzero(label == i), 1], c=colors[i], s=7)
plt.show()
