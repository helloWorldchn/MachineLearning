import numpy as np
import pandas as pd
import operator

'''
    trainData - 训练集  N
    testData - 测试   1
    labels - 训练集标签
'''

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
# 分为训练集和测试集和
N = int(df.shape[0])
N_train = int(N * 0.7)
N_test = N-N_train


def knn(trainData, testData, labels, k):
    # 计算训练样本的行数
    rowSize = trainData.shape[0]
    # 计算训练样本和测试样本的差值
    diff = np.tile(testData, (rowSize, 1)) - trainData
    # 计算差值的平方和
    sqrDiff = diff ** 2
    sqrDiffSum = sqrDiff.sum(axis=1)
    # 计算距离
    distances = sqrDiffSum ** 0.5
    # 对所得的距离从低到高进行排序
    sortDistance = distances.argsort()

    count = {}

    for i in range(k):
        vote = labels[sortDistance[i]]
        # print(vote)
        count[vote] = count.get(vote, 0) + 1
    # 对类别出现的频数从高到低进行排序
    sortCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)

    # 返回出现频数最高的类别
    return sortCount[0][0]


file_data = 'iris.data'

# 数据读取
data = np.loadtxt(file_data, dtype=float, delimiter=',', usecols=(0, 1, 2, 3))
lab = np.loadtxt(file_data, dtype=str, delimiter=',', usecols=(4))

# 分为训练集和测试集和
# N = 150
# N_train = 100
# N_test = 50

perm = np.random.permutation(N)

index_train = perm[:N_train]
index_test = perm[N_train:]

data_train = data[index_train, :]
lab_train = lab[index_train]

data_test = data[index_test, :]
lab_test = lab[index_test]

# 参数设定
k = 5
n_right = 0
for i in range(N_test):
    test = data_test[i, :]

    det = knn(data_train, test, lab_train, k)

    if det == lab_test[i]:
        n_right = n_right + 1

    print('Sample %d  lab_ture = %s  lab_det = %s' % (i, lab_test[i], det))

# 结果分析
print('Accuracy = %.2f %%' % (n_right * 100 / N_test))
