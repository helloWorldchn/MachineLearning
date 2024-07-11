import numpy as np
from numpy import matrix as mat
import matplotlib.pyplot as plt


def load_DataSet(file_data, col_X=(0, 1), col_Y=(2), add_bias=False):
    data_X = np.loadtxt(file_data, dtype=float, delimiter="\t", usecols=col_X)
    data_Y = np.loadtxt(file_data, dtype=float, delimiter="\t", usecols=col_Y)

    # 如果需要偏置，则为data_X 增加一个数值是1的维度
    if add_bias:
        N, D = np.shape(data_X)
        add_dim = np.ones((N, 1))
        data_X = np.concatenate((data_X, add_dim), axis=-1)

    # 对data_Y 进行维度修正 使其为 (N,1)
    if len(np.shape(data_Y)) == 1:
        data_Y = np.expand_dims(data_Y, axis=-1)

    return data_X, data_Y


def ridge_regress(X, Y, lam=0.2):
    # 数据转为mat格式
    N, D = np.shape(X)
    xMat = mat(X)
    yMat = mat(Y)
    xTx = xMat.T * xMat + mat(lam * np.eye(D))
    # 返回ws
    ws = xTx.I * (xMat.T * yMat)
    return np.array(ws)


def get_ws_by_lams(train_X, train_Y, lams):
    # 数据正则化
    mean_X = np.mean(train_X, axis=0, keepdims=True)
    std_X = np.std(train_X, axis=0, keepdims=True)
    X = (train_X - mean_X) / std_X

    mean_Y = np.mean(train_Y, axis=0, keepdims=True)
    Y = train_Y - mean_Y

    # 针对不同的lam求取不同的ws
    N, D = np.shape(train_X)
    N_ws = len(lams)
    save_ws = np.zeros((D, N_ws))
    for i, lam in enumerate(lams):
        ws = ridge_regress(X, Y, lam=lam)
        save_ws[:, i] = ws[:, 0]
    return save_ws


if __name__ == "__main__":
    # 真实数据测试：
    X, Y = load_DataSet('鲍鱼.txt', col_X=(0, 1, 2, 3, 4, 5, 6, 7), col_Y=(8))

    # 测试1
    train_X = X[:99, :]
    train_Y = Y[:99, :]

    N_lam = 30
    lams = [np.exp(i - 10) for i in range(N_lam)]

    save_ws = get_ws_by_lams(X, Y, lams)

    # 绘图
    fig = plt.figure()  # 创建绘图对象
    ax = fig.add_subplot(1, 1, 1)
    N, D = np.shape(train_X)
    for i in range(D):
        ax.plot([(i - 10) for i in range(N_lam)], save_ws[i, :], label=str(i))
    ax.set_xlabel('log(lam)')
    plt.legend(loc='best')
    plt.show()
