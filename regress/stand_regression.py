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


def linear_Regress(X, Y):
    #  # mat()函数将xArr，yArr转换为矩阵 mat().T 代表的是对矩阵进行转置操作
    xMat = mat(X)
    yMat = mat(Y)
    # 矩阵乘法的条件是左矩阵的列数等于右矩阵的行数
    xTx = xMat.T * xMat
    # 因为要用到xTx的逆矩阵，所以事先需要确定计算得到的xTx是否可逆，条件是矩阵的行列式不为0
    # linalg.det() 函数是用来求得矩阵的行列式的，如果矩阵的行列式为0，则这个矩阵是不可逆的，就无法进行接下来的运算
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 最小二乘法
    ws = xTx.I * xMat.T * yMat
    return np.array(ws)


if __name__ == "__main__":
    # 数据加载
    X_train, Y_train = load_DataSet('ex0.txt')

    # 参数获取
    ws = linear_Regress(X_train, Y_train)
    print(ws)

    X_test, Y_test = X_train, Y_train
    # X_test,Y_test = load_DataSet('ex1.txt')

    # 回归预测
    Y_hat = np.dot(X_test, ws)

    # 绘图
    fig = plt.figure()  # 创建绘图对象
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X_test[:, 1], Y_test)
    # 数据排序
    index = np.argsort(X_test[:, 1])

    X_copy = X_test[index, :]
    Y_hat = Y_hat[index, :]

    ax.plot(X_copy[:, 1], Y_hat, color=(1, 0, 0))

    plt.show()
