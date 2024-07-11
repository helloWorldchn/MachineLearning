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


def local_weight_LR(test_point, train_X, train_Y, k=1.0):
    xMat = mat(train_X)
    yMat = mat(train_Y)
    N, D = np.shape(xMat)
    # 构建weights 矩阵
    diff_mat = np.tile(test_point, [N, 1]) - train_X
    weights = np.exp(np.sum(diff_mat ** 2, axis=1) / (-2 * k ** 2))
    weights = mat(np.diag(weights))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("数据错误，无法求逆矩")
        return
    ws = xTx.I * xMat.T * weights * yMat
    return test_point * ws


def test_local_weight_LR(test_points, train_X, train_Y, k=1.0):
    N, D = test_points.shape
    Y_hat = np.zeros((N, 1))
    for i in range(N):
        Y_hat[i] = local_weight_LR(test_points[i], train_X, train_Y, k=k)
    return Y_hat


def compute_error_MSE(Y_hat, Y_real):
    return np.sum((Y_real - Y_hat) ** 2) / Y_hat.shape[0]


def stand_Regress(X, Y):
    # mat()函数将xArr，yArr转换为矩阵 mat().T 代表的是对矩阵进行转置操作
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
    ws = xTx.I * (xMat.T * yMat)
    return np.array(ws)


if __name__ == "__main__":
    X, Y = load_DataSet('ex0.txt')
    N, D = X.shape

    Y_hat_1 = test_local_weight_LR(X, X, Y, k=1.0)
    Y_hat_2 = test_local_weight_LR(X, X, Y, k=0.01)
    Y_hat_3 = test_local_weight_LR(X, X, Y, k=0.003)

    # 绘图
    index = np.argsort(X[:, 1])
    X_copy = X[index, :]

    fig = plt.figure()  # 创建绘图对象
    fig.subplots_adjust(hspace=0.5)
    # 子图1
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.scatter(X[:, 1], Y)
    Y_hat = Y_hat_1[index]
    ax1.plot(X_copy[:, 1], Y_hat, color=(1, 0, 0))
    ax1.set_title("k=1")

    # 子图2 
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.scatter(X[:, 1], Y)
    Y_hat = Y_hat_2[index]
    ax2.plot(X_copy[:, 1], Y_hat, color=(1, 0, 0))
    ax2.set_title("k=0.01")

    # 子图 3
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.scatter(X[:, 1], Y)
    Y_hat = Y_hat_3[index]
    ax3.plot(X_copy[:, 1], Y_hat, color=(1, 0, 0))
    ax3.set_title("k=0.003")

    plt.show()

    # 真实数据测试：
    X, Y = load_DataSet('鲍鱼.txt', col_X=(0, 1, 2, 3, 4, 5, 6, 7), col_Y=(8))
    # 测试1
    train_X = X[:99, :]
    train_Y = Y[:99, :]
    test_X = X[100:199, :]
    real_Y = Y[100:199, :]
    # test_X = X[0:99,:]
    # real_Y = Y[0:99]

    Y_hat1 = test_local_weight_LR(test_X, train_X, train_Y, k=0.1)
    error1 = compute_error_MSE(Y_hat1, real_Y)
    print(error1)

    Y_hat2 = test_local_weight_LR(test_X, train_X, train_Y, k=1)
    error2 = compute_error_MSE(Y_hat2, real_Y)
    print(error2)

    Y_hat3 = test_local_weight_LR(test_X, train_X, train_Y, k=10)
    error3 = compute_error_MSE(Y_hat3, real_Y)
    print(error3)

    ws = stand_Regress(train_X, train_Y)
    y_hat_LR = np.dot(test_X, ws)
    error_LR = compute_error_MSE(y_hat_LR, real_Y)
    print(error_LR)
