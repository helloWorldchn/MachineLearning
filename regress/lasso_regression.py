import numpy as np
import copy
import matplotlib.pyplot as plt
from stand_regression import linear_Regress
from ridge_regression import ridge_regress


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


# 坐标轴下降法
def CoordinateDescent(X, Y, epochs, lr, lam):
    N, D = X.shape
    XMat = np.mat(X)
    YMat = np.mat(Y)
    w = np.ones([D, 1])

    # 进行 epoches 轮迭代
    for k in range(epochs):
        # 保存上一轮的w
        pre_w = copy.copy(w)
        # 逐维度进行参数寻优
        for i in range(D):
            # 在每个维度上找到最优的w_i
            for j in range(epochs):
                Y_hat = XMat * w
                g_i = XMat[:, i].T * (Y_hat - YMat) / N + lam * np.sign(w[i])
                # 进行梯度下降
                w[i] = w[i] - g_i * lr
                if np.abs(g_i) < 1e-3:
                    break
        # 计算上一轮的w 和当前轮 w 的差值，如果每个维度的w都没有什么变化则退出
        diff_w = np.array(list(map(lambda x: abs(x) < 1e-3, pre_w - w)))
        if diff_w.all():
            break
    return w


def show_gress_with_W(X, Y, W=None, Colors=None):
    # 绘图
    fig = plt.figure()  # 创建绘图对象
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:, 1], Y)

    # 回归预测
    if not W is None:
        for w, color in zip(W, Colors):
            Y_hat = np.dot(X, w)
            index = np.argsort(X[:, 1])
            X_copy = X[index, :]
            Y_hat = Y_hat[index, :]

            ax.plot(X_copy[:, 1], Y_hat, color=color)
    plt.show()


def compute_error(X, Y, W):
    Y_hat = np.dot(X, W)
    return np.sum((Y - Y_hat) ** 2) / Y_hat.shape[0]


if __name__ == "__main__":
    # 数据加载
    X_train, Y_train = load_DataSet('ex0.txt')
    show_gress_with_W(X_train, Y_train)

    # 添加几个额外的数据

    x_add = np.zeros([5, 2])
    x_add[:, 1] = np.linspace(0.6, 1, 5)
    x_add[:, 0] = 1

    y_add = np.zeros([5, 1])
    y_add[:, 0] = [10.5, 10.8, 12.1, 13.3, 12.8]

    X = np.concatenate([X_train, x_add], axis=0)
    Y = np.concatenate([Y_train, y_add], axis=0)
    show_gress_with_W(X, Y)

    # 线性回归
    W_linear = linear_Regress(X, Y)
    W_linear_2 = linear_Regress(X_train, Y_train)
    show_gress_with_W(X, Y, W=[W_linear, W_linear_2], Colors=[(1, 0, 0), (0, 1, 0)])
    show_gress_with_W(X_train, Y_train, W=[W_linear, W_linear_2], Colors=[(1, 0, 0), (0, 1, 0)])

    # 岭回归
    W_ridge = ridge_regress(X, Y, lam=15)
    show_gress_with_W(X, Y, W=[W_linear, W_linear_2, W_ridge], Colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    show_gress_with_W(X_train, Y_train, W=[W_linear, W_linear_2, W_ridge], Colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])

    # lasso 回归
    W_lasso = CoordinateDescent(X, Y, lam=0.13, lr=0.001, epochs=250)
    show_gress_with_W(X, Y, W=[W_linear, W_linear_2, W_lasso], Colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    show_gress_with_W(X_train, Y_train, W=[W_linear, W_linear_2, W_lasso], Colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])

    # 测试鲍鱼数据
    # 真实数据测试：
    X, Y = load_DataSet('鲍鱼.txt', col_X=(0, 1, 2, 3, 4, 5, 6, 7), col_Y=(8))
    mean_X = np.mean(X, axis=0, keepdims=True)
    std_X = np.std(X, axis=0, keepdims=True)
    X = (X - mean_X) / std_X

    mean_Y = np.mean(Y, axis=0, keepdims=True)
    Y = Y - mean_Y
    W_ridge = ridge_regress(X, Y, lam=15)
    print(W_ridge)
    print(compute_error(X, Y, W_ridge))

    W_lasso = CoordinateDescent(X, Y, lam=0.1, lr=0.001, epochs=250)
    print(W_lasso)
    print(compute_error(X, Y, W_lasso))
