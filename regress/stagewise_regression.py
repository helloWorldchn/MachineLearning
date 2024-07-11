import numpy as np
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


def rss_error(Y_hat, Y):
    N, _ = Y.shape
    error = np.sum((Y_hat - Y) * (Y_hat - Y)) / N
    return error


def lasso_error(Y_hat, Y, w, lam):
    N, _ = Y.shape
    error = np.sum((Y_hat - Y) * (Y_hat - Y)) / N + lam * np.sum(np.abs(w))
    return error


def ride_error(Y_hat, Y, w, lam):
    N, _ = Y.shape
    error = np.sum((Y_hat - Y) * (Y_hat - Y)) / N + lam * np.sum(np.abs(w) ** 2)
    return error


def stageWise(X, Y, eps=0.01, n_Iter=100):
    # 获取数据信息
    N, D = np.shape(X)

    # 记录每次迭代得到的w
    return_ws = np.zeros((n_Iter, D))

    ws = np.zeros((D, 1))
    ws_Test = ws.copy()
    ws_Max = ws.copy()
    # 最小误差
    lowest_Error = np.inf
    for i in range(n_Iter):
        # print(ws.T)
        lowest_Error = np.inf
        # 每一个维度进行搜索
        for j in range(D):
            # 2个搜索方向
            for sign in [-1, 1]:
                ws_Test = ws.copy()
                ws_Test[j] += eps * sign
                Y_hat = np.dot(X, ws_Test)
                # 计算误差
                error = lasso_error(Y_hat, Y, ws_Test, lam=0.1)
                # error = ride_error(Y_hat,Y,ws_Test,lam=0.1)
                # error = rss_error(Y_hat,Y)
                # 如果误差减小则进行w保存
                if error < lowest_Error:
                    lowest_Error = error
                    ws_Max = ws_Test
        ws = ws_Max.copy()
        print(ws.T, lowest_Error)
        return_ws[i] = ws.T
    return return_ws, ws


def plot_ws(ws_all):
    fig = plt.figure()  # 创建绘图对象
    ax = fig.add_subplot(1, 1, 1)
    n_Iter, D = np.shape(ws_all)

    for i in range(D):
        ax.plot([i for i in range(n_Iter)], ws_all[:, i], label=str(i))
    ax.set_xlabel('n_Iter')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    # 真实数据测试：
    X, Y = load_DataSet('鲍鱼.txt', col_X=(0, 1, 2, 3, 4, 5, 6, 7), col_Y=(8))

    # 对数据进行正则化
    mean_X = np.mean(X, axis=0, keepdims=True)
    std_X = np.std(X, axis=0, keepdims=True)
    X = (X - mean_X) / std_X

    mean_Y = np.mean(Y, axis=0, keepdims=True)
    Y = Y - mean_Y

    # 进行 stageWise 回归
    ws_all, ws_best = stageWise(X, Y, eps=0.001, n_Iter=5000)

    plot_ws(ws_all)
