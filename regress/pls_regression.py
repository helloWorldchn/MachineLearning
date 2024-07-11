import numpy as np


class PLS1():
    # Y 回归目标 维度 Nx1
    # X 观测/训练数据 维度 NxD
    # g 将观测数据 X 变为潜变量T，维度从D降维到g
    def __init__(self, Y, X, g, bias=False):

        # 最终的潜变量的成分数
        self.components = g

        # 对数据进行预处理,计算均值
        # 不在数据上加偏置，则需要对数据进行减均值的处理
        self.bias = bias
        if not bias:
            self.mean_X = np.mean(X, axis=0, keepdims=True)
            self.mean_Y = np.mean(Y, axis=0, keepdims=True)

            self.X = X - self.mean_X
            self.Y = Y - self.mean_Y
        else:
            self.X = X
            self.Y = Y

        # 获取特征维度
        self.N, self.D = np.shape(X)
        self.g = g

        # 对X进行降维时，g个基的系数 
        W = np.empty([self.D, self.g])

        # 利用潜变量对X进行回归的系数
        P = np.empty([self.D, self.g])

        # 存储变换后的潜变量
        T = np.empty([self.N, self.g])

        # 潜变量T 对 Y的 回归系数
        c = np.empty([1, self.g])

        # 最终的回归系数
        b = np.empty((self.D, 1))

        X_j = self.X
        Y_j = self.Y
        for j in range(g):
            # 计算X，每个维度上的特征与Y的相关性
            # 并利用这个相关性作为初始权重
            w_j = X_j.T @ Y_j
            w_j /= np.linalg.norm(w_j, 2)

            # 对 X 进行加权求和得到 t
            t_j = X_j @ w_j
            tt_j = t_j.T @ t_j

            # 利用 t 对 Y 进行回归得到系数 c    
            c_j = (t_j.T @ Y_j) / tt_j

            if c_j < 1e-6:
                self.components = j
                break

            # 利用t对X 进行回归得到回归系数 P
            p_j = (X_j.T @ t_j) / tt_j

            # 利用 t,P 计算 X的残差
            X_j = X_j - np.outer(t_j, p_j.T)

            # 利用 t,c 计算 Y的残差
            Y_j = Y_j - t_j * c_j

            # 中间结果存储
            W[:, j] = w_j[:, 0]
            P[:, j] = p_j[:, 0]
            T[:, j] = t_j[:, 0]
            c[:, j] = c_j
            # 返回利用X,Y残差，进行下一轮的迭代

        self.W = W[:, 0:self.components]
        self.P = P[:, 0:self.components]
        self.T = T[:, 0:self.components]
        self.c = c[0:self.components]

        # 迭代结束后计算b, Y= Xb^T
        # b = W*（P^T*W）^-1*C^T
        b = self.W @ np.linalg.inv(self.P.T @ self.W) @ self.c.T
        self.b = b

    def prediction(self, Z):
        if not self.bias:
            return self.mean_Y + (Z - self.mean_X) @ self.b

        else:
            return Z @ self.b

    def prediction_iterative(self, Z):
        N, _ = np.shape(Z)
        if not self.bias:
            result = self.mean_Y.copy()
            X_j = Z - self.mean_X
        else:
            X_j = Z
            result = np.zeros([N, 1])

        t = np.empty((N, self.components))
        for j in range(self.components):
            w_j = np.expand_dims(self.W[:, j], axis=-1)
            p_j = np.expand_dims(self.P[:, j], axis=-1)

            t_j = X_j @ w_j
            X_j = X_j - np.outer(t_j, p_j.T)
            t[:, j] = t_j[:, 0]
        result = result + t @ self.c.T

        return result


if __name__ == "__main__":
    X = np.array([[126, 38], [128, 40], [128, 42], [130, 42], [130, 44], [132, 46]]).astype(np.float32)
    # X = np.array([[1,126,38],[1,128,40],[1,128,42],[1,130,42],[1,130,44],[1,132,46]]).astype(np.float32)
    Y = np.array([[120], [125], [130], [121], [135], [140]]).astype(np.float32)
    m_PLS = PLS1(Y, X, g=2, bias=False)

    result = m_PLS.prediction(X)
    print(m_PLS.b)
    print(result)

    result2 = m_PLS.prediction_iterative(X)
    print(result2)
