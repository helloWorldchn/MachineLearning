import numpy as np


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
    return dataSet, labels  # 返回数据集和分类属性


# 获取概率模型, 输入feat np.array格式 大小[N,D]
def trainPbmodel_X(feats):
    N, D = np.shape(feats)

    model = {}
    # 对每一维度的特征进行概率统计
    for d in range(D):
        data = feats[:, d].tolist()
        keys = set(data)
        N = len(data)
        model[d] = {}
        for key in keys:
            model[d][key] = float(data.count(key) / N)
    return model


# datas： list格式 每个元素表示1个特征序列
# labs：  list格式 每个元素表示一个标签
def trainPbmodel(datas, labs):
    # 定义模型
    model = {}
    # 获取分类的类别
    keys = set(labs)
    for key in keys:
        # 获得P(Y)
        Pbmodel_Y = labs.count(key) / len(labs)

        # 收集标签为Y的数据
        index = np.where(np.array(labs) == key)[0].tolist()
        feats = np.array(datas)[index]

        # 获得 P(X|Y)
        Pbmodel_X = trainPbmodel_X(feats)

        # 模型保存
        model[key] = {}
        model[key]["PY"] = Pbmodel_Y
        model[key]["PX"] = Pbmodel_X
    return model


# feat : list格式 一条输入特征
# model: 训练的概率模型
# keys ：考察标签的种类 
def getPbfromModel(feat, model, keys):
    results = {}
    eps = 0.00001
    for key in keys:
        # 获取P(Y)
        PY = model.get(key, eps).get("PY")

        # 分别获取 P(X|Y)
        model_X = model.get(key, eps).get("PX")
        list_px = []
        for d in range(len(feat)):
            pb = model_X.get(d, eps).get(feat[d], eps)
            list_px.append(pb)

        result = np.log(PY) + np.sum(np.log(list_px))
        results[key] = result
    return results


if __name__ == '__main__':

    '''实验一  自制贷款数据集'''

    # # 获取数据集
    # dataSet, labels = createDataSet()

    # # 截取数据和标签
    # datas = [i[:-1] for i in dataSet]
    # labs = [i[-1] for i in dataSet] 

    # # 获取标签种类
    # keys = set(labs)

    # # 进行模型训练
    # model = trainPbmodel(datas,labs)
    # print(model)

    # # 根据输入数据获得预测结果
    # feat = [0,1,0,1]
    # result = getPbfromModel(feat,model,keys)
    # print(result)

    # # 遍历结果找到概率最大值进行数据
    # for key,value in result.items():
    # if(value == max(result.values())):
    # print("预测结果是",key)

    # '''实验二  隐形眼睛数据集'''

    # 读取数据文件 截取数据和标签
    with open("train-lenses.txt", 'r', encoding="utf-8") as f:
        lines = f.read().splitlines()
    dataSet = [line.split('\t') for line in lines]

    datas = [i[:-1] for i in dataSet]
    labs = [i[-1] for i in dataSet]

    # 获取标签种类
    keys = set(labs)
    # 进行模型训练
    model = trainPbmodel(datas, labs)
    print(model)

    # 测试
    # 读取测试文件
    with open("test-lenses.txt", 'r', encoding="utf-8") as f:
        lines = f.read().splitlines()

    # 逐行读取数据并测试   
    for line in lines:
        data = line.split('\t')
        lab_true = data[-1]
        feat = data[:-1]
        result = getPbfromModel(feat, model, keys)

        key_out = ""
        for key, value in result.items():
            if (value == max(result.values())):
                key_out = key
        print("输入特征：")
        print(data)
        print(result)
        print("预测结果 %s  医生推荐 %s" % (key_out, lab_true))
