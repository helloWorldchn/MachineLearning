import re
import os
import math
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 求tf值

# strs=["我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]
# for str in strs:
#     seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
#     print(", ".join(seg_list))

def tf(text):
    with open(os.path.join("./data", text), 'r', encoding='utf-8') as f:
        text = f.read()
        text_words = jieba.cut(text)

    r = "[z0-9_.!+-=——,$.%^，。？、\n~@#￥%……&*《》<>「」{}【】()/\\\[\]'\"]"
    strings = re.sub(r, " ", text)  # 表示只匹配特定字符，并将每一个特定字符替换为一个空格
    strings1 = strings.split(" ")
    data = {}
    for string in strings1:
        if string in data.keys():
            data[string] += (1.0 / len(strings1))
        else:
            data[string] = (1.0 / len(strings1))
    return data


# 遍历文件夹,求文件总数以及tfidf
def tfidf(value):
    file = os.listdir("./data")
    files = 0
    time = 0
    tfidf = []
    for var in file:
        if value in tf \
                    (var).keys():
            time += 1.0
        files += 1.0
    for var in file:
        if value in tf(var).keys():
            tfidf.append(math.log10(files / time) * tf(var)[value])
        else:
            tfidf.append(0)
    return tfidf


# print(tfidf('have'))
# print(tfidf('haved'))
# print(tfidf('sd'))
# 单词总表
def words():
    words = {}
    file = os.listdir("./data")
    for var in file:
        for word in tf(var).keys():
            if word not in words.keys():
                words[word] = 0
    return words


print(words())


# 求所有文章的tfidf向量
def tfidflist():
    file = os.listdir("./data")
    i = 0
    v = []
    for var in file:
        w = words()
        for word in tf(var).keys():
            if word in words().keys():
                w[word] = tfidf(word)[i]
        # print(w)
        v.append(list(w.values()))
        i += 1
    return v


print(tfidflist())
# 句子相似度比较
t1 = np.array(tfidflist()[int(input('请输入想比较的文章编号')) - 1])
t2 = np.array(tfidflist()[int(input('请输入被比较的文章编号')) - 1])
print(t1, t2)


def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos


print(cos_sim(t1, t2))
