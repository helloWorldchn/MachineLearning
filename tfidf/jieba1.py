import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取文本文件
with open("D:\桌面\文本相似度匹配\老师", "r", encoding="utf-8") as f:
    doc1 = f.read()
with open("D:\桌面\文本相似度匹配\老师", "r", encoding="utf-8") as f:
    doc2 = f.read()

# 对文本进行分词
doc1_words = jieba.cut(doc1)
doc2_words = jieba.cut(doc2)

# 将分词结果转为字符串
doc1_seg = " ".join(doc1_words)
doc2_seg = " ".join(doc2_words)

# 计算TF-IDF向量
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform([doc1_seg, doc2_seg])
doc1_vec = tfidf[0].toarray()
doc2_vec = tfidf[1].toarray()

# 计算余弦相似度
similarity = cosine_similarity(doc1_vec, doc2_vec)[0][0]

# 打印分词结果和TF-IDF向量
print("doc1分词结果：", doc1_seg)
print("doc2分词结果：", doc2_seg)
print("doc1 TF-IDF向量：", doc1_vec)
print("doc2 TF-IDF向量：", doc2_vec)
print("两个文档之间的相似度：", similarity)