import pandas as pd
import numpy as np
import csv

from sklearn.decomposition import PCA

# # 读取数据集
# def read_dataset():
#     try:
#         dataset = pd.read_csv('ColorHistogram.asc',
#                               skiprows=0,
#                               encoding="gbk",
#                               engine='python',
#                               sep=' ',
#                               delimiter=None,
#                               index_col=False, header=None, skipinitialspace=True)
#     except FileNotFoundError:
#         print('数据集读取失败!')
#     return dataset
def data():
    s = [[35, 21, 13, 20], [1, 19, 33, 25], [13, 38, 32, 26], [2, 39, 8, 10], [4, 9, 5, 8],
         [41, 24, 24, 35], [34, 22, 47, 25], [15, 32, 17, 36], [47, 35, 29, 12], [1, 37, 11, 32]]
    return np.array(s)


# 定义转换函数
# 数据集和目标维数
def PCA1(data, tar_dim):
    x = data
    # 进行中心化
    meanX = np.mean(x, axis=0)
    centerX = x - meanX
    # 求协方差矩阵
    covX = np.cov(centerX, rowvar=0)
    covX = np.dot(centerX.T, centerX)
    # 求特征值和特征向量
    eig_vals, eig_vecs = np.linalg.eig(covX)
    # 特征值从大到小排序, 取前tar_dim行
    indexs = np.argsort(-eig_vals)[:tar_dim]
    # 计算正交矩阵P
    p_vals = eig_vals[indexs]
    p_vecs = eig_vecs[:, indexs]
    # 结果矩阵
    data_sub = np.dot(centerX, p_vecs)
    return data_sub

# 保存结果
def save(vec, name):
    with open(name, 'w', encoding='utf-8', newline='')as f:
        csv_writer = csv.writer(f)
        for i in range(vec.shape[0]):
            csv_writer.writerow(vec[i, :])

#dataset = read_dataset()
# 去掉编号
#dataset = np.array(dataset)[:, 1:]
#data_sub = PCA1(dataset, 5)
s = data()
# print(dataset.var(axis=1))
# print(data_sub.var(axis=1))
s_sub = PCA1(s, 2)
# print(s.var(axis=1))
# print(s_sub.var(axis=1))


#save(PCA(dataset, 5), 'data_sub.csv')
# 保存方差
# with open('datasetVar.csv', 'w', encoding='utf-8', newline='')as f:
#     csv_writer = csv.writer(f)
#     csv_writer.writerow(dataset.var(axis=1))
#     csv_writer.writerow(data_sub.var(axis=1))

pca = PCA(n_components=2)
newx = pca.fit_transform(s)
print(newx)
print('sdada')
print(s_sub)

