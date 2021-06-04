import pandas as pd
import numpy as np
import random
import time

# 定义大素数
C = pow(2, 32) - 5
# 欧氏距离
def euclideanDistance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

# 计算数据点的近邻
def neighbor(dataset):
    Distance = []  # 用以存放近邻索引
    for i in range(1000):
        Dis = []
        for j in range(dataset.shape[0]):
            if i == j:
                distance = -1
            else:
                # 利用欧式距离计算距离
                distance = np.sqrt(euclideanDistance(dataset[i], dataset[j]))
            Dis.append(distance)
        # 排序
        sort = np.argsort(np.array(Dis))
        Distance.append(sort)
    return Distance

# 创建哈希类
class Hash():
    def __init__(self, index):
        self.val = index
        self.buckets = {}  # 哈希桶

# 生成哈希函数
def FuncHash(n, r):
    # n为数据维数
    a = []
    for i in range(n):
        a.append(random.gauss(0, 1))
        b = random.uniform(0, r)  # 在【0，r】范围内随机产生一个实数
    return a, b

    # 生成哈希函数族
def gen_LSH_family(n, k, r):
    result = []
    # k为生成的函数个数
    for i in range(k):
        result.append(FuncHash(n, r))
    return result

# 将n维数据映射到一个整数
def HashVals(LSH_family, v, r):
    '''
    :param LSH_family: k个哈希函数族
    :param v: 数据向量
    :return: 映射后的结果g函数
    '''
    hashVals = []
    for i in LSH_family:
        hashVal = (np.inner(i[0], v) + i[1]) // r
        hashVals.append(hashVal)

    return hashVals

# 生出数据向量的一种特征
def feature(hashVals, Rand, k, C):
    '''
    :param hashVals:k维哈希值
    :param Rand:用于生成特征的随机k维向量
    :param k:hashVals维数，Rand维数
    :param C:一个大素数
    :return:数据[x1,x2,...,xk]的指纹 int类型
    '''
    return int(sum([(hashVals[i] * Rand[i]) for i in range(k)]) % C)

# 计算每个数据点的g函数，存入对应的哈希桶中
def LSH(dataset, k, L, r, tablesize):
    # 建立tablesize个哈希表
    hashTable = [Hash(i) for i in range(tablesize)]

    # 获取数据集维度，数据个数
    n = len(dataset[0])
    num = dataset.shape[0]

    # 存放哈希函数族
    hashFuncs = []
    # 用于生成桶索引和哈希表索引的随机整数
    Rand = [random.randint(-10, 10) for i in range(k)]

    # 计算L个函数族，每个函数族中有k个随机函数
    for l in range(L):
        # 第一个函数族
        LSH_family = gen_LSH_family(n, k, r)
        hashFuncs.append(LSH_family)

        for data_index in range(num):
            # 生成g函数
            hashVals = HashVals(LSH_family, dataset[data_index], r)

            # 生成特征
            ft = feature(hashVals, Rand, k, C)
            # 数据点所在的哈希表索引
            table_index = ft % tablesize
            # 找到索引为table_index的哈希表
            table = hashTable[table_index]

            # 判断指纹fp在该table对应的桶中
            # 是：将当前数据点编号放入该桶
            # 否：新建fp对应的索引
            if ft in table.buckets:
                table.buckets[ft].append(data_index)
            else:
                table.buckets[ft] = [data_index]
    # 返回哈希表，哈希函数族表，随机整数表
    return hashTable, hashFuncs, Rand
# 进行最邻近搜索
def nn_search(dataset, querylist, k, L, r, tablesize):
    # 存放1000个点所有的最近邻
    resultSet = []
    hashTable, hashFuncs, Rand = LSH(dataset, k, L, r, tablesize)

    # 计算每个查询点query的g函数值，找到query所在的哈希桶，将哈希桶中的数据索引存入resultSet
    for query in querylist:
        query_result = set()
        for hashFunc in hashFuncs:
            # 产生特征
            queryft = feature(HashVals(hashFunc, query, r), Rand, k, C)
            # 找到特征所在的哈希桶
            query_table_index = queryft % tablesize
            if queryft in hashTable[query_table_index].buckets:
                query_result.update(hashTable[query_table_index].buckets[queryft])
        # 添加到resultSet中
        resultSet.append(query_result)
    return resultSet

if __name__ == "__main__":
    # 读取数据集
    dataset = pd.read_csv('ColorHistogram.asc', skiprows=0, encoding="gbk", engine='python',
                          sep=' ', delimiter=None, index_col=False, header=None, skipinitialspace=True)
    dataset = np.array(dataset)[:, 1:]
    queryList = dataset[:1000]  # 选取1000个数据
    time_start1 = time.time()
    Truth = neighbor(dataset)
    time_end1 = time.time()
    print("算法1耗时: ", time_end1-time_start1, " s ")

    time_start2 = time.time()
    result_index_list = []  # 存放LSH查询后的1000点的最邻近索引
    resultSet = nn_search(dataset, queryList, k=20, L=5, r=1, tablesize=20)
    time_end2 = time.time()
    print("算法2耗时: ", time_end2-time_start2, " s")

    i = 0
    time_start3 = time.time()
    # 按欧氏距离给resultSet排序
    for result in resultSet:
        index_List = []
        distance = []

        for index in result:
            index_List.append(index)
            distance.append(euclideanDistance(dataset[i], dataset[index]))

        sort_index = np.argsort(np.array(distance))
        index_List = np.array(index_List)[sort_index]
        result_index_list.append(index_List)
        i += 1
    time_end3 = time.time()
    print("耗时： ", time_end3 - time_start3)

    ACC = []  # ACC存放准确率
    Recall = []  # Recall存放召回率
    file = open('info.txt', 'w')

    for j in range(queryList.shape[0]):
        # 实际前10个最近邻索引
        truth = np.array(Truth[j][1:11])
        # 经LSH算法预测前10最近邻索引
        predict = np.array(result_index_list[j][1:11])

        TP = len(set(truth) & set(predict))
        FN = len(truth) - TP

        file.write(str(j) + ' true ' + str(truth) + '\n')
        file.write(str(j) + ' predict ' + str(predict) + '\n')
        file.write('预测成功个数： '+str(TP)+'\n\n')

        recall = TP / (TP + FN)
        acc = TP*1.0/10
        Recall.append(recall)
        ACC.append(acc)

    file.write('耗时: ' + str(time_end3 - time_start1) + '\n')
    file.write("Recall:" + str(np.array(Recall).mean()) + '\n')
    file.write("Accuary:" + str(np.array(ACC).mean()) + '\n')
    print("Recall:", np.array(Recall).mean())
    print("Accuary:", np.array(ACC).mean())
