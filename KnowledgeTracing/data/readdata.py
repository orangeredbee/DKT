import numpy as np
from KnowledgeTracing.data.DKTDataSet import DKTDataSet
import itertools
import tqdm

# 读取数据一：按指定形状返回
class DataReader():

    def __init__(self, path, maxstep, numofques):
        self.path = path  # path: 数据文件存储路径
        self.maxstep = maxstep   # maxstep: 最大序列长度
        self.numofques = numofques  # numofques: 此数据集中所有题目的总个数（去重后）

    def getTrainData(self):
        trainqus = np.array([])  # 存储题目编号
        trainans = np.array([])  # 存储对应的答题结果
        with open(self.path, 'r') as train:
            # 每次读取三行; tqdm是进度条展示; itertools.zip_longest(*[train] * 3) 连续三行 [train]*3, 并打包成元组（三元组）
            for len, ques, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 3), desc='loading train data:    ', mininterval=2):
                # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                len = int(len.strip().strip(','))  # 移除字符串*头尾*指定的字符序列（这里操作两次）
                ques = np.array(ques.strip().strip(',').split(',')).astype(np.int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(np.int)
                # 然后是处理长度不一致的问题，将所有答题序列的长度都处理成maxstep的整数倍，长度不够的补0。
                mod = 0 if len%self.maxstep == 0 else (self.maxstep - len%self.maxstep)  # mod=需要padding的长度
                zero = np.zeros(mod) - 1  # 用 -1 padding
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                trainqus = np.append(trainqus, ques).astype(np.int)
                trainans = np.append(trainans, ans).astype(np.int)
        return trainqus.reshape([-1, self.maxstep]), trainans.reshape([-1, self.maxstep])  # 处理成N*maxstep的矩阵形式，N即可看做学生个数。maxstep即为答题个数。


    def getTestData(self):
        testqus = np.array([])
        testans = np.array([])
        with open(self.path, 'r') as test:
            for len, ques, ans in tqdm.tqdm(itertools.zip_longest(*[test] * 3), desc='loading test data:    ', mininterval=2):
                len = int(len.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                mod = 0 if len % self.maxstep == 0 else (self.maxstep - len % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                testqus = np.append(testqus, ques).astype(np.int)
                testans = np.append(testans, ans).astype(np.int)
        return testqus.reshape([-1, self.maxstep]), testans.reshape([-1, self.maxstep])