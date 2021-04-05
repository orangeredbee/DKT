
import torch
import torch.utils.data as Data
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.data.readdata import DataReader
from KnowledgeTracing.data.DKTDataSet import DKTDataSet


def getTrainLoader(train_data_path):
    handle = DataReader(train_data_path, C.MAX_STEP, C.NUM_OF_QUESTIONS)
    trainques, trainans = handle.getTrainData()  # 1. 读取数据。trainques N*1 ; trainans:N*maxstep的矩阵形式，N即可看做学生个数。maxstep即为答题个数。
    dtrain = DKTDataSet(trainques, trainans)  # 2. onehot化数据class
    trainLoader = Data.DataLoader(dtrain, batch_size=C.BATCH_SIZE, shuffle=True)  # 3.装入DataLoader
    return trainLoader


def getTestLoader(test_data_path):
    handle = DataReader(test_data_path, C.MAX_STEP, C.NUM_OF_QUESTIONS)
    testques, testans = handle.getTestData()
    dtest = DKTDataSet(testques, testans)
    testLoader = Data.DataLoader(dtest, batch_size=C.BATCH_SIZE, shuffle=False)
    return testLoader


# 函数封装了getTrainLoader和getTestLoader，通过调用此函数直接获取训练和测试的loader。
def getLoader(dataset):
    trainLoaders = []
    testLoaders = []

    trainLoader = getTrainLoader(C.Dpath + C.Datasets[C.DATASET_NAME]['train_path'])
    trainLoaders.append(trainLoader)
    testLoader = getTestLoader(C.Dpath + C.Datasets[C.DATASET_NAME]['test_path'])
    testLoaders.append(testLoader)

    return trainLoaders, testLoaders