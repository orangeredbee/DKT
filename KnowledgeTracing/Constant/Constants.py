Dpath = '../../KTDataset'  # 数据集相对路径

# Dpath = 'KTDataset'  # 数据集相对路径
# 所有数据都做过处理；
# 第一行：答题数
# 第二行：题目编号
# 第三行：答题结果，0表示错，1表示对


# datasets = {
#     'assist2009': 'assist2009',
#     'assist2015': 'assist2015',
#     'assist2017': 'assist2017',
#     'static2011': 'static2011',
#     'kddcup2010': 'kddcup2010',
#     'synthetic': 'synthetic'
# }
# # question number of each dataset  习题数量、原文使用了synthetic、assist2009、khan Math 三个数据集。
# numbers = {
#     'assist2009': 124,
#     'assist2015': 100,
#     'assist2017': 102,
#     'static2011': 1224,
#     'kddcup2010': 661,
#     'synthetic': 50
# }
# DATASET = datasets['static2011']
# NUM_OF_QUESTIONS = numbers['static2011']


Datasets = {
    'assist2009': {'filename': 'assist2009',
                   'que_nums': 124,
                   'train_path': '/assist2009/builder_train.csv',
                   'test_path': '/assist2009/builder_test.csv'},
    'assist2015': {'filename': 'assist2015',
                   'que_nums': 100,
                   'train_path': '/assist2015/assist2015_train.txt',
                   'test_path': '/assist2015/assist2015_test.txt'},
    'assist2017': {'filename': 'assist2017',
                   'que_nums': 102,
                   'train_path': '/assist2017/assist2017_train.txt',
                   'test_path': '/assist2017/assist2017_test.txt'},
    'static2011': {'filename': 'static2011',
                   'que_nums': 1224,
                   'train_path': '/statics2011/static2011_train.txt',
                   'test_path': '/statics2011/static2011_test.txt'},
    'kddcup2010': {'filename': 'kddcup2010',
                   'que_nums': 50,
                   'train_path': '/kddcup2010/kddcup2010_tarin.txt',
                   'test_path': '/kddcup2010/kddcup2010_test.txt'},
    'synthetic': {'filename': 'synthetic',
                  'que_nums': 50,
                  'train_path': '/synthetic/synthetic_train_v0.txt',
                  'test_path': '/synthetic/synthetic_test_v0.txt'}
}

DATASET_NAME = 'static2011'
DATASET = Datasets[DATASET_NAME]['filename']
NUM_OF_QUESTIONS = Datasets[DATASET_NAME]['que_nums']

# the max step of RNN model
MAX_STEP = 50
BATCH_SIZE = 64    # 原文：mini_batchsize 100
LR = 0.002
EPOCH = 1000
# input dimension 输入维度 = 习题数*2
INPUT = NUM_OF_QUESTIONS * 2
# embedding dimension
EMBED = NUM_OF_QUESTIONS  # 未见被使用
# hidden layer dimension 原文：200
HIDDEN = 200
# nums of hidden layers
LAYERS = 1
# output dimension
OUTPUT = NUM_OF_QUESTIONS
