
from KnowledgeTracing.model.RNNModel import DKT
from KnowledgeTracing.data.dataloader import getTrainLoader, getTestLoader, getLoader
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.evaluation import eval
import torch.optim as optim
import torch
import argparse
import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # @gpu
print("device = ",  device)

# 代码来源：https://chsong.live/20201124_DKT-Pytorch/index.html

# 建立模型 RNN
model = DKT(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT)
model = model.to(device)  # GPU

# optimizer_adam = optim.Adam(model.parameters(), lr=C.LR)
optimizer_adgd = optim.Adagrad(model.parameters(), lr=C.LR)

loss_func = eval.lossFunc()
loss_func = loss_func.to(device)  # GPU

trainLoaders, testLoaders = getLoader(C.DATASET)  # N*maxstep, 学生X最大序列; type = list[Data.DataLoader]


for epoch in range(C.EPOCH):
    print('epoch: ' + str(epoch))
    model, optimizer = eval.train(trainLoaders, model, optimizer_adgd, loss_func)
    eval.test(testLoaders, model)

