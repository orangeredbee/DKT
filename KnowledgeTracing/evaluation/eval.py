
import tqdm
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from KnowledgeTracing.Constant import Constants as C

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # @gpu


# 包括AUC、F1、Recall、Precision，可以根据需要自行添加，计算方式可自定义或者直接掉包：
def performance(ground_truth, prediction):
    ground_truth_numpy = ground_truth.detach().cpu().numpy()
    prediction_numpy = prediction.detach().cpu().numpy()
    round_prediction_numpy = torch.round(prediction).detach().cpu().numpy()

    fpr, tpr, thresholds = metrics.roc_curve(ground_truth_numpy, prediction_numpy)
    auc = metrics.auc(fpr, tpr)

    f1 = metrics.f1_score(ground_truth_numpy, round_prediction_numpy)
    recall = metrics.recall_score(ground_truth_numpy, round_prediction_numpy)
    precision = metrics.precision_score(ground_truth_numpy, round_prediction_numpy)

    print('auc:' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) + ' precision: ' + str(precision) + '\n')


# 交叉熵函数
class lossFunc(nn.Module):
    def __init__(self):
        super(lossFunc, self).__init__()

    # pred是N*MAX_STEP*ques_num ; batch是N*MAX_STEP*(2*ques_num)
    def forward(self, pred, batch):
        loss = torch.Tensor([0.0]).to(device)  # GPU
        for student in range(pred.shape[0]):
            delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]  # delta是压缩后的答题记录onehot表示
            temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t())  # DKT 原始损失是预测值与下一步实际作答的交叉熵
            index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]]).to(device)  # tmp是二维数组，对角线元素才是需要的
            p = temp.gather(0, index)[0]  # 取对角元素，为预测值
            a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1)//2)[1:]  # 处理后，答对为1，答错为0
            # 计算交叉熵
            for i in range(len(p)):
                if p[i] > 0:
                    loss = loss - (a[i]*torch.log(p[i]) + (1-a[i])*torch.log(1-p[i]))
        return loss

# 对于test_epoch，由于知识追踪任务比较特殊，每一个时刻的输出都是预测下一个时刻答对题目的概率，因此有一些额外的处理
def test_epoch(model, testLoader):
    gold_epoch = torch.Tensor([]).to(device)  # 真实结果ground truth
    pred_epoch = torch.Tensor([]).to(device)  # 预测的结果pred
    # 读取数据，分多个batch进行预测，因为一次预测可能数据量过大导致内存溢出而出错。Note：每一个batch中包含多个学生，每个学生有maxstep个题目，每个题目表示成了2*num_of_ques维的onehot向量。
    for batch in tqdm.tqdm(testLoader, desc='Testing:    ', mininterval=2):
        batch = batch.to(device)  # GPU
        pred = model(batch)  # len(pred)= question_num
        # 预测完之后，整理数据，把学生所有的题目的预测结果存储起来，方便后面的评估。
        # 对于每一个学生，先创建两个列表，分别存储真是答题结果ground truth和预测结果pred。
        # 然后再将每个学生的结果添加进开始定义的两个总结果列表gold_epoch和pred_epoch中去。
        for student in range(pred.shape[0]):
            temp_pred = torch.Tensor([]).to(device)  # torch.Tensor()默认张量类型torch.FloatTensor()的别名
            temp_gold = torch.Tensor([]).to(device)
            # 然后是获取预测结果，这里先将2*num_of_ques维的题目onehot向量分成前后两个部分，每部分分别是num_of_ques维，然后相加，乘以预测结果，即可得到对应的题目的预测结果，这里的计算过程可自行推敲，
            # 等有机会再给出可视化的计算过程。因为每一个时刻都是预测的下一个时刻的结果，所以题目编号需要向后移一个，体现在delta[1:]这里：
            delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]
            temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t())  # mm(): matrix multiplication
            index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]]).to(device)
            p = temp.gather(0, index)[0]  # 在tmp的0维上，按照index的值作为索引，取出tmp对应的值（对tmp重新排序或切片）
            # 对于答题的真实结果，其实在onehot的向量中就已经体现了，答对则向量前半部分对应的位置为1，答错则向量后半部分对应的位置为1。根据这个特点，按照下面的方式就可以直接通过onehot向量推出真实答题结果：
            a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1)//2)[1:]
            # 到此处为止，预测结果和真实结果就已经都得到了。但是，这里还要在做一个筛选，别忘了我们之前在数据长度不够的时候是补0了的，这里需要把补0的结果全部都过滤掉。
            # 由于补零的题目的onehot向量为全零向量，那么全零向量经过神经网络之后预测结果肯定为0。而正常题目不是非零的，那么预测结果为0的可能性极小，因为神经网络参数为0的可能性极小。
            # 所以我们根据预测结果是否为0，直接把为0的全部去除掉（我们这里的处理方法似乎不是很合理，因为正常题目也是有可能出现预测结果为0的情况，但是这种可能性极小，对模型整体而言几乎没什么影响，所以这么做也是合理的，并且十分方便）：
            for i in range(len(p)):
                if p[i] > 0:  # 此处忽略预测值为0的情况，因为概率很小。为了完整可以加上。
                    temp_pred = torch.cat([temp_pred, p[i:i+1]])
                    temp_gold = torch.cat([temp_gold, a[i:i+1]])  # [i:i+1]多次一举? 并不是，保持一维向量，否则是标量；
            # 在每次处理完一个学生的数据之后，将其添加到总结果列表中去：
            pred_epoch = torch.cat([pred_epoch, temp_pred])
            gold_epoch = torch.cat([gold_epoch, temp_gold])
    return pred_epoch, gold_epoch


def test(testLoaders, model):
    ground_truth = torch.Tensor([]).to(device)  # 真实答题
    prediction = torch.Tensor([]).to(device)  # 预测答题
    # 对于测试过程，由于某些测试集可能会很大，导致内存一次存不下，所以将测试集分成多个loader，然后对于每一个loader都调用一次test_epoch，
    for i in range(len(testLoaders)):
        pred_epoch, gold_epoch = test_epoch(model, testLoaders[i])
        prediction = torch.cat([prediction, pred_epoch])
        ground_truth = torch.cat([ground_truth, gold_epoch])
    performance(ground_truth, prediction)


# train_epoch，过程跟一般的pytorch模型训练过程一样，读取数据loader、预测、计算损失、反向传播等
def train_epoch(model, trainLoader, optimizer, loss_func):

    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = batch.to(device)  # GPU
        pred = model(batch)  # =model.forward(data).   batch 是一个batch的数据; 返回pred是N*MAX_STEP*ques_num
        optimizer.zero_grad()  # 梯度清零
        loss = loss_func(pred, batch)  # Loss
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降
    return model, optimizer


# eval：在eval.py文件中，定义了两个函数train和test分别实现模型的训练和测试：
def train(trainLoaders, model, optimizer, lossFunc):
    for i in range(len(trainLoaders)):
        model, optimizer = train_epoch(model, trainLoaders[i], optimizer, lossFunc)
    return model, optimizer







