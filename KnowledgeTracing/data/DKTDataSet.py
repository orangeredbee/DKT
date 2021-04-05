import numpy as np
from torch.utils.data.dataset import Dataset
from KnowledgeTracing.Constant import Constants as C
import torch

# 在readdata读取好数据之后，我们在DKTDataSet中对其进行封装（onehot）处理
class DKTDataSet(Dataset):
    def __init__(self, ques, ans):
        self.ques = ques
        self.ans = ans

    def __len__(self):
        return len(self.ques)

    # 对象通过索引[]访问元素时,会传入index，由函数得到返回结果
    def __getitem__(self, index):
        questions = self.ques[index]
        answers = self.ans[index]
        onehot = self.onehot(questions, answers)  # 直接返回（题目和回答）的one-hot形式而不再是题目编号
        return torch.FloatTensor(onehot.tolist())

    def onehot(self, questions, answers):
        result = np.zeros(shape=[C.MAX_STEP, 2 * C.NUM_OF_QUESTIONS])   # 记录个数  X one-hot（维度为两倍的总题目数）
        for i in range(C.MAX_STEP):
            if answers[i] > 0:  # 答对 1
                result[i][questions[i]] = 1
            elif answers[i] == 0:  # 答错 0
                result[i][questions[i] + C.NUM_OF_QUESTIONS] = 1
        return result