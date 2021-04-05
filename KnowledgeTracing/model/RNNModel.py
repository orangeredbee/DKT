import torch
import torch.nn as nn
# import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

# 模型中定义了模型参数和结构，训练步骤在KnowledgeTracing.eval中给出
class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DKT, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层_size
        self.layer_dim = layer_dim  # num_隐藏层
        self.output_dim = output_dim
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        # batch_first – 如果True的话，那么输入Tensor的shape应该是[batch_size, time_step, feature], 输出也是这样。
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()

    # x为训练集 N*maxstep 的矩阵形式（学生数x答题长度）
    def forward(self, x):
        # h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))  # H_num X minibatch_size X H_size
        # output (seq_len, batch, hidden_size * num_directions): 为每个时间步得到的hidden_state
        # h_n (num_layers * num_directions, batch, hidden_size): 为最后一个时间步的hidden_state
        out, hn = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state,表示没有第一个h
        # 预测结果，经过sigmoid输出
        res = self.sig(self.fc(out))
        return res


