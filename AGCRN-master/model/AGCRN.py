import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from model.AGCRNCell2 import AGCRNCell,AGCRNCell2

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()



class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x) # 残差
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """

        return self.network(x)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AVWDCRNN2(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, Adj,num_layers=1):
        super(AVWDCRNN2, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.adj=Adj
        # self.dcrnn_cells = nn.ModuleList()
        # self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        # self.tcn=nn.Conv1d(dim_in,dim_out,)
        # self.tcn=TemporalConvNet(dim_in,[1,1,1],3,0.2)
        self.gate_cnn1 = nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.gate_cnn2 = nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.alpha = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)  # D
        self.beta = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)  # S

        self.dcrnn_cells2 = nn.ModuleList()
        self.dcrnn_cells2.append(AGCRNCell2(node_num, dim_in, dim_out, self.adj))
        for _ in range(1, num_layers):
            # self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

            self.dcrnn_cells2.append(AGCRNCell2(node_num, dim_out, dim_out, self.adj))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        device=torch.device('cuda')
        # device=torch.device('cpu')
        # print("x:::",x.shape)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        b, t, n, d=x.shape
        # x=x.to(device=device)
        gate_input=x.permute(0,2,3,1).reshape(b*n,d,t) # b*n d t
        gate_cnn_out=torch.tanh(self.gate_cnn1(gate_input))*torch.sigmoid(self.gate_cnn2(gate_input))
        gate_cnn_out=gate_cnn_out.permute(0,2,1).reshape(b,t,n,d)
        print("gate:",gate_cnn_out.shape)
        # current_inputs = self.tcn(x).reshape(b,n,d,t).permute(0,3,1,2) # [b*n d t] --> [b n d t] -->[b t n d]
        current_inputs=x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                # state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                state = self.dcrnn_cells2[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        print("current:",current_inputs.shape)
        # current_inputs=self.alpha*current_inputs+self.beta*gate_cnn_out
        return current_inputs, output_hidden,gate_cnn_out

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells2[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, args, Adj):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        args.embed_dim=2
        self.adj=Adj
        # self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        # self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,args.embed_dim, args.num_layers)
        self.encoder = AVWDCRNN2(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, self.adj, args.num_layers)

        #predictor
        # self.conv = nn.Conv2d(12, 12, kernel_size=(1, self.hidden_dim), bias=True)
        # self.linear=nn.Linear(self.hidden_dim,self.hidden_dim,bias=True)
        self.end_conv = nn.Conv2d(6, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        # print("source:",source.shape)
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ ,gate_cnn_out= self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        # output = self.linear(output)
        # output=self.conv((output))

        output = output[:, -6:, :, :]+gate_cnn_out[:, -6:, :, :]                                  #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                      #B, T, N, C

        return output


if __name__=='__main__':
    import argparse
    import configparser
    config = configparser.ConfigParser()
    config_file = './PEMSD8_AGCRN.conf'

    config.read(config_file)

    args = argparse.ArgumentParser(description='arguments')

    args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    args.add_argument('--horizon', default=config['data']['horizon'], type=int)
    args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
    # args.add_argument('--embed_dim', default=2, type=int)
    args = args.parse_args()

    num_node = args.num_nodes
    input_dim = args.input_dim
    hidden_dim = args.rnn_units
    output_dim = args.output_dim
    horizon = args.horizon
    num_layers = args.num_layers
    adj = torch.ones((num_node,num_node))
    # print(adj.shape)
    node_embeddings = nn.Parameter(torch.randn(num_node, 2), requires_grad=True)
    agcrn=AGCRN(args,adj)
    # source: B, T_1, N, D
    # target: B, T_2, N, D
    x=torch.randn(32,12,170,1)
    tar=torch.randn(32,12,170,1)
    out=agcrn(x,tar)
    print(out.shape)
