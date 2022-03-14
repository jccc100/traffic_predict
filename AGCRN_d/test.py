# _*_ coding: utf-8 _*_
# @time     :2022/3/1 15:22
# @Author   :jc
# @File     :test.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

def clones(module, N):
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None,score_his=None):
    '''

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)
    # if score_his != None:
    #     scores=scores+score_his
    #     score_his=scores
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(32, d_model), 4)
        self.lin=nn.Linear(32,64)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, score_his=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask: (batch, T, T)
        :return: x: (batch, N, T, d_model)
        '''
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # 这里是将前面定义的多个linear与query key value分别取出然后经过线性层变换然后转换维度得到query key value
        # query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) for l, x in
        #                      zip(self.linears, (query, key, value))]
        # query=self.lin(query)
        # key=self.lin(key)
        # value=self.lin(key)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        # 下面是res-attention
        # b,n,t1,_=query.shape
        # h_s=self.h
        # t2=value.shape[2]
        # score_his=torch.zeros((b,n,h_s,t1,t2)) # scores: (batch, N, h, T1, T2)
        # x, self.attn,score_his = res_attention(query, key, value,score_his, mask=mask, dropout=self.dropout)
        print(x.shape)
        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        # return self.linears[-1](x)
        return x

def aa():
    node_embeddings = nn.Parameter(torch.randn(170,3), requires_grad=True)
    print(node_embeddings.shape)
    supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
    print(supports.shape)

if __name__=="__main__":
    # x=torch.randn((64,12,10,32))
    # mha=MultiHeadAttention(8,64)
    # out=mha(x,x,x)
    aa()

