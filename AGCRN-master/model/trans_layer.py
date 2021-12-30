# _*_ coding: utf-8 _*_
# @time     :2021/12/30 12:08
# @Author   :jc
# @File     :trans_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import math

# device=torch.device('cuda')
device=torch.device('cpu')

class Transform(nn.Module):
    def __init__(self, outfea, d):
        super(Transform, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea)
        )

        self.d = d

    def forward(self, x,score_his=None):
        query = self.qff(x)
        key = self.kff(x)
        value = self.vff(x)

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)
        # print(query.shape)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 2, 3, 1)
        # print(key.shape)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)

        A = torch.matmul(query, key)
        # print("A:",A.shape)
        A /= (self.d ** 0.5)
        if score_his is not None:
            A=A+score_his
        score_his=A
        A = torch.softmax(A, -1)

        value = torch.matmul(A, value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0, 2, 1, 3)
        value += x

        value = self.ln(value)
        x = self.ff(value) + value
        return self.lnff(x),score_his


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, outfea, max_len=12):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, outfea).to(device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, outfea, 2) *
                             -(math.log(10000.0) / outfea))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)  # [1,T,1,F]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe,
                         requires_grad=False)
        return x

class transformer_layer(nn.Module):
    def __init__(self,dim_in,dim_out,num_layer,d):
        super(transformer_layer,self).__init__()
        # self.linear1=nn.Linear(dim_in,dim_out)
        self.trans_layers=nn.ModuleList(Transform(dim_out,d) for l in range(num_layer))
        self.PE=PositionalEncoding(dim_out)
        self.num_layer=num_layer

    def forward(self, x):
        # x=self.linear1(x)
        x=self.PE(x)
        for l in range(self.num_layer):
            x,self.score_his=self.trans_layers[l](x)
        return x

if __name__=="__main__":
    x = torch.randn(32, 12, 170, 64)
    dim_in=64
    dim_out=64
    num_layer=2
    d=2
    transformer=transformer_layer(dim_in,dim_out,num_layer,d)
    res=transformer(x)
    print(res.shape)