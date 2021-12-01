# _*_ coding: utf-8 _*_
# @time     :2021/5/20 15:39
# @Author   :jc
# @File     :test.py

import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch

aa=torch.randn(32,12,107,1)
bb=aa[:,1,:,:]
print(aa.shape)
print(bb.shape)


exit()



data_pre=np.load('PEMSD8_pred.npy')
data_true=np.load('PEMSD8_true.npy')
print(data_pre.shape)
print(data_true.shape)
# print(data_pre[0, :, 2, :])
# print(data_true[0, :, 2, :])
d1=data_pre[0, :, 2, :]
d2=data_true[0, :, 2, :]
d=[d1,d2]
# plt.plot(d)
# print(data_pre[1])
plt.figure()
# plt.plot(data_pre[0,:,3,0])
# plt.plot(data_pre[0,:,3,0])
# plt.plot(data_true[0,:,3,0],color='red')

# d=np.reshape(d1,(12,170,1))
# print(data_true[0, :, 1, 0].shape)
my_d=[]
for i in range(170):
    my_d.append(data_true[0,:,i,0])
# print(len(my_d))
plt.plot(my_d[0])
plt.plot(my_d[1])

# plt.title('True')
plt.show()