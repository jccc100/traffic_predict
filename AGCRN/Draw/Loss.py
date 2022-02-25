# _*_ coding: utf-8 _*_
# @time     :2022/2/24 17:28
# @Author   :jc
# @File     :Loss.py
import matplotlib.pyplot as plt
import numpy as np
# 22-02-24 PEMSD4
file_path=[r"C:\Users\jc\Desktop\毕业相关\实验数据\2022-02-24\PEMS04\val_loss.npy",r"C:\Users\jc\Desktop\毕业相关\实验数据\2022-02-24\PEMS04\train_loss.npy"]
val_loss=np.load(file_path[0])
train_loss=np.load(file_path[1])
# print(val_loss)
# ax1=plt.subplot(1,2,1)
plt.title("PEMSD4")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(val_loss[:],label="val_loss")
plt.plot(train_loss[:],label="train_loss")
plt.legend() # 显示图例
# plt.show()
plt.savefig(r"C:\Users\jc\Desktop\毕业相关\实验数据\2022-02-24\PEMS04\图\train_val_loss.png")
