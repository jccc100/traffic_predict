# _*_ coding: utf-8 _*_
# @time     :2022/2/24 17:27
# @Author   :jc
# @File     :Pred_True.py
# 22-02-24 PEMSD4
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

file_path_D4=[r"C:\Users\jc\Desktop\毕业相关\实验数据\2022-02-24\PEMS04\PEMSD4_pred.npy",
           r"C:\Users\jc\Desktop\毕业相关\实验数据\2022-02-24\PEMS04\PEMSD4_true.npy"]

file_path_D8=[r"C:\Users\jc\Desktop\毕业相关\实验数据\2022-02-24\PEMS04\PEMSD4_pred.npy",
           r"C:\Users\jc\Desktop\毕业相关\实验数据\2022-02-24\PEMS04\PEMSD4_true.npy"]

file_path_D3=[r"C:\Users\jc\Desktop\毕业相关\实验数据\2022-02-24\PEMS04\PEMSD4_pred.npy",
           r"C:\Users\jc\Desktop\毕业相关\实验数据\2022-02-24\PEMS04\PEMSD4_true.npy"]
# 12代表0-60分钟的预测
pred_flow=np.load(file_path[0])
pred_flow=pred_flow.reshape(3375,307,12)
true_flow=np.load(file_path[1])
true_flow=true_flow.reshape(3375,307,12)

drow_pred_flow_15=pred_flow[:24*12,111,2]
drow_true_flow_15=true_flow[:24*12,111,2]

drow_pred_flow_30=pred_flow[:24*12,111,5]
drow_true_flow_30=true_flow[:24*12,111,5]

drow_pred_flow_60=pred_flow[:24*12,111,11]
drow_true_flow_60=true_flow[:24*12,111,11]

# ax = plt.gca()   #表明设置图片的各个轴，plt.gcf()表示图片本身

# ax.xaxis.set_major_locator(mdates.HourLocator())  # 横坐标标签显示的日期格式
# xs = ['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00','24:00']
#  = [datetime.strptime(d, '%Y%m%d%H') for d in dates]
# fig = plt.figure( [figsize=(6,3)] )
fig=plt.figure(figsize=(12,14))
# 15 min
ax1=plt.subplot(3,1,1)
plt.plot(drow_pred_flow_15,label="pred_flow")
plt.plot(drow_true_flow_15,label="true_flow")
plt.title("PEMSD4 15min")
plt.xlabel("Time")
plt.ylabel("Traffic Flow")
plt.legend() # 显示图例

# 30 min
ax2=plt.subplot(3,1,2)
plt.plot(drow_pred_flow_30,label="pred_flow")
plt.plot(drow_true_flow_30,label="true_flow")
plt.title("PEMSD4 30min")
plt.xlabel("Time")
plt.ylabel("Traffic Flow")
plt.legend() # 显示图例

# 60 min
ax3=plt.subplot(3,1,3)
plt.plot(drow_pred_flow_60,label="pred_flow")
plt.plot(drow_true_flow_60,label="true_flow")
plt.title("PEMSD4 60min")
plt.xlabel("Time")
plt.ylabel("Traffic Flow")

plt.legend() # 显示图例

plt.savefig(r"C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD4\train_val_loss.png")
plt.show()