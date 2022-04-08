# _*_ coding: utf-8 _*_
# @time     :2022/2/24 17:27
# @Author   :jc
# @File     :Pred_True.py
# 22-02-24 PEMSD4
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

file_path_D4=[r"C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD4\3.1\PEMSD4_pred.npy",
           r"C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD4\3.1\PEMSD4_true.npy"]

file_path_D8=[r"C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD8\3.1\PEMSD8_pred.npy",
           r"C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD8\3.1\PEMSD8_true.npy"]

file_path_D8_noTA=[r"C:\Users\jc\Desktop\毕业相关\实验数据\TA-GCN_noTA\PEMSD8_pred.npy",
           r"C:\Users\jc\Desktop\毕业相关\实验数据\TA-GCN_noTA\PEMSD8_true.npy"]

file_path_D3=[r"C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD3\3.1\PEMS03_pred.npy",
           r"C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD3\3.1\PEMS03_true.npy"]
# 12代表0-60分钟的预测
def D4():
    pred_flow=np.load(file_path_D4[0])
    # pred_flow=pred_flow.reshape(3375,307,12)
    pred_flow = pred_flow.swapaxes(1, 2)
    true_flow=np.load(file_path_D4[1])
    # true_flow=true_flow.reshape(3375,307,12)
    true_flow = true_flow.swapaxes(1, 2)

    node = 111
    day=1
    shifting=250
    drow_pred_flow_15 = pred_flow[shifting:day*24 * 12+shifting, node, 2, :]
    drow_true_flow_15 = true_flow[shifting:day*24 * 12+shifting, node, 2, :]

    drow_pred_flow_30 = pred_flow[shifting:day*24 * 12+shifting, node, 5, :]
    drow_true_flow_30 = true_flow[shifting:day*24 * 12+shifting, node, 5, :]

    drow_pred_flow_60 = pred_flow[shifting:day*24 * 12+shifting, node, 11, :]
    drow_true_flow_60 = true_flow[shifting:day*24 * 12+shifting, node, 11, :]

    # ax = plt.gca()   #表明设置图片的各个轴，plt.gcf()表示图片本身

    # ax.xaxis.set_major_locator(mdates.HourLocator())  # 横坐标标签显示的日期格式
    # xs = ['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00','24:00']
    #  = [datetime.strptime(d, '%Y%m%d%H') for d in dates]
    # fig = plt.figure( [figsize=(6,3)] )
    # fig=plt.figure(figsize=(12,15))
    fig=plt.figure(figsize=(8,5))
    # 15 min
    # ax1=plt.subplot(3,1,1)
    # plt.plot(drow_pred_flow_15,label="Pred_flow")
    # plt.plot(drow_true_flow_15,label="Truth_flow")
    # plt.title("PEMSD4 15min")
    # plt.xlabel("Time")
    # plt.ylabel("Traffic Flow")
    # plt.legend() # 显示图例

    # # 30 min
    # ax2=plt.subplot(3,1,2)
    # plt.plot(drow_pred_flow_30,label="pred_flow")
    # plt.plot(drow_true_flow_30,label="true_flow")
    # plt.title("PEMSD4 30min")
    # plt.xlabel("Time")
    # plt.ylabel("Traffic Flow")
    # plt.legend() # 显示图例
    # #
    # # # 60 min
    # ax3=plt.subplot(3,1,3)
    plt.plot(drow_pred_flow_60,label="Pred_flow")
    plt.plot(drow_true_flow_60,label="Trurh_flow")
    plt.title("PEMSD4 60min")
    plt.xlabel("Time")
    plt.ylabel("Traffic Flow")

    plt.legend() # 显示图例

    plt.savefig(r"C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD4\3.1\Pred_True_{}_day{}_111.png".format(str(node),str(day)))

    plt.show()


def D3():
    pred_flow = np.load(file_path_D3[0])
    pred_flow = pred_flow.swapaxes(1, 2)
    # print(pred_flow.shape)
    true_flow = np.load(file_path_D3[1])
    true_flow = true_flow.swapaxes(1, 2)
    node=111
    day=1
    shifting=50
    drow_pred_flow_15 = pred_flow[shifting:day*24 * 12+shifting, node, 2,:]
    drow_true_flow_15 = true_flow[shifting:day*24 * 12+shifting, node, 2,:]

    drow_pred_flow_30 = pred_flow[shifting:day*24 * 12+shifting, node, 5,:]
    drow_true_flow_30 = true_flow[shifting:day*24 * 12+shifting, node, 5,:]

    drow_pred_flow_60 = pred_flow[shifting:day*24 * 12+shifting, node, 11,:]
    drow_true_flow_60 = true_flow[shifting:day*24 * 12+shifting, node, 11,:]

    # ax = plt.gca()   #表明设置图片的各个轴，plt.gcf()表示图片本身

    # ax.xaxis.set_major_locator(mdates.HourLocator())  # 横坐标标签显示的日期格式
    # xs = ['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00','24:00']
    #  = [datetime.strptime(d, '%Y%m%d%H') for d in dates]
    # fig = plt.figure( [figsize=(6,3)] )
    fig = plt.figure(figsize=(12, 14))
    # 15 min
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(drow_pred_flow_15, label="pred_flow")
    plt.plot(drow_true_flow_15, label="true_flow")
    plt.title("PEMSD3 15min")
    plt.xlabel("Time")
    plt.ylabel("Traffic Flow")
    plt.legend()  # 显示图例

    # 30 min
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(drow_pred_flow_30, label="pred_flow")
    plt.plot(drow_true_flow_30, label="true_flow")
    plt.title("PEMSD3 30min")
    plt.xlabel("Time")
    plt.ylabel("Traffic Flow")
    plt.legend()  # 显示图例

    # 60 min
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(drow_pred_flow_60, label="pred_flow")
    plt.plot(drow_true_flow_60, label="true_flow")
    plt.title("PEMSD3 60min")
    plt.xlabel("Time")
    plt.ylabel("Traffic Flow")

    plt.legend()  # 显示图例

    plt.savefig(r"C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD3\3.1\Pred_True_{}_day{}.png".format(str(node),str(day)))
    plt.show()

def D8():
    pred_flow = np.load(file_path_D8[0])
    # pred_flow = pred_flow.reshape(3375, 307, 12)
    pred_flow=pred_flow.swapaxes(1,2)
    true_flow = np.load(file_path_D8[1])
    # true_flow = true_flow.reshape(3375, 307, 12)
    true_flow=true_flow.swapaxes(1,2)

    node = 110
    day=2
    shifting=50
    drow_pred_flow_15 = pred_flow[shifting+9:day*24 * 12+shifting+9, node, 2, :]
    drow_true_flow_15 = true_flow[shifting+9:day*24 * 12+shifting+9, node, 2, :]

    drow_pred_flow_30 = pred_flow[shifting+6:day*24 * 12+shifting+6, node, 5, :]
    drow_true_flow_30 = true_flow[shifting+6:day*24 * 12+shifting+6, node, 5, :]

    drow_pred_flow_60 = pred_flow[shifting:day*24 * 12+shifting, node, 11, :]
    drow_true_flow_60 = true_flow[shifting:day*24 * 12+shifting, node, 11, :]

    # ax = plt.gca()   #表明设置图片的各个轴，plt.gcf()表示图片本身

    # ax.xaxis.set_major_locator(mdates.HourLocator())  # 横坐标标签显示的日期格式
    # xs = ['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00','24:00']
    #  = [datetime.strptime(d, '%Y%m%d%H') for d in dates]
    # fig = plt.figure( [figsize=(6,3)] )
    # fig = plt.figure(figsize=(8, 5))
    fig = plt.figure(figsize=(22, 15))
    # 15 min
    ax1 = plt.subplot(3, 1, 1)
    # ax1 = plt.subplot(1, 1, 1)
    # shang_x, shang_y, xia_x, xia_y, zuo_x, zuo_y, you_x, you_y = huakuang(486, 320, 62, 96)
    # plt.plot(shang_x, shang_y, 'red', linewidth=2.0)
    # plt.plot(zuo_x, zuo_y, 'red', linewidth=2.0)
    # plt.plot(xia_x, xia_y, 'red', linewidth=2.0)
    # plt.plot(you_x, you_y, 'red', linewidth=2.0)
    plt.plot(drow_pred_flow_15, label="pred_flow")
    plt.plot(drow_true_flow_15, label="Truth_flow")
    plt.title("PEMSD8 15min")
    plt.xlabel("Time")
    plt.ylabel("Traffic Flow")
    plt.legend()  # 显示图例

    # 30 min
    ax2 = plt.subplot(3, 1, 2)
    # # ax2 = plt.subplot(3, 1, 2)
    # shang_x, shang_y, xia_x, xia_y, zuo_x, zuo_y, you_x, you_y = huakuang(486, 320, 62, 96)
    # plt.plot(shang_x, shang_y, 'red', linewidth=2.0)
    # plt.plot(zuo_x, zuo_y, 'red', linewidth=2.0)
    # plt.plot(xia_x, xia_y, 'red', linewidth=2.0)
    # plt.plot(you_x, you_y, 'red', linewidth=2.0)
    plt.plot(drow_pred_flow_30, label="pred_flow")
    plt.plot(drow_true_flow_30, label="Truth_flow")
    plt.title("PEMSD8 30min")
    plt.xlabel("Time")
    plt.ylabel("Traffic Flow")
    plt.legend()  # 显示图例

    # # 60 min
    ax3 = plt.subplot(3, 1, 3)
    # 画框
    # shang_x,shang_y,xia_x,xia_y,zuo_x,zuo_y,you_x,you_y=huakuang(486,320,62,96)
    # plt.plot(shang_x,shang_y,'red',linewidth=2.0)
    # plt.plot(zuo_x,zuo_y,'red',linewidth=2.0)
    # plt.plot(xia_x,xia_y,'red',linewidth=2.0)
    # plt.plot(you_x,you_y,'red',linewidth=2.0)
    #
    # # 画线
    plt.plot(drow_pred_flow_60, label="Pred_flow")
    plt.plot(drow_true_flow_60, label="Truth_flow")
    plt.title("PEMSD8 60min")
    plt.xlabel("Time")
    plt.ylabel("Traffic Flow")
    plt.legend()  # 显示图例

    plt.savefig(r"C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD8\3.1\Pred_True_{}_day{}_15min.png".format(str(node),str(day)))
    plt.show()

def different_node():
    true_flow = np.load(file_path_D8[1])
    true_flow = true_flow.swapaxes(1, 2)
    node = 111
    day = 2
    shifting = 110
    drow_true_flow_1 = true_flow[shifting:day * 24 * 12 + shifting, 1, 11, :]
    drow_true_flow_2 = true_flow[shifting:day * 24 * 12 + shifting, 16, 11, :]
    drow_true_flow_3 = true_flow[shifting:day * 24 * 12 + shifting, 121, 11, :]
    drow_true_flow_4 = true_flow[shifting:day * 24 * 12 + shifting, 164, 11, :]
    fig = plt.figure(figsize=(12, 6))

    plt.plot(drow_true_flow_1, label="node1")
    plt.plot(drow_true_flow_2, label="node2")
    plt.plot(drow_true_flow_3, label="node3")
    plt.plot(drow_true_flow_4, label="node3")
    # plt.title("PEMSD3 60min")
    plt.xlabel("Time")
    plt.ylabel("Traffic Flow")
    plt.legend()  # 显示图例
    # plt.savefig(r"C:\Users\jc\Desktop\毕业相关\实验数据\PEMSD3\3.1\Pred_True_{}_day{}.png".format(str(node), str(day)))
    plt.show()

def huakuang(shang,xia,zuo,you):
    shang_y = []
    shang_x = []
    xia_x = []
    xia_y = []
    for i in range(zuo, you):
        shang_y.append(shang)
        shang_x.append(i)
        xia_y.append(xia)
        xia_x.append(i)
    zuo_x = []
    zuo_y = []
    you_x = []
    you_y = []
    for i in range(xia, shang):
        zuo_x.append(zuo)
        zuo_y.append(i)
        you_x.append(you)
        you_y.append(i)
    return shang_x,shang_y,xia_x,xia_y,zuo_x,zuo_y,you_x,you_y

if __name__=="__main__":
    # D3()
    # D4()
    D8()
    # different_node()
