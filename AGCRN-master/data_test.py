# _*_ coding: utf-8 _*_
# @time     :2021/5/21 15:12
# @Author   :jc
# @File     :data_test.py
import numpy as np
import matplotlib.pyplot as plt


def get_flow(file_name):  # 将读取文件写成一个函数

    flow_data = np.load(file_name)  # 载入交通流量数据
    # print([key for key in flow_data.keys()])  # 打印看看key是什么

    # print(flow_data["data"].shape)  # (16992, 307, 3)，16992是时间(59*24*12)，307是节点数，3表示每一维特征的维度（类似于二维的列）
    flow_data = flow_data['data']  # [T, N, D]，T为时间，N为节点数，D为节点特征

    return flow_data


# 做工程、项目等第一步对拿来的数据进行可视化的直观分析
if __name__ == "__main__":
    traffic_data1 = get_flow("./data/PEMS03/PEMS03.npz") # 358个节点
    traffic_data2 = get_flow("./data/PEMSD4/PEMS04.npz") # 883个节点
    traffic_data3 = get_flow("./data/PEMS07/PEMS07.npz") # 883个节点
    traffic_data4 = get_flow("./data/PEMSD8/PEMS08.npz") # 883个节点
    traffic_data5 = get_flow("./data/PEMS04/PEMS04.npz") # 883个节点
    node_id = 10
    print('3',traffic_data1.shape)
    print('4',traffic_data2.shape)
    print('7',traffic_data3.shape)
    print('8',traffic_data4.shape)
    print('44',traffic_data4.shape)

    # plt.plot(traffic_data[:24 * 12, node_id, 0])  # 0维特征
    # plt.savefig("node_{:3d}_1.png".format(node_id))
    #
    # plt.plot(traffic_data[:24 * 12, node_id, 1])  # 1维特征
    # plt.savefig("node_{:3d}_2.png".format(node_id))
    #
    # plt.plot(traffic_data[:24 * 12, node_id, 2])  # 2维特征
    # plt.savefig("node_{:3d}_3.png".format(node_id))
