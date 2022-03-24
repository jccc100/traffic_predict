import matplotlib.pyplot as plt
import numpy as np


class PeMSD8():
    def __init__(self):
        super(PeMSD8, self).__init__()

    def MAE(self):
        VAR  = [17.78, 18.93, 19.52, 20.45, 21.26, 22, 22.97, 23.72, 24.45, 25.04, 25.6, 26.19]
        SAR  = [14.48, 16.26, 17.94,19.4, 20.92, 22.33, 23.98, 25.55, 27.03, 28.61, 30.39, 32.12]
        DCRNN  = [13.5,14.81,15.66,16.46,17.18,17.88,18.66,19.36,20.19,20.84,21.64,22.38]
        ASTGCN  = [14.35,15.59,16.48,17.24,18.05,18.69,19.36,20,20.62,21.15,21.89,22.41]
        STSGCN  = [14.45,15.28,15.67,16.08,16.5,16.95,17.38,17.81,18.14,18.61,18.94,19.49]
        STGCN  = [13.13,14.29,15.38,16.11,16.9,17.73,18.52,19.32,20.16,20.87,21.51,22.36]
        AGCRN  = []
        Ours_model=[13.46,13.85,14.25,14.58,14.87,15.15,15.42,15.70,15.95,16.17,16.46,16.86]
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD8 MAE")
        plt.xlabel("Time")
        plt.ylabel("MAE")

        plt.plot(my_x_ticks,VAR [:], 'red',label="VAR")
        plt.plot(my_x_ticks,VAR [:],'ro')

        plt.plot(my_x_ticks,SAR [:], 'green',label="SVR")
        plt.plot(my_x_ticks,SAR [:], 'go')

        plt.plot(my_x_ticks,DCRNN [:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN [:], 'bo')

        plt.plot(my_x_ticks,ASTGCN [:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN [:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN [:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN [:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN [:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN [:], 'o',color='blueviolet')

        # plt.plot(my_x_ticks,AGCRN [:],'black', label="AGCRN")
        # plt.plot(my_x_ticks,AGCRN [:], 'o',color='nlack')

        plt.plot(my_x_ticks,Ours_model[:], 'yellow',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='yellow')



        plt.legend()  # 显示图例
        plt.show()
    def RMSE(self):
        VAR = [26.99,28.77,29.77,31.09,32.19,33.41,34.51,35.71,36.62,37.47,38.28,39]
        SAR = [22.16,25.17,27.74,30.27,31.41,34.53,36.68,38.86,40.95,42.86,45,59.12]
        DCRNN= [20.75,22.98,24.81,25.77,26.62,27.92,28.97,29.91,31.12,31.91,32.96,34.72]
        ASTGCN = [21.77,23.65,25.38,26.51,27.32,28.19,29.18,30.2,30.78,31.65,32.49,33.75]
        STSGCN= [22.3,23.85,24.34,25.24,25.92,26.59,27.3,28,28.48,29.06,29.62,30.51]
        STGCN = [20.38,22.28,23.78,25.24,26.28,27.3,28.49,29.62,30.7,31.65,32.69,33.72]
        AGCRN = []
        Ours_model=[21.07,21.93,22.68,23.29,23.81,24.32,24.81,25.27,25.65,26.01,26.42,26.97]
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD8 RMSE")
        plt.xlabel("Time")
        plt.ylabel("RMSE")

        plt.plot(my_x_ticks,VAR[:], 'red',label="VAR")
        plt.plot(my_x_ticks,VAR[:],'ro')

        plt.plot(my_x_ticks,SAR[:], 'green',label="SVR")
        plt.plot(my_x_ticks,SAR[:], 'go')

        plt.plot(my_x_ticks,DCRNN[:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN[:], 'bo')

        plt.plot(my_x_ticks,ASTGCN[:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN[:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN[:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN[:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN[:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN[:], 'o',color='blueviolet')

        # plt.plot(my_x_ticks,AGCRN[:],'black', label="AGCRN")
        # plt.plot(my_x_ticks,AGCRN[:], 'o',color='nlack')

        plt.plot(my_x_ticks,Ours_model[:], 'yellow',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='yellow')
        # y_ticks=np.arange(20,50,5)
        # plt.yticks(y_ticks)


        plt.legend()  # 显示图例
        plt.show()
    def MAPE(self):
        VAR = [11.21,12.18,12.52,13.16,13.69,14.31,14.9,15.43,15.94,16.42,16.89,17.32]
        SAR = [9.05,10.11,11.08,12,13,14.05,15.09,16.14,17.32,18.46,19.58,20.81]
        DCRNN= [8.86,9.5,10,10.46,10.89,11.34,11.82,12.32,12.75,13.12,13.63,14.21]
        ASTGCN = [9.95,10.57,11.03,11.48,11.82,12.25,12.7,13.1,13.52,13.92,14.5,15.23]
        STSGCN= [9.48,9.85,10.14,10.35,10.57,10.81,11.07,11.28,11.48,11.66,11.91,12.27]
        STGCN = [8.66,9.28,9.77,10.23,10.61,11.05,11.46,11.85,12.26,12.64,12.95,13.34]
        AGCRN = []
        Ours_model=[8.73,8.93,9.15,9.32,9.51,9.68,9.91,10.12,10.27,10.50,10.71,11.01]
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD8 RMSE")
        plt.xlabel("Time")
        plt.ylabel("MAPE(%)")

        plt.plot(my_x_ticks,VAR[:], 'red',label="VAR")
        plt.plot(my_x_ticks,VAR[:],'ro')

        plt.plot(my_x_ticks,SAR[:], 'green',label="SVR")
        plt.plot(my_x_ticks,SAR[:], 'go')

        plt.plot(my_x_ticks,DCRNN[:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN[:], 'bo')

        plt.plot(my_x_ticks,ASTGCN[:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN[:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN[:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN[:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN[:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN[:], 'o',color='blueviolet')

        # plt.plot(my_x_ticks,AGCRN[:],'black', label="AGCRN")
        # plt.plot(my_x_ticks,AGCRN[:], 'o',color='nlack')

        plt.plot(my_x_ticks,Ours_model[:], 'yellow',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='yellow')
        # y_ticks=np.arange(20,50,5)
        # plt.yticks(y_ticks)


        plt.legend()  # 显示图例
        plt.show()

class PeMSD4():
    def __init__(self):
        super(PeMSD4, self).__init__()

    def MAE(self):
        VAR = [19.52,20.93,21.97,22.72,23.23,23.72,24.21,24.66,25.46,25.79,26.15,26.95]
        SAR = [18.84,20.67,22.5,24.23,25.94,27.72,29.51,31.25,33.08,34.82,36,42.6]
        DCRNN  = [17.62,19.05,20.3,21.28,22.19,23.17,24.17,25.22,26.16,27.07,28.14,29.32]
        ASTGCN  = [18.15,19.23,20.15,20.82,21.43,22.15,22.8,23.41,23.58,24.76,25.68,26.55]
        STSGCN  = [17.75,18.59,19.16,19.63,20.16,20.6,21.09,21.55,21.95,22.34,22.86,23.38]
        STGCN  = [16.8,18.14,19.23,20.28,21.06,21.83,22.74,23.72,24.86,25.48,26.19,26.97]
        AGCRN  = [18.6,18.63,18.81,19.05,19.31,19.56,19.76,19.94,20.08,20.28,20.65,21.18]
        Ours_model=[18.09,18.26,18.55,18.81,19.01,19.20,19.41,19.62,19.83,19.99,20.19,20.59]
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD4 MAE")
        plt.xlabel("Time")
        plt.ylabel("MAE")

        plt.plot(my_x_ticks,VAR [:], 'red',label="VAR")
        plt.plot(my_x_ticks,VAR [:],'ro')

        plt.plot(my_x_ticks,SAR [:], 'green',label="SVR")
        plt.plot(my_x_ticks,SAR [:], 'go')

        plt.plot(my_x_ticks,DCRNN [:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN [:], 'bo')

        plt.plot(my_x_ticks,ASTGCN [:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN [:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN [:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN [:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN [:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN [:], 'o',color='blueviolet')

        plt.plot(my_x_ticks,AGCRN [:],'black', label="AGCRN")
        plt.plot(my_x_ticks,AGCRN [:], 'o',color='nlack')

        plt.plot(my_x_ticks,Ours_model[:], 'yellow',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='yellow')



        plt.legend()  # 显示图例
        plt.show()
    def RMSE(self):
        VAR = [26.99,28.77,29.77,31.09,32.19,33.41,34.51,35.71,36.62,37.47,38.28,39]
        SAR = [22.16,25.17,27.74,30.27,31.41,34.53,36.68,38.86,40.95,42.86,45,59.12]
        DCRNN= [20.75,22.98,24.81,25.77,26.62,27.92,28.97,29.91,31.12,31.91,32.96,34.72]
        ASTGCN = [21.77,23.65,25.38,26.51,27.32,28.19,29.18,30.2,30.78,31.65,32.49,33.75]
        STSGCN= [22.3,23.85,24.34,25.24,25.92,26.59,27.3,28,28.48,29.06,29.62,30.51]
        STGCN = [20.38,22.28,23.78,25.24,26.28,27.3,28.49,29.62,30.7,31.65,32.69,33.72]
        AGCRN = [19.83,30.12,30.52,30.95,31.38,31.8,32.19,32.56,32.82,33.17,33.67,34.36]
        Ours_model=[21.07,21.93,22.68,23.29,23.81,24.32,24.81,25.27,25.65,26.01,26.42,26.97]
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD4 RMSE")
        plt.xlabel("Time")
        plt.ylabel("RMSE")

        plt.plot(my_x_ticks,VAR[:], 'red',label="VAR")
        plt.plot(my_x_ticks,VAR[:],'ro')

        plt.plot(my_x_ticks,SAR[:], 'green',label="SVR")
        plt.plot(my_x_ticks,SAR[:], 'go')

        plt.plot(my_x_ticks,DCRNN[:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN[:], 'bo')

        plt.plot(my_x_ticks,ASTGCN[:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN[:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN[:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN[:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN[:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN[:], 'o',color='blueviolet')

        # plt.plot(my_x_ticks,AGCRN[:],'black', label="AGCRN")
        # plt.plot(my_x_ticks,AGCRN[:], 'o',color='nlack')

        plt.plot(my_x_ticks,Ours_model[:], 'yellow',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='yellow')
        # y_ticks=np.arange(20,50,5)
        # plt.yticks(y_ticks)


        plt.legend()  # 显示图例
        plt.show()
    def MAPE(self):
        VAR = [11.21,12.18,12.52,13.16,13.69,14.31,14.9,15.43,15.94,16.42,16.89,17.32]
        SAR = [9.05,10.11,11.08,12,13,14.05,15.09,16.14,17.32,18.46,19.58,20.81]
        DCRNN= [8.86,9.5,10,10.46,10.89,11.34,11.82,12.32,12.75,13.12,13.63,14.21]
        ASTGCN = [9.95,10.57,11.03,11.48,11.82,12.25,12.7,13.1,13.52,13.92,14.5,15.23]
        STSGCN= [9.48,9.85,10.14,10.35,10.57,10.81,11.07,11.28,11.48,11.66,11.91,12.27]
        STGCN = [8.66,9.28,9.77,10.23,10.61,11.05,11.46,11.85,12.26,12.64,12.95,13.34]
        AGCRN = [12.21,12.16,12.16,12.4,12.55,12.68,12.81,12.89,12.95,13.07,13.29,13.67]
        Ours_model=[8.73,8.93,9.15,9.32,9.51,9.68,9.91,10.12,10.27,10.50,10.71,11.01]
        my_x_ticks=np.arange(5,65,5)
        # my_x_ticks=['5','10']

        plt.title("PEMSD4 RMSE")
        plt.xlabel("Time")
        plt.ylabel("MAPE(%)")

        plt.plot(my_x_ticks,VAR[:], 'red',label="VAR")
        plt.plot(my_x_ticks,VAR[:],'ro')

        plt.plot(my_x_ticks,SAR[:], 'green',label="SVR")
        plt.plot(my_x_ticks,SAR[:], 'go')

        plt.plot(my_x_ticks,DCRNN[:],'b',label="DCRNN")
        plt.plot(my_x_ticks,DCRNN[:], 'bo')

        plt.plot(my_x_ticks,ASTGCN[:], 'peru',label="ASTGCN")
        plt.plot(my_x_ticks,ASTGCN[:], 'o',color='peru')

        plt.plot(my_x_ticks,STGCN[:], 'pink',label="STGCN")
        plt.plot(my_x_ticks,STGCN[:], 'o',color='pink')

        plt.plot(my_x_ticks,STSGCN[:], 'blueviolet',label="STSGCN")
        plt.plot(my_x_ticks,STSGCN[:], 'o',color='blueviolet')

        # plt.plot(my_x_ticks,AGCRN[:],'black', label="AGCRN")
        # plt.plot(my_x_ticks,AGCRN[:], 'o',color='nlack')

        plt.plot(my_x_ticks,Ours_model[:], 'yellow',label="Ours")
        plt.plot(my_x_ticks,Ours_model[:], 'o',color='yellow')
        # y_ticks=np.arange(20,50,5)
        # plt.yticks(y_ticks)


        plt.legend()  # 显示图例
        plt.show()

if __name__=='__main__':
    PeMSD8=PeMSD8()
    PeMSD4=PeMSD4()
    # PeMSD8.MAE()
    # PeMSD8.RMSE()
    # PeMSD8.MAPE()
    PeMSD4.MAE()
    PeMSD4.RMSE()
    PeMSD4.MAPE()
