import  pickle
from pylab import mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sko.DE import DE
#import data_processing as dp
from keras.models import  load_model
import pandas as pd
import GRU
import GRU_T
import Data_processing as DP
import time
from sklearn.preprocessing import MinMaxScaler
print('输入成功')


'''FD-NNMPC'''

'''优化'''
i=100  #初始值
P = 1#预测时域


model = load_model('gru.h5')# 调用控制模型
model_p = load_model('gru_T.h5')#调用预测模型


arr = np.zeros((1,GRU.T,GRU.test_X.shape[2]))
#arr = np.zeros((1, 11*P+1))
def obj_func(p):
    arr[0] = p
    #YRk_i
    #print(YRk_i)
    teYY = model.predict(arr)
    #J = np.square(YRk_i-teYY[0][0])
    #np.array(YRk_i)
    #J = teYY- 0.1
    J = (teYY-YRk_i)**2
    #J =teYY-YRk_i
    J_.append(J[0][0])

    return  J[0][0] # 误差平方最小
constraint_eq = [
    # lambda x: 1 - x[1] - x[2]
]
constraint_ueq = [
    # lambda x: 1 - x[0] * x[1],
    # lambda x: x[0] * x[1] - 5
    #lambda x: 0.20930 - x[15]
]

# lb = np.hstack((dp.trXm[i,0:10*P],np.array([0]*P),dp.trXm[i,0])).tolist()
# ub = np.hstack((dp.trXm[i,0:10*P],np.array([1]*P),dp.trXm[i,0])).tolist()
# lb = np.hstack(np.array([0]*20)).tolist()
# ub = np.hstack(np.array([1]*20)).tolist()

#B= obj_func

if __name__ == "__main__":
    '''Main'''

    Yk_=[]     #控制器输出
    Ym_=[]     #预测模型输出
    Yr_=[]     #设定值序列
    bestx_ = []
    besty_ = []  #优化后的结果
    J_=[]
    #data = np.array(pd.read_csv('H:/北科课题/程序调试/PCA/故障3质量相关故障/pkx301.csv', engine='python'))
    U = GRU_T.train_X0[1,:]  #求解的x
    Ux = GRU.train_X[i:i + 1,:,:]
    Ux_ = GRU_T.train_X[i:i + 1,:,:]
    # Ux = data[i:i + 2,:-1]
    # Ux_ = data[i:i + 2,:-1]
    #正常工况
    Yx =GRU.train_y[0:3000,0,0]#xuanlianb

    #异常工况
    #Yx = DP.Yx[500:3500]
    #Yx = DP.Yx[500:3500]  #3500

    for j in range(500,3500,1):
        start_time = time.time()
        print(j)

        Yk=model.predict(Ux)     #控制对象输出
        # with open('E:/徐清泉/预测控制/MPC/nn.pickle','rb') as f:
        #     nn= pickle.load(f)

        Ym=model_p.predict(Ux_)     #预测模型输出
        h= 0.5             #反馈系数
        Yp =Ym+h*(Yk-Ym)  #反馈校正

        if j<=1000:
            Yr = 0.15  #设定值
        else:
            Yr = 0.3

        #Yr = 0.2     #设定值恒定

        # if j<=1568:
        #     Yr = 0.2  #设定值
        # elif 1568< j <= 2585:
        #     Yr = 0.65
        # else:
        #     Yr = 0.2

        # if j<=2279:
        #     Yr = 0.2  #设定值
        # else:
        #     Yr = 0.65

        #Yr1 = 0.2
        Yr_.append(Yr)    #设定值序列
        a=0.5   #柔化系数
        global YRk_i
        YRk_i = a * Yk[0] + (1 - a) * Yr  #跟踪曲线
        #DE=DE.de
        # de = DE(func=obj_func, n_dim=11 * P+1, size_pop=3, max_iter=5,  # size_pop 种群个数， max_iter迭代次数，
        #         lb=lb, ub=ub, constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)
        de = DE(func=obj_func, n_dim=21, size_pop=8, max_iter=10,  # size_pop 8种群个数， max_iter迭代次数10，
                 constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)
        # 种群大小NP、缩放因子F与杂交概率CR
        best_x, best_y = de.run() #best为返回值J
        Ux_i = best_x
        # print(best_x)
        print('best_y',best_y)

        #Ux=np.c_[Ux_i, best_y]
        Ux1  = Ux_i.reshape((1,1,GRU.test_X.shape[2]))
        #Ux = Ux_i.reshape((1, -1))
        # print(Ux.shape,Ux1.shape)
        Ux = np.concatenate((Ux, Ux1), axis=1)
        Ux = Ux[:,1:,:]


        # print(Ux)
        Ux_ = np.concatenate((Ux_, Ux1), axis=1)
        Ux_ = Ux_[:,1:,:]  #暂时相等
        #Ux_ = Ux_i.reshape((1, -1))
        #besty = best_y**0.5+ YRk_i
        besty = best_y  + YRk_i
        print('Yk',Yk)
        Yk_.append(Yk[0,0,0])
        Ym_.append(Ym[0,0,0])
        besty_.append(besty[0,0])

        U = np.concatenate([U, best_x], axis=0)# U曲线
        end_time = time.time()
        print('time',end_time-start_time)
        #bestx_.append(best_x[0][11*P-1])


    '''反归一化'''
    # maxd = float(max(GRU.data[:, -1]))
    # mind = float(min(GRU.data[:, -1]))
    '''保存U'''

    # U = U.reshape((-1, Ux_.shape[2]))
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled = scaler.fit_transform(U)
    # U = np.array(scaled)
    # U = pd.DataFrame(U)
    #
    # U.to_csv('U.csv')

    maxd = float(max(DP.data[:,-1]))
    mind = float(min(DP.data[:,-1]))
    Yk_ = np.array(Yk_ )* (maxd - mind) + mind
    Yx =  np.array(Yx) * (maxd - mind) + mind
    Yr_ = np.array(Yr_) * (maxd - mind) + mind
    '''绘图'''
    mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    print(J_)
    print(besty_)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.xlabel('时间 (分钟)', fontsize=18, labelpad=1)   # fontsize字体大小，labelpad距离坐标轴远近
    # plt.ylabel('氮氧化物排放浓度', fontsize=18, labelpad=1)
    plt.xlabel('Number of sample(s)')  # fontsize字体大小，labelpad距离坐标轴远近
    plt.ylabel('Thickness(mm)')
    x_n1 = range(0,3000)
    plt.plot(Yk_, label="PM-NNMPC", color='r', linestyle="-", linewidth=1)      #预测控制 PM-NNMPC
    #plt.plot(x_n1, Ym_, label="$Ym$", color='k', linestyle="-", linewidth=1)      # 预测模型输出
    plt.plot(Yx, label="AGC",  color='b', linestyle="--", linewidth=1)      #原始数据 Traditional method
    plt.plot(Yr_, label="Set value", color='lime', linestyle="--", linewidth=1)  #设定值
    #plt.plot(x_n1, besty_, label="$bY$", color='magenta', linestyle="-", linewidth=2)  #优化结果
    #plt.step(x_n1,best_x,'r-')
#    plt.plot(x_n1, J_, label="$J$", color='lime', linestyle="-", linewidth=2)  #J
    plt.legend()
    plt.show()

    result = np.concatenate([Yk_, Yx, Yr_], axis=0)
    result = result.reshape(3,-1)
    print(result)

    df = pd.DataFrame(data=result)
    df.to_csv('故障2控制结果.csv')