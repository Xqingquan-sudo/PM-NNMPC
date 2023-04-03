
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
# from sklearn import preprocessing
import random
from sklearn import svm
import  pickle

#import xlrd
import scipy.io as sio
from pylab import mpl
import math

data = np.array(pd.read_csv('G:/博士/北科课题/数据集/故障3质量相关故障/pkx301.csv',engine='python'))


data = data[1:]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)
reframed=np.array(scaled)

#划分训练数据和测试数据
#values = reframed.values     # (66442,19)
train = reframed[:-int(len(reframed)*0.2), :]
test  = reframed[-int(len(reframed)*0.2):, :]
#分割训练集与测试集(预测五分钟时p为15)

# p=15   #预测时间
# trX = train_x[0:8]
# trY = train_y[0+p:8+p]
# teX = train_x[8:10]
# teY = train_y[8+p:10+p]
# trYI = train_y[0:8]
# teYI = train_y[8:10]
# i=1
# while i<1450 :
#     trX = np.r_[trX,train_x[10*i:10*i+7]]
#     trY = np.r_[trY,train_y[10*i+p:10*i+7+p]]
#     teX = np.r_[teX,train_x[10*i+8:10*i+9]]     # #####shape=(36,411)
#     teY = np.r_[teY,train_y[10*i+8+p:10*i+9+p]]
#     trYI = np.r_[trYI,train_y[10*i:10*i+7]]
#     teYI = np.r_[teYI,train_y[10*i+8:10*i+9]]
#     i+=1
# trX = np.c_[trX,trYI]
# teX = np.c_[teX,teYI]
# print('输入成功')
#
#
#
# '''归一化'''
# trX2=[]
# for i in range(len(trX[1])):
#     my_matrix = trX[:,i].reshape(-1,1)
#  #将数据集进行归一化处理
#     scaler = MinMaxScaler( )
#     scaler.fit(my_matrix)
#     scaler.data_max_
#     trX2.extend(scaler.transform(my_matrix))
# trX2 = np.array(trX2).reshape((12,-1))
#
# teX2=[]
# for i in range(len(teX[1])):
#     my_matrix = teX[:,i].reshape(-1,1)
#  #将数据集进行归一化处理
#     scaler = MinMaxScaler( )
#     scaler.fit(my_matrix)
#     scaler.data_max_
#     teX2.extend(scaler.transform(my_matrix))
# teX2 = np.array(teX2).reshape((12,-1))
#
# my_matrix = trY.reshape(-1,1)
#  #将数据集进行归一化处理
# scaler = MinMaxScaler( )
# scaler.fit(my_matrix)
# scaler.data_max_
# trY2 = scaler.transform(my_matrix)
#
# my_matrix = teY.reshape(-1,1)
#  #将数据集进行归一化处理
# scaler = MinMaxScaler( )
# scaler.fit(my_matrix)
# scaler.data_max_
# teY2 = scaler.transform(my_matrix)
# """
# prepare the forecast data
# """
# trX = trX2.astype(np.float32).transpose()
# trY = trY2.astype(np.float32)
# teX = teX2.astype(np.float32).transpose()
# teY = teY2.astype(np.float32)
trX,trY,teX,teY = train, train[:, -1:],test, test[:, -1:]
#预测p步后
P=2
trX = trX[P:,:]
trY = trY[:-P,:]
teX = teX[P:,:]
teY = teY[:-P,:]



print('输入成功')

X = trX
y = trY
clf = svm.SVR()
clf.fit(X, y)
teYY = clf.predict(teX)
teYY = teYY.reshape(-1,1)

trYY = clf.predict(trX)   #计算训练误差用
trYY = trYY.reshape(-1,1)
#print(result)
def rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))######均方根误差


def mre(x, y):
    return np.mean(np.divide(np.abs(x - y), x))####平均相对误差


def mae(x, y):
    return np.mean(np.abs(x - y))##################平均绝对误差

def smape(x, y):
    return 2.0 * np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y))) ########对称平均绝对百分比误差

def R2(x,y):
    return 1 - np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2)

def AIC(y_test, y_pred, k, n):
    resid = y_test - y_pred
    SSR = sum(resid ** 2)
    AICValue = 2*k+n*math.log(float(SSR)/n)
    return AICValue

def BIC(y_test, y_pred, k, n):
    resid = y_test - y_pred
    SSR = sum(resid ** 2)
    BICValue = k*math.log(n) + n*math.log(float(SSR)/n)
    return BICValue

# def R22(x,y):R2的另外一种算法
#     MSE = np.sum(np.power((x - y), 2)) / len(x)
#     R22 = 1 - MSE / np.var(x)
#     return R22



print("train_MRE", mre(trY[:, 0], trYY[:, 0]), "train_MAE", mae(trY[:, 0], trYY[:,    0]), "train_RMSE",
      rmse(trY[:, 0], trYY[:, 0]),"train_SMAPE",smape(trY[:, 0], trYY[:, 0]))
print("R2",R2(trY[:, 0], trYY[:, 0]))
print("AIC", AIC(trY[:, 0], trYY[:, 0], k=21, n=trYY[:, 0].shape[0]))
print("BIC", BIC(trY[:, 0], trYY[:, 0], k=21, n=trYY[:, 0].shape[0]))

print("test_MRE", mre(teY[:, 0], teYY[:, 0]), "test_MAE", mae(teY[:, 0], teYY[:,    0]), "test_RMSE",
      rmse(teY[:, 0], teYY[:, 0]))
print("test_SMAPE",smape(teY[:, 0], teYY[:, 0]))
print("R2",R2(teY[:, 0], teYY[:, 0]))
print("AIC", AIC(teY[:, 0], teYY[:, 0], k=21, n=teYY[:, 0].shape[0]))
print("BIC", BIC(teY[:, 0], teYY[:, 0], k=21, n=teYY[:, 0].shape[0]))

mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体#'Times New Roman','FangSong'
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# config = {
#     "font.family":'serif',
#     "font.size": 7.5,
#     "mathtext.fontset": 'stix',
#     "font.serif": ['SimSun'],
#
# }
#plt.rcParams.update(config)
with open('SVR.pickle','wb') as f:
    pickle.dump(clf,f)


dim1 = len(teY)
dim2 = len(trY)

#反归一化

maxd = float(max(data[:,-1]))
mind = float(min(data[:,-1]))
trY = trY * (maxd - mind) + mind
trYY = trYY * (maxd - mind) + mind
# for i in range(0, dim2):
#     trY[i, 0] = (trY[i, 0]) * (maxd - mind) + mind
#     trYY[i, 0] = (trYY[i, 0]) * (maxd - mind) + mind
teY = teY * (maxd - mind) + mind
teYY = teYY * (maxd - mind) + mind
# for i in range(0, dim1):
#     teY[i, 0] = (teY[i, 0]) * (maxd - mind) + mind
#     teYY[i, 0] = (teYY[i, 0]) * (maxd - mind) + mind

"反归一化后的误差"
# print( "train_MAE", mae(trY[:, 0], trYY[:,    0]), "train_RMSE",rmse(trY[:, 0], trYY[:, 0]))
#
# print( "test_MAE", mae(teY[:, 0], teYY[:,    0]), "test_RMSE",rmse(teY[:, 0], teYY[:, 0]))

"""设置图片"""
# # 设置输出的图片大小
#
# figsize = 11, 9
#
# figure, ax = plt.subplots(figsize=figsize)
#
# # 在同一幅图片上画两条折线
# A, = plt.plot(x1, y1, '-r', label='A', linewidth=5.0)
# B, = plt.plot(x2, y2, 'b-.', label='B', linewidth=5.0)
# # 设置图例并且设置图例的字体及大小
# font1 = {'family': 'Times New Roman','weight': 'normal', 'size': 23, }
# legend = plt.legend(handles=[A, B], prop=font1)
#
# # 设置坐标刻度值的大小以及刻度值的字体
# plt.tick_params(labelsize=23)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# # 设置横纵坐标的名称以及对应字体格式
# font2 = {'family': 'Times New Roman','weight': 'normal', 'size': 30, }
# plt.xlabel('round', font2)
# plt.ylabel('value', font2)



fig = plt.figure(1)###################脱硝NOx训练图
ax1 = fig.add_subplot(1, 1, 1)# 1, 1, 1 表示一行一列 第一个
#ax1.set_title('脱硝NOx预测')
plt.xlabel('Number of sample(s)')
plt.ylabel('Thickness(mm)')
x_n2 = range(0, dim2)
plt.plot(x_n2, trYY[:, 0], label="$train$", color='r', linestyle="-", linewidth=1)
plt.plot(x_n2, trY[:, 0], label="$lab$", color='b', linestyle="--", linewidth=2)
#plt.ylim(30,300)
plt.legend()
plt.show()

# """放大用"""
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# plt.plot(x_n2, trYY[:, 0],  color='r', linestyle="-", linewidth=1)
# plt.plot(x_n2, trY[:, 0],  color='b', linestyle="--", linewidth=2)
# plt.ylim(30, 300)
# plt.show()


fig = plt.figure(2)###################脱硝NOx预测图
ax1 = fig.add_subplot(1, 1, 1)# 1, 1, 1 表示一行一列 第一个

# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
#ax1.set_title('脱硝NOx预测')
# plt.xlabel('Number of sample(s)' ,fontsize=18,labelpad = 1)
# plt.ylabel('Thickness(mm)',fontsize=18,labelpad = 1)
plt.xlabel('Number of sample(s)')
plt.ylabel('Thickness(mm)')
x_n1 = range(0, dim1)
plt.plot(x_n1, teYY[:, 0], label="$pre$", color='r', linestyle="-", linewidth=1)
plt.plot(x_n1, teY[:, 0], label="$lab$", color='b', linestyle="--", linewidth=2)
#plt.ylim(30,300)
plt.legend()
plt.show()