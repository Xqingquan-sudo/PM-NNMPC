import numpy as np
from scipy.linalg import norm, pinv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
import time
import csv
import math
import  pickle

def rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))  # 均方根误差

def mae(x, y):
    return np.mean(np.abs(x - y))  # 平均绝对误差

def mre(x, y):
    return np.mean(np.divide(np.abs(x - y), x))####平均相对误差

def smape(x, y):
    return 2.0 * np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y)))  # 对称平均绝对百分比误差

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

start1 = time.perf_counter()

np.random.seed(50)


class RBF:
    """
    RBF神经网络类
    """

    def __init__(self, input_dim, num_centers, out_dim):  # 输入变量数 中间层个数 输出变量数目
        self.input_dim = input_dim
        self.num_center = num_centers
        self.out_dim = out_dim
        self.beta = 0.1  # 标准差初始化定义
        self.centers = [np.random.uniform(-1, 1, input_dim) for i in range(num_centers)]  # 中心点初始赋值
        """
        中心点个数和number_centers个数相同，坐标数量和input_dim对应
        """
        self.W = np.random.random((self.num_center, self.out_dim))  # num_centers*out_dim矩阵形式

    def _basisfunc(self, c, d):  # 基函数计算
        return np.exp(-(self.num_center * (norm(c - d) ** 2)) / self.beta ** 2)

    def _calcAct(self, x):
        G = np.zeros((x.shape[0], self.num_center), dtype=np.float)  # 定义一个激活矩阵 样本数*中间层数
        for ci, c in enumerate(self.centers):
            for x1, xx in enumerate(x):
                G[x1, ci] = self._basisfunc(c, xx)
        return G

    def _calcbeat(self):  # 找到选取中心点最大值——及求解σ的值
        bate_temp = np.zeros((self.num_center, self.num_center))  # 定义一个矩阵 隐藏层中心值确定的
        for iindex, ivalue in enumerate(self.centers):
            for jindex, jvalue in enumerate(self.centers):
                bate_temp[iindex, jindex] = norm(ivalue - jvalue)  # 依次求解各中心层的值
        return np.argmax(bate_temp)  # 返回最大值

    def train(self, x, y):
        """
        :param x: 样本数*输入变量数
        :param y: 样本数*输出变量数
        :return:无
        """
        rnd_idx = np.random.permutation(x.shape[0])[:self.num_center]  # 随机我们的输入样本
        self.centers = [x[i, :] for i in rnd_idx]  # 根据随机值找打对应样本中心
        self.beta = 3400  # 4410   #    #self._calcbeat()  # 根据样本中心计算σ值
        print(self.beta)
        G = self._calcAct(x)  # 返回G值
        self.W = np.dot(pinv(G), y)  # 求解W值
        # print(self.W)
        pass

    def predict(self, x):
        G = self._calcAct(x)
        y = np.dot(G, self.W)
        return y

#读取训练与测试集
data = np.array(pd.read_csv('G:/博士/北科课题/数据集/故障3质量相关故障/pkx301.csv',engine='python'))
#data1=data.drop(['时间'],axis=1)
#归一化
data = data[1:]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)
reframed=np.array(scaled)

#划分训练数据和测试数据
#values = reframed.values     # (66442,19)
train = reframed[:-int(len(reframed)*0.2), :]
test  = reframed[-int(len(reframed)*0.2):, :]
# (53154,19)  (13288,19)

#拆分输入输出（未加入历史标签值）
#train_X,train_y,test_X,test_y = train[:, :-1], train[:, -1:],test[:, :-1], test[:, -1:]
# (53154,17) (53154,2)  (13288,17)   (13288,2)
#拆分输入输出（未加入历史标签值）
train_X,train_y,test_X,test_y = train[:-1], train[1:, -1:],test[:-1], test[1:, -1:]  #[:, 0:-1]




print ('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
x=train_X.shape[1]
rbf = RBF(train_X.shape[1],1610, 1)#1610 88%
rbf.train(train_X, train_y)

y_train = rbf.predict(train_X)
y_test_pred = rbf.predict(test_X)
#print(y_test_pred)

# with open('RBFmodle1.pickle','wb') as f:
#     pickle.dump(rbf,f)
print('训练指标')
print("train_MAE", mae(train_y, y_train), "train_RMSE", rmse(train_y, y_train),"train_MRE", mre(train_y, y_train))
print("train_SMAPE", smape(train_y, y_train))
print("train_R2", R2(train_y, y_train))
print("AIC", AIC(train_y, y_train, k=21, n=y_train.shape[0]))
print("BIC", BIC(train_y, y_train, k=21, n=y_train.shape[0]))



print('预测指标')
print("test_MAE", mae(test_y, y_test_pred), "test_RMSE", rmse(test_y, y_test_pred), "test_MRE",
      mre(test_y, y_test_pred))
print("test_SMAPE", smape(test_y, y_test_pred))
print("test_R2", R2(test_y, y_test_pred))
print("AIC", AIC(test_y, y_test_pred.reshape(-1,1), k=21, n=test_y.shape[0]))
print("BIC", BIC(test_y, y_test_pred.reshape(-1,1), k=21, n=test_y.shape[0]))

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.xlabel('Number of sample(s)',fontsize=18,labelpad = 1)#fontsize字体大小，labelpad距离坐标轴远近
# plt.ylabel('Thickness(mm)',fontsize=18,labelpad = 1)
plt.xlabel('Number of sample(s)')  # fontsize字体大小，labelpad距离坐标轴远近
plt.ylabel('Thickness(mm)')
x_n1 = range(0, len(y_test_pred))
plt.plot( y_test_pred, label="$pre$", color='r', linestyle="-", linewidth=1)
#y_test1 = y_test.reshape(-1,1)
plt.plot( test_y, label="$lab$", color='b', linestyle="--", linewidth=2)
plt.legend()
plt.show()