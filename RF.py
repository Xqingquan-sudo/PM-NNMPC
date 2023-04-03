from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler
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
#读取训练与测试集
data = np.array(pd.read_csv('G:/博士/北科课题/数据集/故障3质量相关故障/pkx301.csv',engine='python'))
#data1=data.drop(['时间'],axis=1)
#归一化

data = data[1:]
data = np.float64(data)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)
reframed=np.array(scaled)
#划分训练数据和测试数据
#values = reframed.values     # (66442,19)
train = reframed[:-int(len(data)*0.2), :]
test  = reframed[-int(len(data)*0.2):, :]
# (53154,19)  (13288,19)
#X_train, X_test, y_train, y_test = train[:, 0:-1],test[:, 0:-1],train[:, -1:],test[:, -1:]
X_train, X_test, y_train, y_test = train[:-1],test[:-1],train[1:, -1:],test[1:, -1:]



# criterion ：
# 回归树衡量分枝质量的指标，支持的标准有三种：
# 1）输入"mse"使用均方误差mean squared error(MSE)，父节点和叶子节点之间的均方误差的差额将被用来作为特征选择的标准，
# 这种方法通过使用叶子节点的均值来最小化L2损失
# 2）输入“friedman_mse”使用费尔德曼均方误差，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差
# 3）输入"mae"使用绝对平均误差MAE（mean absolute error），这种指标使用叶节点的中值来最小化L1损失
forest = RandomForestRegressor(n_estimators=1000,        #1000
                               criterion='mse',
                               random_state=1,         #1
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

# print('MSE train: %.3f, test: %.3f' % (
#         mean_squared_error(y_train, y_train_pred),
#         mean_squared_error(y_test, y_test_pred)))
# print('R^2 train: %.3f, test: %.3f' % (
#         r2_score(y_train, y_train_pred),
#         r2_score(y_test, y_test_pred)))

print('训练指标')
print("train_MAE", mae(y_train_pred, y_train), "train_RMSE", rmse(y_train_pred, y_train),"train_MRE", mre(y_train_pred, y_train))
print("train_SMAPE", smape(y_train_pred, y_train))
print("train_R2", R2(y_train_pred, y_train))
print("AIC", AIC(y_train_pred.reshape(-1,1), y_train, k=21, n=y_train.shape[0]))
print("BIC", BIC(y_train_pred.reshape(-1,1), y_train, k=21, n=y_train.shape[0]))

print('预测指标')
print("test_MAE", mae(y_test, y_test_pred), "test_RMSE", rmse(y_test, y_test_pred), "test_MRE",
      mre(y_test, y_test_pred))
print("test_SMAPE", smape(y_test, y_test_pred))
print("test_R2", R2(y_test, y_test_pred))
print("AIC", AIC(y_test, y_test_pred.reshape(-1,1), k=21, n=y_test.shape[0]))
print("BIC", BIC(y_test, y_test_pred.reshape(-1,1), k=21, n=y_test.shape[0]))

# 绘制残差图
# 残差分布似乎并不是围绕零中心点完全随机，这表明该模型不能捕捉所有的探索性信息
# plt.scatter(y_train_pred,
#             y_train_pred - y_train,
#             c='steelblue',
#             edgecolor='white',
#             marker='o',
#             s=35,
#             alpha=0.9,
#             label='training data')
# plt.scatter(y_test_pred,
#             y_test_pred - y_test,
#             c='limegreen',
#             edgecolor='white',
#             marker='s',
#             s=35,
#             alpha=0.9,
#             label='test data')

# plt.xlabel('Predicted values')
# plt.ylabel('Residuals')
# plt.legend(loc='upper left')
# plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
# plt.xlim([-10, 50])
# plt.tight_layout()
#
# # plt.savefig('images/10_14.png', dpi=300)
# plt.show()

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
plt.plot( y_test, label="$lab$", color='b', linestyle="--", linewidth=2)

plt.legend()
plt.show()


