import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import *
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
import math
from keras import optimizers
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体#'Times New Roman','FangSong'
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# config = {
#     "font.family":'serif',
#     "font.size": 7.5,
#     "mathtext.fontset": 'stix',
#     "font.serif": ['SimSun'],
#
# }
# plt.rcParams.update(config)


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
train_X,train_y,test_X,test_y = train, train[:, -1:],test, test[:, -1:]


#reshape输入为LSTM的输入格式
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))  #input_shape=(samples, time_steps, input_dim)
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
train_y = train_y.reshape((train_y.shape[0], 1, 1))#output_shape=(samples, time_steps, output_dim)
test_y = test_y.reshape((test_y.shape[0], 1, 1))

print ('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# time_steps=T,
'''GRU输入格式'''
T=5
train_X2=[]
for i in range(T,train_X.shape[0]):
    train_X1 = train_X[i-T:i,0,:]
    train_X2=np.append(train_X2,train_X1)
train_X=train_X2.reshape(-1,T,train_X.shape[2])

test_X2=[]
for i in range(T,test_X.shape[0]):
    test_X1 = test_X[i-T:i,0,:]
    test_X2=np.append(test_X2,test_X1)
test_X =test_X2.reshape(-1,T,test_X.shape[2])

train_y2=[]
for i in range(T,train_y.shape[0]):
    train_y1 = train_y[i-T:i,0,:]
    train_y2=np.append(train_y2,train_y1)
train_y=train_y2.reshape(-1,T,train_y.shape[2])

test_y2=[]
for i in range(T,test_y.shape[0]):
    test_y1 = test_y[i-T:i,0,:]
    test_y2=np.append(test_y2,test_y1)
test_y =test_y2.reshape(-1,T,test_y.shape[2])

#np.concatenate()方法
# T=2
# train_X2=np.empty((1,T,train_X.shape[2]))
# for i in range(T,train_X.shape[0]):
#     train_X1 = train_X[i-T:i,0,:].reshape((1,T,train_X.shape[2]))
#     train_X2=np.concatenate((train_X2,train_X1),axis=0)
# train_X=train_X2
#
#
# test_X2=np.empty((1,T,test_X.shape[2]))
# for i in range(T,test_X.shape[0]):
#     test_X1 = test_X[i-T:i,0,:].reshape((1,T,test_X.shape[2]))
#     test_X2=np.concatenate((test_X2,test_X1),axis=0)
# test_X =test_X2
#
# train_y2=np.empty((1,T,train_y.shape[2]))
# for i in range(T,train_y.shape[0]):
#     train_y1 = train_y[i-T:i,0,:].reshape((1,T,train_y.shape[2]))
#     train_y2=np.concatenate((train_y2,train_y1),axis=0)
# train_y=train_y2
#
# test_y2=np.empty((1,T,test_y.shape[2]))
# for i in range(T,test_y.shape[0]):
#     test_y1 = test_y[i-T:i,0,:].reshape((1,T,test_y.shape[2]))
#     test_y2=np.concatenate((test_y2,test_y1),axis=0)
# test_y =test_y2

if __name__ == "__main__":
# GUR
    model = Sequential()
    model.add(GRU(10, activation='relu', input_shape=(T, train_X.shape[2]), return_sequences=True))      # input_shape = (1,17);input_shape=(train_X.shape[1], train_X.shape[2])
    model.add(GRU(20, activation='relu',  return_sequences=True))
    model.add(GRU(10, activation='relu',  return_sequences=True))

    model.add(Dropout(0.2))
    model.add(Dense(1))
    '''学习率实验'''
    # sgd = optimizers.SGD(lr=0.5, decay=0, momentum=0, nesterov=True)
    # #lr：大或等于0的浮点数，学习率LearningRate，可以自己定义大小；decay：衰减率，用于对学习率进行衰减运算;momentum：大或等于0的浮点数，动量参数;nesterov：布尔值，确定是否使用Nesterov动量;注：如果想设置成固定学习率，只要把衰减率decay和动量momentum都设置为0即可
    #model.compile(loss='mae',optimizer=sgd)#,metrics='mae'
    model.compile(loss='mae',optimizer='adam')#,metrics='mae'
    model.summary()
#模型训练
    history = model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=200, batch_size=30, verbose=2,shuffle=False) #epochs=100, batch_size=16, validation_data=(test_X, test_y)为验证集
#预测
    y_pre = model.predict(test_X)
#loss,acc = model.evaluate(x=test_X,y=test_y,batch_size=16,verbose=1) #适用于训练后



# # train-test-mse
    plt.plot(history.history['loss'], label='train-loss')
    plt.plot(history.history['val_loss'], label='test-loss')
    plt.legend()
#plt.savefig("loss.png", dpi=750)
    plt.show()

# # train-test-mae
# plt.plot(history.history['mae'], label='train-mae')
# plt.plot(history.history['val_mae'], label='test-mae')
# plt.legend()
# #plt.savefig("mae.png", dpi=750)
# plt.show()

#训练评价指标
    y_lab = model.predict(train_X)
    waviness_train = y_lab[:,0]

    waviness_lab = train_y[:,0]
    print('训练指标')
    print("train_MAE", mae(waviness_lab, waviness_train), "train_RMSE", rmse(waviness_lab, waviness_train),"train_MRE", mre(waviness_lab, waviness_train))
    print("train_SMAPE", smape(waviness_lab, waviness_train))
    print("train_R2", R2(waviness_lab, waviness_train))
    print("AIC", AIC(waviness_lab, waviness_train, k=21, n=waviness_train.shape[0]))
    print("BIC", BIC(waviness_lab, waviness_train, k=21, n=waviness_train.shape[0]))
    #反归一化
    maxd = float(max(data[:,-1]))
    mind = float(min(data[:,-1]))
    waviness_train = waviness_train * (maxd - mind) + mind
    waviness_lab = waviness_lab * (maxd - mind) + mind

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.xlabel('Number of sample(s)',fontsize=18,labelpad = 1)#fontsize字体大小，labelpad距离坐标轴远近
    # plt.ylabel('Thickness(mm)',fontsize=18,labelpad = 1)
    plt.xlabel('Number of sample(s)')#fontsize字体大小，labelpad距离坐标轴远近
    plt.ylabel('Thickness(mm)')
    x_n1 = range(0, len(waviness_train))
    plt.plot(x_n1, waviness_train, label="$train$", color='r', linestyle="-", linewidth=1)
    plt.plot(x_n1, waviness_lab, label="$lab$", color='b', linestyle="--", linewidth=2)

    plt.legend()
    plt.show()


# 测试集预测 评价指标

    waviness_pre = y_pre[:,0]
    waviness_y = test_y[:,0]

    print('预测指标')
    print("test_MAE", mae(waviness_y, waviness_pre), "test_RMSE", rmse(waviness_y, waviness_pre),"test_MRE", mre(waviness_lab, waviness_train))
    print("test_SMAPE", smape(waviness_y, waviness_pre))
    print("test_R2", R2(waviness_y, waviness_pre))
    print("AIC", AIC(waviness_y, waviness_pre, k=21, n=waviness_y.shape[0]))
    print("BIC", BIC(waviness_y, waviness_pre, k=21, n=waviness_y.shape[0]))

    waviness_pre = waviness_pre * (maxd - mind) + mind
    waviness_y = waviness_y * (maxd - mind) + mind

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.xlabel('Number of sample(s)',fontsize=18,labelpad = 1)#fontsize字体大小，labelpad距离坐标轴远近
    # plt.ylabel('Thickness(mm)',fontsize=18,labelpad = 1)
    plt.xlabel('Number of sample(s)')  # fontsize字体大小，labelpad距离坐标轴远近
    plt.ylabel('Thickness(mm)')
    x_n1 = range(0, len(waviness_pre))
    plt.plot(x_n1, waviness_pre, label="$pre$", color='r', linestyle="-", linewidth=1)
    plt.plot(x_n1, waviness_y, label="$lab$", color='b', linestyle="--", linewidth=2)
    #plt.ylim(30,300)
    plt.legend()
    plt.show()

    model.save('gru.h5')