import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from pylab import mpl
#数据读取

#故障1
Xtrain1 = np.array(pd.read_csv('G:/博士/北科课题/数据集/故障1质量无关故障/pkx201.csv',engine='python'))#3.9mm
Xtest1 = np.array(pd.read_csv('G:/博士/北科课题/数据集/故障1质量无关故障/pkx202.csv',engine='python'))
#故障2
Xtrain2 = np.array(pd.read_csv('G:/博士/北科课题/数据集/故障2质量相关故障/pkx101.csv',engine='python'))#2.7mm
Xtest2 = np.array(pd.read_csv('G:/博士/北科课题/数据集/故障2质量相关故障/pkx102.csv',engine='python'))
#故障3
Xtrain3 = np.array(pd.read_csv('G:/博士/北科课题/数据集/故障3质量相关故障/pkx301.csv',engine='python'))#3.9mm
Xtest3 = np.array(pd.read_csv('G:/博士/北科课题/数据集/故障3质量相关故障/pkx302.csv',engine='python'))

Xtrain = Xtrain3[1:]
Xtest = Xtest3[1:]
# #Xtest = Xtest1[499:3500]


#三种故障拼接
# Xtrain = np.r_[Xtrain1[1:],Xtrain3[1:]]
# Xtest = np.r_[Xtest1[1:],Xtest3[1:]]

Xtrain = Xtrain.astype(np.float32)
Xtest = Xtest.astype(np.float32)


def PCA(data):
    data_mean = np.mean(data,0)
    data_std = np.std(data,0)
    data_nor = (data - data_mean)/data_std
    X = np.cov(data_nor.T)
    P,v,P_t = np.linalg.svd(X)  #载荷矩阵计算  此函数返回三个值 u s v 此时v是u的转置
    Z = np.dot(P,P_t)
    v_sum = np.sum(v)
    k = []##主元个数
    for x in range(len(v)):
        PE_k = v[x]/v_sum
        if x == 0:
            PE_sum = PE_k
        else:
            PE_sum = PE_sum + PE_k
        if PE_sum < 0.85:   #累积方差贡献率
            pass
        else:
            k.append(x+1)
            print(k)
            break
    ##新主元
    p_k = P[:,:k[0]]
    v_I = np.diag(1/v[:k[0]])
    ##T统计量阈值计算
    coe = k[0]*(np.shape(data)[0]-1)*(np.shape(data)[0]+1)/((np.shape(data)[0]-k[0])*np.shape(data)[0])
    T_95_limit = coe*stats.f.ppf(0.95,k[0],(np.shape(data)[0]-k[0]))
    T_99_limit = coe*stats.f.ppf(0.99,k[0],(np.shape(data)[0]-k[0]))
    ##SPE统计量阈值计算
    O1 = np.sum((v[k[0]:])**1)
    O2 = np.sum((v[k[0]:])**2)
    O3 = np.sum((v[k[0]:])**3)
    h0 = 1 - (2*O1*O3)/(3*(O2**2))
    c_95 = 1.645
    c_99 = 2.325
    SPE_95_limit = O1*((h0*c_95*((2*O2)**0.5)/O1 + 1 + O2*h0*(h0-1)/(O1**2))**(1/h0))
    SPE_99_limit = O1*((h0*c_99*((2*O2)**0.5)/O1 + 1 + O2*h0*(h0-1)/(O1**2))**(1/h0))

    return v_I, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit,v,P,k


v_I, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit,v,P,k = PCA(Xtrain)

#计算T统计量
def T2(data_in, data_mean, data_std, p_k, v_I):
    test_data_nor = ((data_in - data_mean)/data_std).reshape(len(data_in),1)
    T_count = np.dot(np.dot((np.dot((np.dot(test_data_nor.T,p_k)), v_I)), p_k.T),test_data_nor)
    return T_count
 #计算SPE统计量
def SPE(data_in, data_mean, data_std, p_k):
    test_data_nor = ((data_in - data_mean)/data_std).reshape(len(data_in),1)
    I = np.eye(len(data_in))
    Q_count = np.dot(np.dot((I - np.dot(p_k, p_k.T)), test_data_nor).T,np.dot((I - np.dot(p_k, p_k.T)), test_data_nor))
    #Q_count = np.linalg.norm(np.dot((I - np.dot(p_k, p_k.T)), test_data_nor), ord=None, axis=None, keepdims=False)
    return Q_count


 #循环计算
test_data = Xtest
t_total = []
q_total = []
for x in range(np.shape(test_data)[0]):
    data_in = Xtest[x,:]
    t = T2(data_in, data_mean, data_std, p_k, v_I)
    q = SPE(data_in, data_mean, data_std, p_k)
    t_total.append(t[0,0])
    q_total.append(q[0,0])


mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

##画图
plt.figure(2,figsize=(16,9))
ax1=plt.subplot(2,1,1)
plt.plot(t_total,'b',label='$T^2$ statistic')
plt.plot(np.ones((len(test_data)))*T_99_limit,'r',linestyle='--',label='Threshold')
#ax1.set_ylim(0,100)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0,len(t_total))
ax1.set_xlabel(u'Samples',fontsize=14)
ax1.set_ylabel(u'$T^2$',fontsize=14)
plt.legend(fontsize=14)
#plt.show()

ax1=plt.subplot(2,1,2)
plt.plot(q_total,'b',label='Q statistic')
plt.plot(np.ones((len(test_data)))*SPE_99_limit,'r',linestyle='--',label='Threshold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.plot(q_total)
# plt.plot(np.ones((len(test_data)))*SPE_99_limit,'r',label='99% Q control limit')
#ax1.set_ylim(0,30)
#plt.xlim(0,100)
plt.xlim(0,len(q_total))
ax1.set_xlabel(u'Samples',fontsize=14)
ax1.set_ylabel(u'Q',fontsize=14)
plt.legend(fontsize=14)
plt.show()