import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from xgboost import plot_importance
from xgboost import XGBClassifier
#from sklearn.metrics import accuracy_score, plot_confusion_matrix
from matplotlib import pyplot
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import keras
time_start = time.time()  # 记录开始时间

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


data1 = np.array(pd.read_csv('G:/博士/北科课题/数据集/故障分类/故障1.csv',engine='python'))
data2 = np.array(pd.read_csv('G:/博士/北科课题/数据集/故障分类/故障2.csv',engine='python'))
data3 = np.array(pd.read_csv('G:/博士/北科课题/数据集/故障分类/故障3.csv',engine='python'))

"""包含正常工况数据"""
# train=np.r_[data1[0:1799],data1[1999:],data2[:2799],data2[2999:],data3[:1799],data3[2099:]]
# test1 =data1[1799:1999]
# test2 =data2[2799:2999]
# test3 =data3[1799:2099]
# test = np.r_[test1,test2,test3]

"""故障数据"""
train=np.r_[data1[1000:1799],data2[2000:2799],data3[1000:1799],data3[2099:2499]]
test1 =data1[1799:1999]
test2 =data2[2799:2999]
test3 =data3[1799:2099]
test = np.r_[test1,test2,test3]

"""正常+故障数据"""
# train = np.r_[data1[:499],data1[:1799],data1[999:1799],data2[:2799],data2[2999:],data3[:1799],data3[2099:]]
# test0 = data1[499:999]
# test1 = data1[1799:1999]
# test2 = data2[2799:2999]
# test3 = data3[1799:2099]
# test = np.r_[test0,test1,test2,test3]


"""分割特征和标签"""
train_x = train[:,0:-1]
train_Y = train[:,-1]
test_x  = test[:,0:-1]
test_Y  = test[:,-1]

le = LabelEncoder()
train_Y = le.fit_transform(train_Y)
test_Y = le.fit_transform(test_Y)

model = keras.models.Sequential()
# flatten层的作用是将28*28维度的输入数据展平成一层
#model.add(keras.layers.Flatten(input_shape =  [1,Xtrain.shape[1]]))
model.add(keras.layers.Dense(21, activation="relu"))
model.add(keras.layers.Dense(40, activation="relu"))
model.add(keras.layers.Dense(80, activation="relu"))
model.add(keras.layers.Dense(21, activation="relu"))

#model.compile(loss='mae',optimizer='adam')#,metrics='mae'
#history = model.fit(Xtrain, epochs=200, batch_size=30, verbose=2,shuffle=False)#validation_data=(test_X, test_y)
# Xtrain = model(Xtrain)
# Xtrain = np.array(Xtrain)
train_x = model.predict(train_x)
test_x = model.predict(test_x)

"""XGBoost训练过程"""
seed = 7
test_size = 0.33
model = XGBClassifier()
model.fit(train_x, train_Y)
# print(model.feature_impotances_)
plot_importance(model)  # 绘制特征重要性
pyplot.show()
# make predictions for test data
y_pred = model.predict(test_x)
predictions = [round(value) for value in y_pred]
# evaluate predictions
#accuracy = accuracy_score(test_Y, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))
# model.score(X_test,y_test) #你能想出这里应该返回什么模型评估指标么？

"评估"
tp = 0
fp = 0
p = 0
for i in range(0, len(test_Y)):
    if test_Y[i] == 1:
        p = p+1
        if y_pred[i] == 1:
            tp = tp+1
    elif y_pred[i] == 1:
            fp = fp+1

print("recall:", 100*tp/p, "%")
print("precision:", 100*tp/(tp+fp), "%")

"绘图"
dim1 = len(test_Y)
dim2 = len(train_Y)
fig2 = plt.figure(figsize=(5, 4))

ax1 = fig2.add_subplot(1, 1, 1)  # 1, 1, 1 表示一行一列 第一个
# ax1.set_title('Specific Surface Area test plot')
plt.xlabel('Number of predicted samples')
plt.ylabel('Classification')
x_n1 = range(0, dim1)
# plt.plot(n, test_pre, 'b*')
# plt.plot(n, test_pre, 'r')
plt.ylim(-0.2,3.2)
plt.plot(x_n1, predictions, label="Predicted Values", color='r', marker="*", markersize=4, linewidth=1)
# plt.plot(n, test_labels, 'g*',marker = "+")
# plt.plot(n, test_labels, 'b')
plt.plot(x_n1, test_Y, label="Actual Values", color='lightsteelblue', marker="o", markersize=4, linewidth=1)
plt.legend()  # 显示上面定义的label的内容，否则会报error: No handles with labels found to put in legend.
# plt.ion()

# AUC-ROC
y_pred_proba = model.predict_proba(test_x)
fpr, tpr, thresholds = metrics.roc_curve(test_Y, y_pred_proba[:, 1], pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2  # 线宽
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="AUC = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--", label="Random Guess")
plt.legend()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")  # 图例显示在右下角
plt.savefig('auc_roc.pdf')
# 图中图（）
# plt.axes([0, 0.99, 0.5, 0.5])
# plt.yticks(())
# plt.plot(
#     fpr,
#     tpr,
#     color="darkorange",
#     lw=lw,
#     label="AUC = %0.2f)" % roc_auc,
# )


# 混淆矩阵
C = confusion_matrix(test_Y, predictions)
FP = C.sum(axis=0) - np.diag(C)
FN = C.sum(axis=1) - np.diag(C)
TP = np.diag(C)
TN = C.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

print("FPR:", 100*tp/(tp+fp), "%")

plt.matshow(C, cmap=plt.cm.GnBu)  # 设置颜色

for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

# plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。


plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
# plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
# plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
# plt.yticks(range(0,5), labels=['a','b','c','d','e'])



plt.show()  # 显示图像

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)
