from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from time import time
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error #MSE
from sklearn import linear_model
data=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\Data2-coarse.csv')
print  (data)
data = data.values

data_D = data[:,:-1]

data_L = data[:,-1]
t0 = time()
data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.2)
print(label_test)
model = LogisticRegression(solver='lbfgs')
clf = model.fit(data_train, label_train)

# 预测测试数据上的类标签
LR2_pred_labels = model.predict(data_test)
print(LR2_pred_labels)
# accacy=
score = model.score(data_test,label_test)
print('Accuracy Score: ', score)
print('--------------------------------------------------------')
# 查看分类报告来评估模型
print(classification_report(label_test, LR2_pred_labels))

print("done in %0.3fs" % (time() - t0))
