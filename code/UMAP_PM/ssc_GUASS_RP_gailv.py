from sklearn import random_projection
from sklearn.model_selection import train_test_split
import pandas as pd
from time import time
from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import Counter
import os
import csv
from sklearn.metrics import classification_report
def guass_func(eps, delta, sens = 1):            # Compute the variance for a (eps, delta)-DP Gaussian mechanism with sensitivity = sens
    return 2.*np.log(1.25/delta)*sens**2/(eps**2)

def guass_mech(data, sens, eps):
    n_features = len(data)
    k = max(1, n_features)
    rand_features = np.random.randint(0, n_features, size=k)

    for j in rand_features:
        data[j] += guass_func(eps, 10e-5, sens)
    return data
def get_avd(pro, true_pro):
    delta_pro = np.array(pro) - np.array(true_pro)
    abs_delta = np.abs(delta_pro)
    return np.sum(abs_delta) / 2.0
def l2_err(pro, true_pro):
    delta_pro = np.array(pro) - np.array(true_pro)
    return 1.0 * np.sqrt(np.sum(np.power(delta_pro, 2)) / (1.0))

data=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\ss1acs_int.csv')
print  (data)
data = data.values

data_D = data[:,:-1]

data_L = data[:,-1]
t0 = time()
eps=0.8
n_component=5
transformer = random_projection.SparseRandomProjection(n_components=n_component)
# print(transformer)
ranpro = transformer.fit_transform(data_D)
# print(ranpro)
data_train, data_test, label_train, label_test = train_test_split(ranpro,data_L,test_size=0.2)
print(data_test.shape)
train_array = np.empty((0, n_component), int)
for i in range(15999):
    guass_train_array=guass_mech(np.matrix(list(data_train[i])),1,eps)
    train_array = np.append(train_array, guass_train_array, axis=0)
test_array = np.empty((0, n_component), int)
for i in range(4000):
    PM_test_array=guass_mech(np.matrix(list(data_test[i])),1,eps)
    test_array = np.append(test_array, PM_test_array, axis=0)
# print(test_array)
print('OK')
result = Counter(label_test)
print(result)
result_list=list(result.values())
print(result_list)
gailv=result_list[0]/sum(result_list)
print (gailv)
model = LogisticRegression(solver='lbfgs')
clf = model.fit(train_array, label_train)

# 预测测试数据上的类标签
LR2_pred_labels = model.predict(test_array)

print(LR2_pred_labels)
LR2_sum=Counter(LR2_pred_labels)
LR2_sum_list=list(LR2_sum.values())
print(LR2_sum_list)
LR2_gailv=LR2_sum_list[0]/sum(LR2_sum_list)
avd=get_avd(gailv,LR2_gailv)
l2_error=l2_err(gailv,LR2_gailv)
print (LR2_gailv)
print('AVD',avd)
print('l2_error',l2_error)
# accacy=
score = model.score(test_array,label_test)
print('Accuracy Score: ', score)
print('--------------------------------------------------------')
# 查看分类报告来评估模型
print(classification_report(label_test, LR2_pred_labels))

print("done in %0.3fs" % (time() - t0))
write_list1 = [eps,n_component,score,avd,l2_error,(time() - t0)]
print(write_list1)
os.chdir(r'C:\Users\\28708\\Desktop\\data_result\\SS123')
with open('file-ssc_GUASS_RP_gailv.csv', 'a') as fid:
        fid_csv = csv.writer(fid)
        fid_csv.writerow(write_list1)


