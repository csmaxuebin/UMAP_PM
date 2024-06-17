from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
from time import time
import math
from scipy.special import comb
import csv
import os
from sklearn.preprocessing import MinMaxScaler
data=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\Data2-coarse.csv')
print  (data)
# """
def Duchi_1d(t_i, eps, t_star):
    p = (math.exp(eps) - 1) / (2 * math.exp(eps) + 2) * t_i + 0.5
    coin = np.random.binomial(1, p)
    if coin == 1:
        return t_star[1]
    else:
        return t_star[0]

def Duchi_md(t_i, eps):
    n_features = len(t_i)
    if n_features % 2 != 0:
        C_d = pow(2, n_features - 1) / comb(n_features - 1, (n_features - 1) / 2)
    else:
        C_d = (pow(2, n_features - 1) + 0.5 * comb(n_features, n_features / 2)) / comb(n_features - 1, n_features / 2)

    B = C_d * (math.exp(eps) + 1) / (math.exp(eps) - 1)
    v = []
    for tmp in t_i:
        tmp_p = 0.5 + 0.5 * tmp
        tmp_q = 0.5 - 0.5 * tmp
        v.append(np.random.choice([1, -1], p=[tmp_p, tmp_q]))
    bernoulli_p = math.exp(eps) / (math.exp(eps) + 1)
    coin = np.random.binomial(1, bernoulli_p)

    t_star = np.random.choice([-B, B], len(t_i), p=[0.5, 0.5])
    v_times_t_star = np.multiply(v, t_star)
    sum_v_times_t_star = np.sum(v_times_t_star)
    if coin == 1:
        while sum_v_times_t_star <= 0:
            t_star = np.random.choice([-B, B], len(t_i), p=[0.5, 0.5])
            v_times_t_star = np.multiply(v, t_star)
            sum_v_times_t_star = np.sum(v_times_t_star)
    else:
        while sum_v_times_t_star > 0:
            t_star = np.random.choice([-B, B], len(t_i), p=[0.5, 0.5])
            v_times_t_star = np.multiply(v, t_star)
            sum_v_times_t_star = np.sum(v_times_t_star)
    return t_star.reshape(-1)


# """
data = data.values

data_D = data[:,:-1]

data_L = data[:,-1]

scaler = MinMaxScaler(feature_range=(0, 1))
data_D = scaler.fit_transform(data_D)
data_L = scaler.fit_transform(data_L.reshape(-1,1))
data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.2)

t0 = time()

# train_array = np.empty((0, 14), int)
train_array = []
esp=10
# print(train_array.shape)
for i in range(36176):
    PM_train_array=Duchi_md(np.array(list(data_train[i])),esp)

    train_array = np.append(train_array, PM_train_array, axis=0)
train_array=train_array.reshape(36176,14)
print(train_array)
print (train_array.shape)

# test_array = np.zeros(14)
test_array =[]
for i in range(9045):
    PM_test_array=Duchi_md(np.array(list(data_test[i])),esp)
    test_array = np.append(test_array, PM_test_array, axis=0)
test_array=test_array.reshape(9045,14)
print(test_array)
print (test_array.shape)
print('OK')
clf = SVC(kernel = 'rbf',gamma=0.1,C=18)
clf.fit(train_array,label_train.astype('int'))
k = clf.get_params()
print('clf.param: ', k)
pred = clf.predict(test_array)
accuracy = metrics.accuracy_score(label_test.astype('int'), pred)*100
print('accuracy: ',accuracy)
print("done in %0.3fs" % (time() - t0))

total_time=time() - t0
write_list1 = [esp,accuracy,total_time]
print(write_list1)
os.chdir(r'C:\Users\\28708\\Desktop\\data_result\\output')
with open('file-Adult_DUQI_SVM.csv', 'a') as fid:
        fid_csv = csv.writer(fid)
        fid_csv.writerow(write_list1)