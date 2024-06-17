import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from time import time
import math
import os
import csv
data=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\ss1acs_int.csv')
print  (data)
def PM_1d(t_i, eps):
    C = (math.exp(eps / 2) + 1) / (math.exp(eps / 2) - 1)
    l_t_i = (C + 1) * t_i / 2 - (C - 1) / 2
    r_t_i = l_t_i + C - 1
    # provide 'size' parameter in uniform() would result in a ndarray
    x = np.random.uniform(0, 1)
    threshold = math.exp(eps / 2) / (math.exp(eps / 2) + 1)
    if x < threshold:
        t_star = np.random.uniform(l_t_i, r_t_i)
    else:
        tmp_l = np.random.uniform(-C, l_t_i)
        tmp_r = np.random.uniform(r_t_i, C)
        w = np.random.randint(2)
        t_star = (1 - w) * tmp_l + w * tmp_r
    # print("PM_1d t-star: %.3f" % t_star)
    return t_star

def PM_md(t_i, eps):
    n_features = len(t_i)
    k = max(1, min(n_features, int(eps / 2.5)))
    rand_features = np.random.randint(0, n_features, size=k)
    res = np.zeros(t_i.shape)
    for j in rand_features:
        res[j] = (n_features * 1.0 / k) * PM_1d(t_i[j], eps / k)
    return res


data = data.values

data_D = data[:,:-1]

data_L = data[:,-1]
t0 = time()
esp=10
data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.2)
train_array = np.empty((0, 29), int)
for i in range(15999):
    PM_train_array=PM_md(np.matrix(list(data_train[i])),esp)
    train_array = np.append(train_array, PM_train_array, axis=0)
# print train_array
test_array = np.empty((0, 29), int)
for i in range(4000):
    PM_test_array=PM_md(np.matrix(list(data_test[i])),esp)
    test_array = np.append(test_array, PM_test_array, axis=0)

regr = linear_model.LinearRegression()


regr.fit(train_array, label_train)

data_y_pred = regr.predict(test_array)


print('Coefficients: \n', regr.coef_)

print("Mean squared error: %.2f"
      % mean_squared_error(label_test, data_y_pred)) #Mean squared error: 0.14

print('Variance score: %.2f' % r2_score(label_test, data_y_pred))
print("done in %0.3fs" % (time() - t0)) # done in 0.032s
mse="%.2f" % mean_squared_error(label_test, data_y_pred)
write_list1 = ["Liner",esp,mse,time() - t0]
print(write_list1)
os.chdir(r'C:\Users\\28708\\Desktop\\data_result\\SS123')
with open('file-ssc_UMAP_PM_SVM.csv', 'a') as fid:
        fid_csv = csv.writer(fid)
        fid_csv.writerow(write_list1)