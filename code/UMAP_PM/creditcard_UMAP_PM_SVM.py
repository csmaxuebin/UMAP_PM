from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
from time import time
from umap import UMAP
import math
import os
import csv
from collections import Counter
data=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\creditcard_int.csv')
print  (data)
# """
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


# """
data = data.values

data_D = data[:,:-1]

data_L = data[:,-1]

data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.2)

t0 = time()
reducer2 = UMAP(n_neighbors=15, n_components=3, n_epochs=200,
min_dist=0.1, local_connectivity=2, random_state=42)

print (reducer2)
# X_train_res = reducer2.fit_transform(data_train, label_train)


X_train_umap = reducer2.fit_transform(data_train)
esp=8
train_array = np.empty((0, 3), int)
for i in range(23999):
    PM_train_array=PM_md(np.matrix(list(X_train_umap[i])),esp)
    train_array = np.append(train_array, PM_train_array, axis=0)
print (train_array.shape)
X_test_umap = reducer2.transform(data_test)
# print X_test_pca.shape
test_array = np.empty((0, 3), int)
for i in range(6000):
    PM_test_array=PM_md(np.matrix(list(X_test_umap[i])),esp)
    test_array = np.append(test_array, PM_test_array, axis=0)

print('OK')


clf = SVC(kernel = 'rbf',gamma=0.1,C=10)
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
os.chdir(r'C:\Users\\28708\\Desktop\\data_result\\creditcard')
with open('file-ssc_UMAP_PM_SVM.csv', 'a') as fid:
        fid_csv = csv.writer(fid)
        fid_csv.writerow(write_list1)
