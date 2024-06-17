from sklearn import random_projection
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
from time import time
import os
import csv
from collections import Counter
def guass_func(eps, delta, sens = 1):            # Compute the variance for a (eps, delta)-DP Gaussian mechanism with sensitivity = sens
    return 2.*np.log(1.25/delta)*sens**2/(eps**2)

def guass_mech(data, sens, eps):
    n_features = len(data)
    k = max(1, n_features)
    rand_features = np.random.randint(0, n_features, size=k)

    for j in rand_features:
        data[j] += guass_func(eps, 10e-5, sens)
    return data

data=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\creditcard_int.csv')
print  (data)
data = data.values

data_D = data[:,:-1]

data_L = data[:,-1]
t0 = time()


transformer = random_projection.GaussianRandomProjection(n_components=3)

ranpro = transformer.fit_transform(data_D)
esp=10
data_train, data_test, label_train, label_test = train_test_split(ranpro,data_L,test_size=0.2)
print(label_test)

print(data_test.shape)
train_array = np.empty((0, 3), int)
for i in range(23999):
    guass_train_array=guass_mech(np.matrix(list(data_train[i])),1,esp)
    train_array = np.append(train_array, guass_train_array, axis=0)
test_array = np.empty((0, 3), int)
for i in range(6000):
    PM_test_array=guass_mech(np.matrix(list(data_test[i])),1,esp)
    test_array = np.append(test_array, PM_test_array, axis=0)
print('OK')

clf = SVC(kernel = 'rbf',gamma=0.1,C=10)
clf.fit(train_array,label_train.astype('int'))
k = clf.get_params()
print('clf.param: ', k)
pred = clf.predict(test_array)
print(pred)
accuracy = metrics.accuracy_score(label_test.astype('int'), pred)*100
print('accuracy: ',accuracy)    # 95.60687234435618
print("done in %0.3fs" % (time() - t0))  # done in 13.843s
total_time=time() - t0
write_list1 = [esp,accuracy,total_time]
print(write_list1)
os.chdir(r'C:\Users\\28708\\Desktop\\data_result\\creditcard')
with open('file-ssc_GUASS_RP_SVM.csv', 'a') as fid:
        fid_csv = csv.writer(fid)
        fid_csv.writerow(write_list1)