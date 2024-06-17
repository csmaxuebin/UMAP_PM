from sklearn import random_projection
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from time import time
import os
import csv
def guass_func(eps, delta, sens = 1):            # Compute the variance for a (eps, delta)-DP Gaussian mechanism with sensitivity = sens
    return 2.*np.log(1.25/delta)*sens**2/(eps**2)

def guass_mech(data, sens, eps):
    n_features = len(data)
    k = max(1, n_features)
    rand_features = np.random.randint(0, n_features, size=k)

    for j in rand_features:
        # print(res[j])
        data[j] += guass_func(eps, 10e-5, sens)
    return data

data=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\ss1acs_int.csv')
print  (data)
data = data.values

data_D = data[:,:-1]

data_L = data[:,-1]
t0 = time()

transformer = random_projection.SparseRandomProjection(n_components=3)
# print(transformer)
ranpro = transformer.fit_transform(data_D)
# print(ranpro)
data_train, data_test, label_train, label_test = train_test_split(ranpro,data_L,test_size=0.2)
print(data_test.shape)
esp=10
train_array = np.empty((0, 3), int)
for i in range(15999):
    guass_train_array=guass_mech(np.matrix(list(data_train[i])),1,esp)
    train_array = np.append(train_array, guass_train_array, axis=0)

test_array = np.empty((0, 3), int)
for i in range(4000):
    PM_test_array=guass_mech(np.matrix(list(data_test[i])),1,esp)
    test_array = np.append(test_array, PM_test_array, axis=0)
# print(test_array)
print('OK')

regr = linear_model.LinearRegression()


regr.fit(train_array, label_train)

data_y_pred = regr.predict(test_array)


print('Coefficients: \n', regr.coef_)

print("Mean squared error: %.2f"
      % mean_squared_error(label_test, data_y_pred)) #Mean squared error: 0.14
mse= "%.2f" % mean_squared_error(label_test, data_y_pred)
print('Variance score: %.2f' % r2_score(label_test, data_y_pred))
print("done in %0.3fs" % (time() - t0))  # done in 0.032s
write_list1 = ["Liner",esp,mse,time() - t0]
print(write_list1)
os.chdir(r'C:\Users\\28708\\Desktop\\data_result\\SS123')
with open('file-ssc_GUASS_RP_SVM.csv', 'a') as fid:
        fid_csv = csv.writer(fid)
        fid_csv.writerow(write_list1)