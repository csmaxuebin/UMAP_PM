from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from time import time
import os
import csv

data=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\Data2-coarse.csv')
print  (data)
data = data.values

data_D = data[:,:-1]

data_L = data[:,-1]
t0 = time()
data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.2)
print(label_test)

regr = linear_model.LinearRegression()


regr.fit(data_train, label_train)

data_y_pred = regr.predict(data_test)
print(data_y_pred)

print('Coefficients: \n', regr.coef_)

print("Mean squared error: %.2f"
      % mean_squared_error(label_test, data_y_pred)) #Mean squared error: 0.14

print('Variance score: %.2f' % r2_score(label_test, data_y_pred))
print("done in %0.3fs" % (time() - t0))  # done in 0.032s
write_list1 = ["Liner","%.2f" % mean_squared_error(label_test, data_y_pred),(time() - t0)]
print(write_list1)
os.chdir(r'C:\Users\\28708\\Desktop\\data_result\\output')
with open('file-Adult_SVM.csv', 'a') as fid:
        fid_csv = csv.writer(fid)
        fid_csv.writerow(write_list1)