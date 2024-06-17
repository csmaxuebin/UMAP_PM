from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
from time import time
data=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\creditcard_int.csv')

print  (data)

data = data.values

data_D = data[:,:-1]

data_L = data[:,-1]

data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.2)

t0 = time()

print('OK')
clf = SVC(kernel = 'rbf',gamma=0.1,C=10)
clf.fit(data_train,label_train.astype('int'))
k = clf.get_params()
print('clf.param: ', k)
pred = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test.astype('int'), pred)*100
print('accuracy: ',accuracy)
print("done in %0.3fs" % (time() - t0))
