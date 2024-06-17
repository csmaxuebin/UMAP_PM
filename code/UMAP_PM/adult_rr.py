import numpy as np
from sklearn import random_projection
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import PCA
import pandas as pd
from time import time
import math

data=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\Data2-coarse.csv')
print  (data)
data = data.values

data_D = data[:,:-1]

data_L = data[:,-1]
t0 = time()

transformer = random_projection.SparseRandomProjection(n_components=3)
print(transformer)
ranpro = transformer.fit_transform(data_D)
print(ranpro)
data_train, data_test, label_train, label_test = train_test_split(ranpro,data_L,test_size=0.5)


print('OK')

clf = SVC(kernel = 'rbf',gamma=0.1,C=18)
clf.fit(data_train,label_train.astype('int'))
k = clf.get_params()
print('clf.param: ', k)
pred = clf.predict(data_test)

accuracy = metrics.accuracy_score(label_test.astype('int'), pred)*100
print('accuracy: ',accuracy)    # 95.60687234435618
print("done in %0.3fs" % (time() - t0))  # done in 13.843s