from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn import metrics

import pandas as pd
from umap import UMAP
from time import time
data=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\Data2-coarse.csv')
print  (data)

data = data.values

data_D = data[:,:-1]

data_L = data[:,-1]

data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.5)
t0 = time()
reducer2 = UMAP(n_neighbors=15, n_components=3, n_epochs=200,
min_dist=0.5, local_connectivity=2, random_state=42,
)

# pca = PCA(n_components=0.95, svd_solver='full', whiten=True).fit(data_D)
print (reducer2)
X_train_res = reducer2.fit_transform(data_train, label_train)

X_train_umap = reducer2.transform(data_train)
X_test_umap = reducer2.transform(data_test)
print('OK')


clf = SVC(kernel = 'rbf',gamma=0.1,C=18)
clf.fit(X_train_umap,label_train.astype('int'))
k = clf.get_params()
print('clf.param: ', k)
pred = clf.predict(X_test_umap)

accuracy = metrics.accuracy_score(label_test.astype('int'), pred)*100
print('accuracy: ',accuracy)    # 95.60687234435618
print("done in %0.3fs" % (time() - t0))  # done in 13.843s