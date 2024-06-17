import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
from time import time
import math
from umap import UMAP
import os
import csv
from collections import Counter
data=pd.read_csv(r'C:\Users\\28708\\Desktop\\data_file\\creditcard_int.csv')
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
    return t_star

def PM_md(t_i, eps):
    n_features = len(t_i)
    k = max(1, min(n_features, int(eps / 2.5)))
    rand_features = np.random.randint(0, n_features, size=k)
    res = np.zeros(t_i.shape)
    for j in rand_features:
        res[j] = (n_features * 1.0 / k) * PM_1d(t_i[j], eps / k)
    return res
def get_avd(pro, true_pro):
    delta_pro = np.array(pro) - np.array(true_pro)
    abs_delta = np.abs(delta_pro)
    return np.sum(abs_delta) / 2.0
def l2_err(pro, true_pro):
    delta_pro = np.array(pro) - np.array(true_pro)
    return 1.0 * np.sqrt(np.sum(np.power(delta_pro, 2)) / (1.0))

data = data.values

data_D = data[:,:-1]

data_L = data[:,-1]
t0 = time()
data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.2)
reducer2 = UMAP(n_neighbors=15, n_components=3, n_epochs=200,
min_dist=0.1, local_connectivity=2, random_state=42,
)
result = Counter(label_test)
result_list=list(result.values())
print(result_list)
gailv=result_list[0]/sum(result_list)
print (gailv)
print (reducer2)
X_train_res = reducer2.fit_transform(data_train)
# X_train_res = reducer2.fit_transform(data_train,label_train)


# X_train_umap = reducer2.transform(data_train)
eps=10
train_array = np.empty((0, 3), int)
for i in range(23999):
    PM_train_array=PM_md(np.matrix(list(X_train_res[i])),eps)
    train_array = np.append(train_array, PM_train_array, axis=0)
print (train_array.shape)
X_test_umap = reducer2.transform(data_test)

test_array = np.empty((0, 3), int)
for i in range(6000):
    PM_test_array=PM_md(np.matrix(list(X_test_umap[i])),eps)
    test_array = np.append(test_array, PM_test_array, axis=0)

print('OK')

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

score = model.score(test_array,label_test)
print('Accuracy Score: ', score)
print('--------------------------------------------------------')
# 查看分类报告来评估模型
print(classification_report(label_test, LR2_pred_labels))

print("done in %0.3fs" % (time() - t0))
write_list1 = [eps,score,avd,l2_error,(time() - t0)]
print(write_list1)
os.chdir(r'C:\Users\\28708\\Desktop\\data_result\\creditcard')
with open('file-ssc_UMAP_PM_gailv.csv', 'a') as fid:
        fid_csv = csv.writer(fid)
        fid_csv.writerow(write_list1)
