import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
i = ["0.2","0.4","0.6","0.8","1.0"]
"""
######Adult set
Scores=[0.138,0.1375,0.138,0.1369,0.138]
Scores1=[0.1768,0.1675908,0.16480,0.16,0.15]
Scores2=[0.18578,0.1837712,0.1807684,0.1753379,0.174583]
Scores3=[0.18,0.17473311,0.1736,0.1673,0.167206]
Scores4=[0.1679458,0.1638,0.158,0.148,0.143]
"""
"""
#########ssc set
Scores=[0.125,0.1245,0.1245,0.125,0.1245]
Scores1=[0.20778,0.20,0.190769,0.18625,0.18]
Scores2=[0.2147775,0.21175,0.2035,0.2005675,0.195375]
Scores3=[0.21075,0.2074,0.20,0.1973633,0.192]
Scores4=[0.20,0.19,0.18475,0.178085,0.17]
"""
# """
#####creditcard set
Scores = [0.15, 0.1503, 0.15, 0.1514, 0.15]
Scores1 = [0.171,0.168,0.164781,0.1605,0.1583]
Scores2 = [0.1755, 0.173, 0.172, 0.171, 0.1677]
Scores3 = [0.1737692307693, 0.170, 0.167528, 0.163, 0.161538462]
Scores4 = [0.1678,0.16308,0.160785,0.155788,0.152]
# """
# """
plt.xlabel('f')
plt.ylabel('MSE')
plt.ylim(0,0.2)
# plt.plot(i,Scores,'g.-',i,Scores1,'b.-')
ln1, = plt.plot(i,Scores,'g*',marker='s',markerfacecolor='none',linestyle='--')
ln2, = plt.plot(i,Scores1,'b*',marker='^',markerfacecolor='none',linestyle='--')
ln3, = plt.plot(i,Scores2,'r*',marker='o',markerfacecolor='none',linestyle='--')
ln4, = plt.plot(i,Scores3,'y*',marker='d',markerfacecolor='none',linestyle='--')
ln5, = plt.plot(i,Scores4,'m*',marker='x',linestyle='--')
plt.legend(handles=[ln1,ln2,ln3,ln4,ln5],labels=['Non Privacy','PM','Duchi et al','DPPro','UMAP_PM'],loc='lower right')
plt.savefig(r"C:\Users\\28708\\Desktop\\data_result1\\creditcard_Line_mse.svg", dpi=600,format="svg")
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result1\\ssc_Line_mse.svg", dpi=600,format="svg")
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result1\\Adult_Line_mse.svg", dpi=600,format="svg")
plt.show()
# """
