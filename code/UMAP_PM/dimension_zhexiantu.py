import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
######Adult set
Scores=[0.82155,0.82,0.81917,0.82169,0.82155]
Scores1=[0.7791,0.7915,0.80,0.80908,0.8123]
Scores2=[0.7206,0.74,0.76,0.773311,0.79]
Scores3=[0.76583,0.77379,0.77684,0.78512,0.79665]
Scores4=[0.791,0.8006,0.81227,0.81348,0.819458]
"""
# """
#########ssc set
i = [4,8,12,16,20]
Scores1=[0.178025,0.2026,0.21,0.217,0.223]
Scores2=[0.2,0.21625,0.22077,0.2370776,0.24025]
Scores3=[0.2207625,0.23149,0.2385525,0.24326,0.253371125]
Scores4=[0.16,0.17,0.1829,0.192,0.198]
# """
"""
i = [4,8,12,16,20]
#####creditcard set
Scores1 = [0.105,0.1097,0.11,0.11185,0.112]
Scores2 = [0.119,0.120625,0.124077,0.125380776,0.129]
Scores3 = [0.124,0.137326,0.139,0.1459149,0.15]
Scores4 = [0.081583333,0.092,0.10,0.1085,0.11]
# """
# """
plt.xlabel('Dimension')
plt.ylabel('Average Variation Distance')
plt.ylim(0,0.35)
# plt.plot(i,Scores,'g.-',i,Scores1,'b.-')

ln2, = plt.plot(i,Scores1,'b*',marker='^',markerfacecolor='none',linestyle='--')
ln3, = plt.plot(i,Scores2,'r*',marker='o',markerfacecolor='none',linestyle='--')
ln4, = plt.plot(i,Scores3,'y*',marker='d',markerfacecolor='none',linestyle='--')
ln5, = plt.plot(i,Scores4,'m*',marker='x',linestyle='--')
plt.legend(handles=[ln2,ln3,ln4,ln5],labels=['PM','DPPro','Duchi et al','UMAP_PM'],loc='lower right')
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result\\Adult_SVM.svg", dpi=600,format="svg")
plt.savefig(r"C:\Users\\28708\\Desktop\\data_result1\\SSC_dimensional_AVD1.svg", dpi=600,format="svg")
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result1\\creditcard_dimensional_AVD1.svg", dpi=600,format="svg")
plt.show()
# """
