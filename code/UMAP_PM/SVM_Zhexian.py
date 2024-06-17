import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
i = ["0.2","0.4","0.6","0.8","1.0"]
# """
######Adult set
Scores=[0.82155,0.82,0.81917,0.82169,0.82155]
Scores1=[0.7791,0.7915,0.80,0.80908,0.8123]
Scores2=[0.7206,0.74,0.76,0.773311,0.79]
Scores3=[0.76583,0.77379,0.77684,0.78512,0.79665]
Scores4=[0.791,0.8006,0.81227,0.81348,0.819458]
# """
"""
#########ssc set
Scores=[0.78425,0.785,0.78425,0.785,0.78405]
Scores1=[0.75,0.7625,0.77,0.776,0.78025]
Scores2=[0.69,0.70,0.70975,0.7195,0.73175]
Scores3=[0.7,0.7233,0.73,0.73636,0.7412]
Scores4=[0.768,0.77,0.776,0.78,0.783]
"""
"""
#####creditcard set
Scores = [0.793, 0.7944, 0.79230769, 0.7944, 0.79230769]
Scores1 = [0.77,0.773,0.7781,0.7805,0.783]
Scores2 = [0.742307692307693, 0.755, 0.7528, 0.76, 0.7661538462]
Scores3 = [0.755, 0.76, 0.765, 0.768, 0.769]
Scores4 = [0.778,0.78,0.785,0.788,0.789]
"""
# """
plt.xlabel('ε')
plt.ylabel('Classification Rate')
plt.ylim(0.5,0.85)
# plt.plot(i,Scores,'g.-',i,Scores1,'b.-')
ln1, = plt.plot(i,Scores,'g*',marker='s',markerfacecolor='none',linestyle='--')
ln2, = plt.plot(i,Scores1,'b*',marker='^',markerfacecolor='none',linestyle='--')
ln3, = plt.plot(i,Scores2,'r*',marker='o',markerfacecolor='none',linestyle='--')
ln4, = plt.plot(i,Scores3,'y*',marker='d',markerfacecolor='none',linestyle='--')
ln5, = plt.plot(i,Scores4,'m*',marker='x',linestyle='--')
plt.legend(handles=[ln1,ln2,ln3,ln4,ln5],labels=['Non Privacy','PM','Duchi et al','DPPro','UMAP_PM'],loc='lower right')
plt.savefig(r"C:\Users\\28708\\Desktop\\data_result5\\Adult_SVM.svg", dpi=600,format="svg")
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result5\\ssc_SVM.svg", dpi=600,format="svg")
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result5\\creditcard_SVM.svg", dpi=600,format="svg")
plt.show()
# """
