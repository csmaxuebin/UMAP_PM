import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
i = [0.5,1,1.5,2,2.5,3]
"""
######Adult set

Scores1=[0.127791,0.126,0.125,0.12275,0.122,0.120583333]
Scores2=[0.117,0.115666667,0.1111,0.11,0.1057,0.102875]
Scores3=[0.103,0.102,0.099,0.0973,0.085,0.083]
Scores4=[0.0895,0.0849,0.08225,0.07333333,0.07075,0.068333333]
"""

# """

#####creditcard set
Scores1 = [0.145,0.143,0.1408,0.13558,0.13,0.128]
Scores2 = [0.129,0.1285,0.127,0.1264,0.1258,0.124]
Scores3 = [0.1135,0.113,0.1127,0.1116,0.11075,0.11]
Scores4 = [0.109666667,0.1085,0.107875,0.10625,0.105833333,0.0985]
# """
# """
plt.xlabel('Number of users ($\mathregular{10^4}$)')
plt.ylabel('Average Variation Distance')
plt.ylim(0,0.25)
# plt.plot(i,Scores,'g.-',i,Scores1,'b.-')

ln2, = plt.plot(i,Scores1,'b*',marker='^',markerfacecolor='none',linestyle='--')
ln3, = plt.plot(i,Scores2,'r*',marker='o',markerfacecolor='none',linestyle='--')
ln4, = plt.plot(i,Scores3,'y*',marker='d',markerfacecolor='none',linestyle='--')
ln5, = plt.plot(i,Scores4,'m*',marker='x',linestyle='--')
plt.legend(handles=[ln2,ln3,ln4,ln5],labels=['Duchi et al','DPPro','PM','UMAP_PM'],loc='lower right')
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result1\\Adult_number_AVD1.svg", dpi=600,format="svg")
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result1\\SSC_dimensional_AVD.svg", dpi=600,format="svg")
plt.savefig(r"C:\Users\\28708\\Desktop\\data_result1\\creditcard_number_AVD1.svg", dpi=600,format="svg")
plt.show()
# """
