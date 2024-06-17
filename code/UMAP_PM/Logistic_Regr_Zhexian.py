import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# i = [2,4,6,8,10]
i = ["0.2","0.4","0.6","0.8","1.0"]
"""
######Adult set
Scores=[0.815935323,0.81580984,0.816237148,0.8169,0.8155]
Scores1=[0.74583,0.753379,0.7684,0.7712,0.78]
Scores2=[0.7206,0.73,0.736,0.7473311,0.75]
Scores3=[0.7391,0.74,0.7480,0.75908,0.768]
Scores4=[0.758982863,0.773,0.78,0.7868,0.79458]
"""
"""
#########ssc set
Scores=[0.88125,0.88,0.88125,0.88225,0.88125]
Scores1=[0.79,0.79625,0.806,0.810776,0.82]
Scores2=[0.75,0.7633,0.774,0.7841636,0.7895]
Scores3=[0.775,0.7875,0.79625,0.80175,0.807775]
Scores4=[0.837,0.845,0.853,0.86,0.8677]
"""
# """
#####creditcard set
Scores = [0.797, 0.7964, 0.797, 0.7974, 0.7983]
Scores1 = [0.77,0.773,0.7781,0.7805,0.783]
Scores2 = [0.742307692307693, 0.75, 0.7528, 0.76, 0.7661538462]
Scores3 = [0.755, 0.76, 0.765, 0.768, 0.77]
Scores4 = [0.78,0.7808,0.785,0.788,0.789]
"""
# """
plt.xlabel('ε')
plt.ylabel('Classification Rate')
plt.ylim(0.6,0.9)
# plt.plot(i,Scores,'g.-',i,Scores1,'b.-')
ln1, = plt.plot(i,Scores,'g*',marker='s',markerfacecolor='none',linestyle='--')
ln2, = plt.plot(i,Scores1,'b*',marker='^',markerfacecolor='none',linestyle='--')
ln3, = plt.plot(i,Scores2,'r*',marker='o',markerfacecolor='none',linestyle='--')
ln4, = plt.plot(i,Scores3,'y*',marker='d',markerfacecolor='none',linestyle='--')
ln5, = plt.plot(i,Scores4,'m*',marker='x',linestyle='--')
plt.legend(handles=[ln1,ln2,ln3,ln4,ln5],labels=['Non Privacy','PM','Duchi et al','DPPro','UMAP_PM'],loc='lower right')
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result5\\Adult_LR.svg", dpi=600,format="svg")
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result5\\ssc_LR.svg", dpi=600,format="svg")
plt.savefig(r"C:\Users\\28708\\Desktop\\data_result5\\creditcard_LR.svg", dpi=600,format="svg")
plt.show()
# """
