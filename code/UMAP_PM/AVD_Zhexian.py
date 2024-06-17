import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei']
labels = ["0.2","0.4","0.6","0.8","1.0"]
# """
######Adult set
y1=[0.13,0.123383085,0.121448314,0.12,0.118]
y2=[0.123,0.12073,0.11736,0.113,0.1075]
y3=[0.121420674,0.118,0.1105527916,0.1050359315,0.098148148]
y4=[0.12,0.116141515,0.100552792,0.098,0.081205086]
# """
"""
#########ssc set
y1=[0.256,0.25,0.242,0.23,0.22]
y2=[0.234,0.231293073,0.228,0.21285,0.20128]
y3=[0.21267125,0.21,0.205527916,0.2,0.19]
y4=[0.19956,0.193,0.1839,0.1742,0.16]
"""
"""
#########creditcard set
y1=[0.143125,0.143,0.14,0.132625550359315,0.12]
y2=[0.129,0.1273,0.1236,0.12,0.11291075]
y3=[0.115,0.113,0.1110833,0.109833333,0.1075]
y4=[0.11,0.109,0.1072,0.1058,0.105]
"""
# """
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots()
"""
rects1 = ax.bar(x - width*2, y1, width, label='Duchi et al',color='w',edgecolor='black',hatch="///",lw=.8)
rects2 = ax.bar(x - width+0.01, y2, width, label='DPPro',color='w',edgecolor='black',hatch="---",lw=.8)
rects3 = ax.bar(x + 0.02, y3, width, label='PM',color='grey',edgecolor='black',lw=.8)
rects4 = ax.bar(x + width+ 0.03, y4, width, label='UMAP_PM',color='w',edgecolor='black',hatch=" ",lw=.8)
"""
rects1 = ax.bar(x - width*2, y1, width, label='Duchi et al',color='#015699',lw=.8)
rects2 = ax.bar(x - width+0.01, y2, width, label='DPPro',color='#fac00f',lw=.8)
rects3 = ax.bar(x + 0.02, y3, width, label='PM',color='#4f596d',lw=.8)
rects4 = ax.bar(x + width+ 0.03, y4, width, label='UMAP_PM',color='#5fc6c9',lw=.8)

plt.ylim(0,0.2)
ax.set_ylabel('AVD', fontsize=16)
ax.set_xlabel('ε', fontsize=16)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):

    for rect in rects:
        height = rect.get_height()


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()
plt.savefig(r"C:\Users\\28708\\Desktop\\data_result6\\Adult_AVD1.svg", dpi=600,format="svg")
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result6\\SSC_AVD1.svg", dpi=600,format="svg")
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result6\\creditcard_AVD1.svg", dpi=600,format="svg")
plt.show()
