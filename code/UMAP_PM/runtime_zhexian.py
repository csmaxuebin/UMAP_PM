import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei']
labels = ["0.2","0.4","0.6","0.8","1.0"]
"""
######Adult set

y1 = [354.57,313.777,339,340,345]
y2 = [286.81,286,290,287.9,286.81]
y3=[212.926,213.05,220.06,196.39,220.06]
y4=[187.537,177.59,185,180.93,183.25]
y5=[104,110,101.641,96.223,97.756]
"""
"""
######ssc set

y1 = [176.6646979,160.419014,163.8354175,159.0538311,160.8597488]

y2=[162.8077393,156.0538311,155.8354175,153.257683,149.7099497]
y3=[144.7081757,142.257683,139.1780696,141.694916,138.7783296]

y4 = [125.514,125,123.8354175,123.0538311,122.8597488]
y5=[69.78652835,69.27713299,69.20211864,69.756,70.18108964]
"""
# """
######creditcard set

y1 = [361.418,362,360.8354175,362.0538311,360.8597488]
y2 = [352.6646979,350.419014,350.8354175,349.0538311,350.8597488]
y3=  [342.6646979,340.419014,340.8354175,343.0538311,339.8597488]
y4=[247.7081757,248.257683,247.1780696,241.694916,248.7783296]
y5=[103.78652835,105.27713299,104.20211864,105.756,106.18108964]
# """
# """
x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots()
# color='#015699',color='#fac00f',color='#4f596d',color='#5fc6c9',color='#f3764a'
"""
rects2 = ax.bar(x - width*2, y1, width, label='PM',color='grey',edgecolor='black',lw=.8)
rects1 = ax.bar(x - width+0.01, y2, width, label='Duchi et al',color='w',hatch="///",edgecolor='black',lw=.8)
rects3 = ax.bar(x + 0.02, y3, width, label='Non-private',color='w',hatch="...",edgecolor='black',lw=.8)
rects4 = ax.bar(x + width+ 0.03, y4, width, label='DPPro',color='w',hatch="---",edgecolor='black',lw=.8)
rects5 = ax.bar(x + width*2 + 0.04, y5, width, label='UMAP_PM',color='w',hatch=" ",edgecolor='black',lw=.8)
"""
rects2 = ax.bar(x - width*2, y1, width, label='PM',color='#015699',lw=.8)
rects1 = ax.bar(x - width+0.01, y2, width, label='Duchi et al',color='#fac00f',lw=.8)
rects3 = ax.bar(x + 0.02, y3, width, label='Non-private',color='#4f596d',lw=.8)
rects4 = ax.bar(x + width+ 0.03, y4, width, label='DPPro',color='#5fc6c9',lw=.8)
rects5 = ax.bar(x + width*2 + 0.04, y5, width, label='UMAP_PM',color='#f3764a',lw=.8)

plt.ylim(0,500)
ax.set_ylabel('Running Time(s)', fontsize=16)
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
autolabel(rects5)

fig.tight_layout()
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result6\\Adult.svg", dpi=600,format="svg")
# plt.savefig(r"C:\Users\\28708\\Desktop\\data_result6\\SSC.svg", dpi=600,format="svg")
plt.savefig(r"C:\Users\\28708\\Desktop\\data_result6\\creditcard.svg", dpi=600,format="svg")
plt.show()