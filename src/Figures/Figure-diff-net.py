import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'

WCSGNet = [0.768, 0.865, 0.703, 0.978, 0.966, 0.993, 1.000, 0.927, 0.972]  
WCSGNet_wgcna = [0.824,	0.881, 	0.709, 	0.977, 	0.963, 	0.987, 	0.994, 	0.925, 	0.975]   
WCSGNet_pca_pmi = [0.826,	0.883, 	0.720, 	0.979, 	0.956, 	0.983, 	0.994, 	0.930, 	0.975]  
WCSGNet_grnboost2 = [0.832,	0.876, 	0.708, 	0.980, 	0.965, 	0.988, 	1.000, 	0.920, 	0.972]

datasets = ["Zhang T", "Kang", "Zheng 68k", "Baron Human", "Muraro", "Segerstolpe", "AMB", "TM", "Baron Mouse"]

x = np.arange(len(datasets))  
width = 0.2  

offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

plt.figure(figsize=(8, 4))

plt.bar(x + offsets[0], WCSGNet, width, label='WCSGNet', color='#72B6A1', edgecolor='white', linewidth=1)
plt.bar(x + offsets[1], WCSGNet_wgcna, width, label='WCSGNet(WGCNA)', color='#E99675', edgecolor='white', linewidth=1)
plt.bar(x + offsets[2], WCSGNet_pca_pmi, width, label='WCSGNet(PCA-PMI)', color='#95A3C3', edgecolor='white', linewidth=1)
plt.bar(x + offsets[3], WCSGNet_grnboost2, width, label='WCSGNet(GRNBoost2)', color='#F3C678', edgecolor='white', linewidth=1)

plt.ylabel('Mean F1', fontsize=10, fontweight="bold")
plt.xticks(x, datasets, fontsize=10, rotation=45)  
plt.ylim(0.6, 1.0)  

ax = plt.gca()
ax.spines['top'].set_visible(False) 
ax.spines['right'].set_visible(False)  

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), fontsize=10, ncol=4, frameon=False)

def add_value_labels(ax):
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',  
                    xy=(bar.get_x() + bar.get_width() / 2, height),  
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8,  
                    rotation=90)  

add_value_labels(ax)
plt.tight_layout()

plt.savefig('../../result/Figures/Figure_net1.png', format='png', dpi=1200, bbox_inches='tight')
plt.savefig('../../result/Figures/Figure_net1.svg', format='svg', dpi=1200, bbox_inches='tight')
plt.show()




# Accuracy

WCSGNet = [0.822, 	0.939, 	0.765, 	0.987, 	0.973, 	0.994, 	1.000, 	0.957, 	0.981]  
WCSGNet_wgcna = [0.859,	0.945, 	0.779, 	0.989, 	0.972, 	0.989, 	1.000, 	0.964, 	0.981]  
WCSGNet_pca_pmi = [0.859, 0.945, 0.798,	0.988, 	0.970, 	0.991, 	1.000,	0.964, 0.983]  
WCSGNet_grnboost2 = [0.866,	0.940, 	0.779, 	0.988, 	0.975, 	0.994, 	1.000, 	0.961, 	0.983]

datasets = ["Zhang T", "Kang", "Zheng 68k", "Baron Human", "Muraro", "Segerstolpe", "AMB", "TM", "Baron Mouse"]

x = np.arange(len(datasets))  
width = 0.2  

offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

plt.figure(figsize=(8, 4))

plt.bar(x + offsets[0], WCSGNet, width, label='WCSGNet', color='#72B6A1', edgecolor='white', linewidth=1)
plt.bar(x + offsets[1], WCSGNet_wgcna, width, label='WCSGNet(WGCNA)', color='#E99675', edgecolor='white', linewidth=1)
plt.bar(x + offsets[2], WCSGNet_pca_pmi, width, label='WCSGNet(PCA-PMI)', color='#95A3C3', edgecolor='white', linewidth=1)
plt.bar(x + offsets[3], WCSGNet_grnboost2, width, label='WCSGNet(GRNBoost2)', color='#F3C678', edgecolor='white', linewidth=1)

plt.ylabel('Accuracy', fontsize=10, fontweight="bold")
plt.xticks(x, datasets, fontsize=10, rotation=45)  
plt.ylim(0.6, 1.0) 

ax = plt.gca()
ax.spines['top'].set_visible(False) 
ax.spines['right'].set_visible(False)  

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), fontsize=10, ncol=4, frameon=False)

def add_value_labels(ax):
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',  
                    xy=(bar.get_x() + bar.get_width() / 2, height),  
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8,  
                    rotation=90) 

add_value_labels(ax)

plt.tight_layout()

plt.savefig('../../result/Figures/Figure_net2.png', format='png', dpi=1200, bbox_inches='tight')
plt.savefig('../../result/Figures/Figure_net2.svg', format='svg', dpi=1200, bbox_inches='tight')

plt.show()
