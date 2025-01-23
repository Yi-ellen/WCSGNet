import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
pathjoin = os.path.join
import os

from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'


seq_dict_file = '../../dataset/5fold_data/Baron_Human/seq_dict.npz'
seq_dict = np.load(seq_dict_file, allow_pickle=True) 
label = seq_dict['label']
str_labels = seq_dict['str_labels']
barcodes = seq_dict['barcode']

genes = seq_dict['gene_symbol']

all_filtered_genes_file = '../../dataset/5fold_data/Baron_Human/Baron_Human_filtered_hvgs2000.npy'

all_filtered_genes_array = np.load(all_filtered_genes_file, allow_pickle=True)
filtered_genes_index = all_filtered_genes_array[0]
filtered_genes_index = filtered_genes_index.astype(int)
print(filtered_genes_index.shape)

gene_hvgs = genes[filtered_genes_index]

print(gene_hvgs)


pred_path = '../../result/preds/Baron_Human_a0.01_hvgs2000_prediction.h5'

cell_embedding = pd.read_hdf(pred_path, key='embedding')

cell_type = pd.read_hdf(pred_path, key='cell_type')
pred_prob = pd.read_hdf(pred_path, key='pred_prob')
print(cell_embedding.shape)
print(cell_type.shape)
print(cell_type[:5])

lst = [2]
for k in lst:
    k_fold = k + 1
    print("k_fold: ", k_fold)
    test_index = seq_dict[f'test_index_{k_fold}']
    barcodes_test = barcodes[test_index]
    cur_cell_emb = cell_embedding.loc[barcodes_test]

    print("Embedding shape:", cur_cell_emb.shape)  
    print("Number of unique labels:", len(np.unique(cell_type['true_cell_type'].loc[barcodes_test])))

    perp = 30
    ee = 12
    tsne = TSNE(n_components=2,      
                perplexity=perp,       
                early_exaggeration=ee,
                random_state=42,     
                n_iter=1000,         
                learning_rate='auto')
    
    embeddings_2d = tsne.fit_transform(cur_cell_emb.to_numpy())

    
   
    plt.figure(figsize=(8, 5))
    unique_labels = np.unique(cell_type['true_cell_type'].loc[barcodes_test])
    colors = sns.color_palette('husl', n_colors=len(unique_labels))
    color_dict = dict(zip(unique_labels, colors))
    for label in unique_labels:
        mask = cell_type['true_cell_type'].loc[barcodes_test] == label
        plt.scatter(embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[color_dict[label]],
                label=str_labels[label],
                alpha=0.6,
                s=20)

    plt.legend(bbox_to_anchor=(1.01, 1),
                loc='upper left',
                borderaxespad=0,
                fontsize=12)

    plt.title(f'Baron Human/True_label', fontsize=14)    
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneA_true.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneA_true.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')

    plt.show()

# Figure 6B
    plt.figure(figsize=(8, 5))
    unique_labels = np.unique(cell_type['true_cell_type'].loc[barcodes_test])
    colors = sns.color_palette('husl', n_colors=len(unique_labels))
    color_dict = dict(zip(unique_labels, colors))

    for label in unique_labels:
        mask = cell_type['pred_cell_type'].loc[barcodes_test] == label
        plt.scatter(embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[color_dict[label]],
                label=str_labels[label],
                alpha=0.6,
                s=20)
        
    plt.title(f'Baron Human/Pred_label', fontsize=16)
    plt.legend(bbox_to_anchor=(1.01, 1),
                loc='upper left',
                borderaxespad=0,
                fontsize=12)

    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'../../result/Figures/Figure_tsneB_pred.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneB_pred.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')

    plt.show()    



# Figure 6E

    degree_file = pathjoin('data/Baron_Human/', f'degree_matrix_Baron_Human_per_fold_a0.01_hvgs2000.npz')
    degree_fold = np.load(degree_file, allow_pickle=True)
    degree_f3 = degree_fold['CV_3']
    print(degree_f3.shape)
    print(degree_f3[:5, :5])

    print(gene_hvgs[344])
    target_gene_degree = degree_f3[344]
    gene_degrees = target_gene_degree


    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(embeddings_2d[:, 0], 
                        embeddings_2d[:, 1],
                        c=gene_degrees,  
                        cmap='viridis',  
                        alpha=0.6,
                        s=20)

    cbar = plt.colorbar(scatter)

    cbar.set_label('Gene Degree', rotation=270, labelpad=15,
                   fontsize=14)

    plt.title(f'Gene (CCDC157) Degree', fontsize=14)
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneE.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneE.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')


    plt.show()



# Figure 6G
    print(gene_hvgs[1168])
    target_gene_degree = degree_f3[1168]
    
    gene_degrees = target_gene_degree
    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(embeddings_2d[:, 0], 
                        embeddings_2d[:, 1],
                        c=gene_degrees,  
                        cmap='viridis',  
                        alpha=0.6,
                        s=20)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Gene Degree', rotation=270, labelpad=15,
                   fontsize=14)
    plt.title(f'Gene (KEL) Degree', 
                fontsize=14)
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneG.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneG.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')

    plt.show()



# Figure 6D
    print(gene_hvgs[959])
    target_gene_degree = degree_f3[959]
    
    gene_degrees = target_gene_degree

    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(embeddings_2d[:, 0], 
                        embeddings_2d[:, 1],
                        c=gene_degrees,  
                        cmap='viridis',  
                        alpha=0.6,
                        s=20)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Gene Degree', rotation=270, labelpad=15,
                   fontsize=14)
    plt.title(f'Gene (GPR4) Degree', 
                fontsize=14)
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneD.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneD.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')

    plt.show()



# Figure 6C


    print(gene_hvgs[38])
    target_gene_degree = degree_f3[38]

    gene_degrees = target_gene_degree

    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(embeddings_2d[:, 0], 
                        embeddings_2d[:, 1],
                        c=gene_degrees,  
                        cmap='viridis',  
                        alpha=0.6,
                        s=20)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Gene Degree', rotation=270, labelpad=15,
                   fontsize=14)
    plt.title(f'Gene (ADAMTS17) Degree', 
                fontsize=14)
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneC.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneC.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')

    plt.show()



# Figure 6F

    print(gene_hvgs[607])
    target_gene_degree = degree_f3[607]
    
    gene_degrees = target_gene_degree

    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(embeddings_2d[:, 0], 
                        embeddings_2d[:, 1],
                        c=gene_degrees,  
                        cmap='viridis',  
                        alpha=0.6,
                        s=20)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Gene Degree', rotation=270, labelpad=15,
                   fontsize=14)
    plt.title(f'Gene (DMWD) Degree', 
                fontsize=14)
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneF.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneF.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')
    plt.show()


# Figure 6H

    print(gene_hvgs[837])
    target_gene_degree = degree_f3[837]
    
    gene_degrees = target_gene_degree

    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(embeddings_2d[:, 0], 
                        embeddings_2d[:, 1],
                        c=gene_degrees, 
                        cmap='viridis',  
                        alpha=0.6,
                        s=20)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Gene Degree', rotation=270, labelpad=15,
                   fontsize=14)
    plt.title(f'Gene (FZD5) Degree', 
                fontsize=14)
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneH.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneH.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')

    plt.show()



# edge weight analysis (T-SNE)
# Figure 6K
    
    fold_df = pd.read_hdf('data/Baron_Human/edge_weight_matrices_Baron_Human_per_fold_a0.01_hvgs2000.h5', 
                        key=f'CV_{k_fold}')

    target_edge_degree = fold_df['344-1441']
    print(gene_hvgs[344])
    print(gene_hvgs[1441])

    edge_w = target_edge_degree


    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(embeddings_2d[:, 0], 
                        embeddings_2d[:, 1],
                        c=edge_w,  
                        cmap='viridis',  
                        alpha=0.6,
                        s=20)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Edge weight', rotation=270, labelpad=15,
                   fontsize=14)

    plt.title(f'Gene Pair (CCDC157-NOTUM) Weight', 
                fontsize=14)
    
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneK.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneK.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')

    plt.show()


# Figure 6L
    target_edge_degree = fold_df['426-1825']
    print(gene_hvgs[426])
    print(gene_hvgs[1825])
    edge_w = target_edge_degree

    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(embeddings_2d[:, 0], 
                        embeddings_2d[:, 1],
                        c=edge_w,  
                        cmap='viridis',  
                        alpha=0.6,
                        s=20)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Edge weight', rotation=270, labelpad=15,
                   fontsize=14)

    plt.title(f'Gene Pair (CEP19-SAP18) Weight', 
                fontsize=14)
    
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneL.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneL.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')
    plt.show()


# Figure 6I
    target_edge_degree = fold_df['1129-1908']
    print(gene_hvgs[1129])
    print(gene_hvgs[1908])
    edge_w = target_edge_degree

    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(embeddings_2d[:, 0], 
                        embeddings_2d[:, 1],
                        c=edge_w,  
                        cmap='viridis',  
                        alpha=0.6,
                        s=20)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Edge weight', rotation=270, labelpad=15,
                   fontsize=14)

    plt.title(f'Gene Pair (IL21R-SLC35F5) Weight', 
                fontsize=14)
    
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneI.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneI.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')

    plt.show()



# Figure 6N

    target_edge_degree = fold_df['1114-1880']
    print(gene_hvgs[1114])
    print(gene_hvgs[1880])
    edge_w = target_edge_degree

    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(embeddings_2d[:, 0], 
                        embeddings_2d[:, 1],
                        c=edge_w, 
                        cmap='viridis',  
                        alpha=0.6,
                        s=20)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Edge weight', rotation=270, labelpad=15,
                   fontsize=14)

    plt.title(f'Gene Pair (IGF2BP3-SIAH2) Weight', 
                fontsize=14)
    
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneN.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneN.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')

    plt.show()





# Figure 6M

    target_edge_degree = fold_df['209-313']
    print(gene_hvgs[209])
    print(gene_hvgs[313])

    edge_w = target_edge_degree

    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(embeddings_2d[:, 0], 
                        embeddings_2d[:, 1],
                        c=edge_w,  
                        cmap='viridis',  
                        alpha=0.6,
                        s=20)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Edge weight', rotation=270, labelpad=15,
                   fontsize=14)

    plt.title(f'Gene Pair (BTBD10-CARD8) Weight', 
                fontsize=14)
    
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneM.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneM.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')

    plt.show()




# Figure 6J
    target_edge_degree = fold_df['909-948']
    print(gene_hvgs[909])
    print(gene_hvgs[948])
    edge_w = target_edge_degree
    plt.figure(figsize=(7, 5))


    scatter = plt.scatter(embeddings_2d[:, 0], 
                        embeddings_2d[:, 1],
                        c=edge_w,  
                        cmap='viridis',  
                        alpha=0.6,
                        s=20)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Edge weight', rotation=270, labelpad=15,
                   fontsize=14)

    plt.title(f'Gene Pair (GLRB-GPATCH11) Weight', 
                fontsize=14)
    
    plt.xlabel('t-SNE 1', fontsize=14,fontweight='bold')
    plt.ylabel('t-SNE 2', fontsize=14,fontweight='bold')

    plt.tight_layout()

    plt.savefig(f'../../result/Figures/Figure_tsneJ.svg', 
                dpi=1200, 
                bbox_inches='tight',
                format='svg')

    plt.savefig(f'../../result/Figures/Figure_tsneJ.png', 
                dpi=1200, 
                bbox_inches='tight',
                format='png')

    plt.show()

