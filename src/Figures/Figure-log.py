import numpy as np
import gc

import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'

scRNA_datasets = ['Muraro', 'Baron_Mouse', 'Segerstolpe', 'Baron_Human', 'Zhang_T', 'Kang_ctrl', 'AMB', 'TM', 'Zheng68K']

for base_filename in scRNA_datasets:
    print(base_filename)

    seq_dict = np.load(f'../../dataset/5fold_data/{base_filename}/seq_dict.npz', allow_pickle=True) 

    for k in range(5):
        k_fold = k + 1
        print("k_fold: ", k_fold)

        train_index = seq_dict[f'train_index_{k_fold}']
        print("Number of train_f1: ", len(train_index))

        wcsn_folder = os.path.join(
            f"../../dataset/5fold_data/{base_filename}/{base_filename}_a0.01_hvgs2000", 
            f"train_f{k_fold}", 
            'processed'
        )

        all_edge_weights = np.array([], dtype=np.float32)
        all_edge_weights_LWT = np.array([], dtype=np.float32)
        for idx in range(len(train_index)):
            file_path = os.path.join(wcsn_folder, f'cell_{idx}.pt')
            if os.path.exists(file_path):
                data = torch.load(file_path)
                edge_weight = data.edge_weight.numpy().astype(np.float32)  
                edge_weight_LWT = np.log1p(data.edge_weight.numpy()).astype(np.float32)
                all_edge_weights = np.concatenate((all_edge_weights, edge_weight))
                all_edge_weights_LWT = np.concatenate((all_edge_weights_LWT, edge_weight_LWT))
                del data, edge_weight, edge_weight_LWT
                gc.collect()

            else:
                print(f"File not found: {file_path}")

        print(f"Total number of edges: {len(all_edge_weights)}")

        plt.figure(figsize=(4, 3))
        plt.hist(all_edge_weights, bins=50, density=True, alpha=0.7, label='Edge Weight Distribution')
        plt.legend(fontsize=10)
        plt.xlabel("Edge Weight")
        plt.ylabel("Density")
        plt.title("WCSN", fontsize=10)
        plt.legend()
        plt.savefig(f'../../result/Figures/edge_log/{base_filename}_f{k_fold}_Weight_Distribution.svg', 
                    dpi=1200, bbox_inches='tight',format='svg')  
        plt.savefig(f'../../result/Figures/edge_log/{base_filename}_f{k_fold}_Weight_Distribution.png', 
                    dpi=1200, bbox_inches='tight',format='png')  

        plt.figure(figsize=(4, 3))
        plt.hist(all_edge_weights_LWT, bins=50, density=True, alpha=0.7, label='Edge Weight Distribution')
        plt.legend(fontsize=10)
        plt.xlabel("Edge Weight", fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.title("WCSN(Logarithmic Transformation)", fontsize=10)
        plt.legend()
        plt.savefig(f'../../result/Figures/edge_log/{base_filename}_f{k_fold}_Log_Weight_Distribution.svg', 
                    dpi=1200, bbox_inches='tight',format='svg')  # Specify SVG format
        plt.savefig(f'../../result/Figures/edge_log/{base_filename}_f{k_fold}_Log_Weight_Distribution.png', 
                    dpi=1200, bbox_inches='tight',format='png')
