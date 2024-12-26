import numpy as np
import pandas as pd
import os
import torch
from scipy.stats import norm
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from scipy.sparse import load_npz
from torch_geometric.data import Data
import argparse
import time
import scanpy as sc

from joblib import Parallel, delayed
from multiprocessing import shared_memory
from numba import jit


pathjoin = os.path.join

# Function to parse command line arguments
def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr', type=str, default='dataset/pre_data/scRNAseq_datasets/Muraro.npz')  # Input file (NPZ format)
    parser.add_argument('-outdir', type=str, default='dataset/5fold_data/')  # Output directory
    parser.add_argument('-cuda', '--cuda', type=bool, default=True)  # Use CUDA (GPU) for computation
    parser.add_argument('-hvgs', '--high_var_genes', type=int, default=2000)  # Number of high variance genes
    parser.add_argument('-ca', '--csn_alpha', type=float, default=0.01)  # Significance level for WCSN construction
    parser.add_argument('-nsp', '--n_splits', type=int, default=5)  # Number of splits for cross-validation
    return parser


# Function to perform WCSN construction
def csn_constr_test(args):
    expr_npz = args.expr
    save_folder = args.outdir
    csn_alpha = args.csn_alpha
    HVGs_num = args.high_var_genes
    n_splits = args.n_splits

    # Process the expression file name, e.g., Muraro
    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]

    # Read sequence dictionary for cross-validation
    seq_folder = pathjoin(save_folder, base_filename)
    csn_data_folder = pathjoin(seq_folder, f"WCSN_a{csn_alpha}_hvgs{HVGs_num}")
    
    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')    
    seq_dict = np.load(seq_dict_file, allow_pickle=True) 
    label = seq_dict['label']

    # Load log-transformed expression data
    logExpr0, logExpr1 = get_logExpr(expr_npz) 
    # Load filtered high variance genes
    all_filtered_genes_file = pathjoin(seq_folder, f'{base_filename}_filtered_hvgs{HVGs_num}.npy')
    all_filtered_genes_array = np.load(all_filtered_genes_file, allow_pickle=True)

    # Loop over the splits for cross-validation
    for k in range(n_splits):
        k_fold = k + 1
        print("test k_fold: ", k_fold)
        test_index = seq_dict[f'test_index_{k_fold}']
        train_index = seq_dict[f'train_index_{k_fold}'] 
        label_test = label[test_index]       

        # Get filtered genes for this fold
        filtered_genes_index = all_filtered_genes_array[k]
        filtered_genes_index = filtered_genes_index.astype(int)
        
        # Subset the expression data for testing
        logExpr0_train = logExpr0[np.ix_(train_index, filtered_genes_index)]
        logExpr0_test = logExpr0[np.ix_(test_index, filtered_genes_index)]
        logExpr1_test = logExpr1[np.ix_(test_index, filtered_genes_index)]      

        cell_test_folder = os.path.join(csn_data_folder, f"test_f{k_fold}")
        # Perform CSN construction and save results for test data
        com_csn_constr_and_save_test6(logExpr0_train, logExpr0_test, logExpr1_test, label_test, alpha=csn_alpha, save_folder=cell_test_folder) 
        
        print(f"test fold{k_fold} CSN construction is completed!")


# Function to load log-transformed expression data
def get_logExpr(expr_npz):
    """
    Load and process log-transformed expression data from the NPZ file.

    Args:
        expr_npz (str): Path to the expression NPZ file.

    Returns:
        tuple: logExpr0 and logExpr1 arrays (processed gene expression data).
    """
    data = np.load(expr_npz, allow_pickle=True)

    # Read the raw count expression data
    countExpr = data['count']  # Shape: (cells, genes)
    print("raw (cells, genes): ", countExpr.shape)

    # Normalize and log-transform the expression data
    row_sums = countExpr.sum(axis=1, keepdims=True)
    normalized_data = 1e6 * countExpr / row_sums
    normalized_data = normalized_data.astype(np.float32)
    
    # Log-transform the normalized expression data
    logExpr0 = np.log1p(normalized_data)  # logExpr0: (cells, genes)
    logExpr1 = np.log1p(normalized_data + 1e-5)  # logExpr1: (cells, genes), slightly adjusted for non-zero features

    return logExpr0, logExpr1


# Function to construct WCSN and save results for test data
def com_csn_constr_and_save_test6(logExpr_train, logExpr_test, logExpr1_test, label, c=None, alpha=0.01, boxsize=0.1, save_folder='dataset/5fold_data/Muraro/WCSN_a0.01_hvgs2000/test_f1/'):
    """
    Perform WCSN construction for test data and save the results.

    Args:
        logExpr_train (np.ndarray): Log-transformed expression data for training cells (cells x genes).
        logExpr_test (np.ndarray): Log-transformed expression data for test cells (cells x genes).
        logExpr1_test (np.ndarray): Adjusted log-transformed expression data for test cells (cells x genes).
        label (np.ndarray): Labels for the cells.
        alpha (float): Significance level for WCSN thresholding.
        boxsize (float): Box size for WCSN calculation.
        save_folder (str): Folder to save the processed WCSN results.
    """
    print("logExpr_train.shape", logExpr_train.shape)
    logExpr_train = logExpr_train.transpose()
    logExpr_test = logExpr_test.transpose()
    n1, n2 = logExpr_train.shape  # (gene_num, cell_num)
    print("n1:", n1)
    print("n2:", n2)
    os.makedirs(save_folder, exist_ok=True)
    processed_dir = os.path.join(save_folder, 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    p = -norm.ppf(alpha)  # alpha = 0.01
    print(p)
    
    # Sort the training data
    sorted_data = np.sort(logExpr_train, axis=1)  # Sort each row
    
    # Count unique values and their occurrences for each gene across cells
    unique_counts = [np.unique(row, return_counts=True) for row in logExpr_train]
    count_dicts = [{value: count for value, count in zip(unique, counts)} for unique, counts in unique_counts]

    data = torch.from_numpy(logExpr_train).cuda()

    # Loop through test cells
    for cur_test_index in range(logExpr_test.shape[1]):
        print(f"cell {cur_test_index}: ")
        test_cell = logExpr_test[:, cur_test_index]

        # Find the insertion indices for test cell values
        insert_indices = np.array([np.searchsorted(sorted_data[i], test_cell[i]) for i in range(len(test_cell))])
        
        # Initialize upper and lower bounds for each gene
        upper_col = np.zeros(n1)  
        lower_col = np.zeros(n1)  

        # Compute upper and lower bounds for CSN construction
        for i in range(n1):
            s1 = sorted_data[i]
            n3 = n2 - np.sum(np.sign(s1))
            h = round(boxsize / 2 * np.sum(np.sign(s1)))
            k = insert_indices[i]

            if k < n2 and test_cell[i] == s1[k]:
                s = count_dicts[i].get(s1[k], 0)
                if s == 0:
                    print("Error!")

                if s >= h:
                    upper_col[i] = s1[k]
                    lower_col[i] = s1[k]
                else:
                    upper_col[i] = s1[min(n2-1, k+s+h)]
                    lower_col[i] = s1[max(int(n3 * (n3 > h)), k-h)]                
            else:
                upper_col[i] = s1[min(n2-1, k+h)]
                lower_col[i] = s1[min(max(int(n3 * (n3 > h)), k-h), n2 - 1)]                  

        # Transfer data to GPU for computation
        test_cell = torch.from_numpy(test_cell).cuda()
        upper_col = torch.from_numpy(upper_col).cuda()
        lower_col = torch.from_numpy(lower_col).cuda()

        B = ((data <= upper_col.unsqueeze(1)) & 
            (data >= lower_col.unsqueeze(1)) & 
            (test_cell.unsqueeze(1) > 0)).float()

        # Calculate the score for the test cell
        a = torch.sum(B, dim=1)
        score = (B @ B.T * n2 - torch.outer(a, a)) / torch.sqrt(torch.outer(a, a) * torch.outer(n2 - a, n2 - a) / (n2 - 1) + torch.finfo(torch.float32).eps)
        score[score < p] = 0  # Set scores below threshold to zero
        
        # Create upper triangular sparse matrix
        score = torch.triu(score, diagonal=1)
        indices = torch.nonzero(score, as_tuple=False).t().contiguous()
        values = score[indices[0], indices[1]]
        sparse_score = torch.sparse_coo_tensor(indices, values, score.shape).coalesce().to(torch.float32)
        print("score: ", len(values))

        # Save the WCSN result
        sparse_score = sparse_score.coalesce()
        sparse_score = sparse_score.cpu()     

        save_data(logExpr1_test[cur_test_index], sparse_score, label[cur_test_index], cur_test_index, processed_dir)

    print('WCSN construction for test set is completed!')


# Function to save CSN data
def save_data(logExpr_i, com_csn, label_i, i, save_folder):
    """
    Save the CSN results as a PyTorch Data object.

    Args:
        logExpr_i (np.ndarray): Gene expression data for a single cell.
        com_csn (torch.sparse.Tensor or coo_matrix): The CSN matrix for the cell.
        label_i (int): Label for the cell.
        i (int): Cell index.
        save_folder (str): Directory to save the data.
    """
    x = logExpr_i.reshape(-1, 1).astype(np.float32)
    x = torch.from_numpy(x)

    if isinstance(com_csn, coo_matrix):
        indices = np.array([com_csn.row, com_csn.col], dtype=np.int32)
        edge_index_torch = torch.tensor(indices, dtype=torch.int32)
        edge_weight = torch.tensor(com_csn.data, dtype=torch.float32)
    elif isinstance(com_csn, torch.sparse.Tensor):
        graph = com_csn.coalesce()
        edge_index_torch = graph.indices().to(torch.int32)
        edge_weight = graph.values().to(torch.float32)
    else:
        raise TypeError("Unsupported com_csn type: {}".format(type(com_csn)))

    y = torch.tensor([label_i], dtype=torch.long)
    cell_data = Data(x=x, edge_index=edge_index_torch, edge_weight=edge_weight, y=y)
    
    save_path = os.path.join(save_folder, f'cell_{i}.pt')
    torch.save(cell_data, save_path)
    print(f'Cell {i} data saved to {save_path}')


# Main execution block
if __name__ == '__main__':
    start_time = time.time()  
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)
    
    csn_constr_test(args)
    
    end_time = time.time()
    print(f"Code execution time: {end_time - start_time} seconds")
