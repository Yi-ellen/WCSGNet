import numpy as np
import os
import torch
import argparse
import time

pathjoin = os.path.join

# Function to define command line arguments for the script
def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr',type=str, default='dataset/pre_data/scRNAseq_datasets/Baron_Human.npz') 
    parser.add_argument('-outdir', type=str, default='dataset/5fold_data/')
    parser.add_argument('-hvgs','--high_var_genes',type=int,default=2000)
    parser.add_argument('-ca', '--csn_alpha',type=float, default='0.01')

    return parser


# Function to load data and compute the degree matrix for each cell type
def get_matrix(args):
    expr_npz = args.expr
    save_folder = args.outdir
    csn_alpha = args.csn_alpha
    HVGs_num = args.high_var_genes

    # Extract the base filename from the provided dataset path (e.g., 'Baron_Human')
    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]      

    # Load the seq_dict file for 5-fold cross-validation
    seq_folder = pathjoin(save_folder, base_filename)

    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')    
    seq_dict = np.load(seq_dict_file, allow_pickle=True) 
    label = seq_dict['label']
    str_labels = seq_dict['str_labels']
    print("cell type: ", str_labels)

    cur_label = 0
    matrix_dict = {}
    matrix_dict['str_labels'] = str_labels

    # Iterate through each cell type (str_labels)
    for cell_type in str_labels:
        print("cur_label: ", cur_label)
        print("cell_type: ", cell_type)

        # Dictionary to store degree matrices for each fold
        degree_matrices_for_folds = {}  
        
        # Iterate through each fold (k=0 to 4) in the 5-fold cross-validation
        for k in range(5):
            k_fold = k + 1
            test_index = seq_dict[f'test_index_{k_fold}']
            label_test = label[test_index]
            
            # Find cells with the current label (cur_label)
            cur_label_idxs = np.where(label_test == cur_label)[0].tolist()

            # Define the path to the test data for the current fold
            cell_test_folder = pathjoin(seq_folder, f"wcsn_a{csn_alpha}_hvgs{HVGs_num}", f"test_f{k_fold}", 'processed')
            
            degree_matrix = []

            # Process each cell with the current label
            for idx in cur_label_idxs:
                data = torch.load(os.path.join(cell_test_folder, f'cell_{idx}.pt'))
                
                # Extract edge index from the data
                edge_index = data.edge_index
                # edge_weight = data.edge_weight  # Edge weight is not used, commented out
                
                # Generate symmetric edge indices
                row, col = edge_index
                symmetric_edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
                
                # Calculate the degree (number of connections) for each gene
                degrees = torch.bincount(symmetric_edge_index[0])
                
                # If the number of gene nodes is less than the maximum (2000), pad with zeros
                if degrees.size(0) < 2000:
                    degrees = torch.cat([degrees, torch.zeros(2000 - degrees.size(0))])

                # Add the degree vector for the current cell to the degree matrix
                degree_matrix.append(degrees.numpy())

            # Transpose the degree matrix and save it to the fold's dictionary
            degree_matrix = np.array(degree_matrix).T
            print(degree_matrix.shape)
            degree_matrices_for_folds[f'CV_{k_fold}'] = degree_matrix  # Save the degree matrix for each fold

        # Store the degree matrices for the current label in the matrix_dict
        matrix_dict[str(cur_label)] = degree_matrices_for_folds
        cur_label += 1

    # Define the output file path for the degree matrix
    degree_file = pathjoin(seq_folder, f'degree_matrix_{base_filename}_a{csn_alpha}_hvgs{HVGs_num}.npz')

    # Check the structure of matrix_dict before saving
    print(f"Matrix dict structure before saving: {type(matrix_dict)}")
    for key in matrix_dict:
        print(f"Key: {key}, Type: {type(matrix_dict[key])}")
        if isinstance(matrix_dict[key], dict):  # Ensure it's a dictionary
            for inner_key in matrix_dict[key]:
                print(f" Inner Key: {inner_key}, Type: {type(matrix_dict[key][inner_key])}")

    # Save the degree matrix dictionary to a .npz file
    np.savez(degree_file, **matrix_dict)


# Main entry point of the script
if __name__ == '__main__':
    start_time = time.time()  
    parser = get_parser()  # Parse command line arguments
    args = parser.parse_args()
    print('args:', args)

    # Get the degree matrix for the specified dataset
    get_matrix(args)

    end_time = time.time()
    print(f"Code run time: {end_time - start_time} s")
