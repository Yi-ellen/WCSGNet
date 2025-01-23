import numpy as np
import pandas as pd
import os
import torch
import argparse
import time
import scanpy as sc

import torch.nn.functional as F
from sklearn.metrics import precision_score,f1_score, accuracy_score
from torch_geometric.loader import DataLoader

from model import CSGNet
from datasets_wgcna import MyDataset

pathjoin = os.path.join


def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr', type=str, default='dataset/pre_data/scRNAseq_datasets/Muraro.npz')  # Input file (NPZ format)
    parser.add_argument('-outdir', type=str, default='result/')  # Output directory
    parser.add_argument('-netname','--netname',type=str, default='wgcna')   # different network construction method
    parser.add_argument('-cuda', '--cuda', type=bool, default=True)  # Use CUDA (GPU) for computation
    parser.add_argument('-bs', '--batch_size', type=int, default=32)  # Batch size for training
    parser.add_argument('-epoch', type=int, default=30)  # Number of epochs
    parser.add_argument('-hvgs', '--high_var_genes', type=int, default=2000)  # Number of high variance genes
    parser.add_argument('-addname', type=str, default="")  # Additional name for model saving
    parser.add_argument('-nsp', '--n_splits', type=int, default=5)  # Number of splits for cross-validation
    parser.add_argument('-clist', '--channel_list', nargs='+', type=int, default=[256, 64], help='Model parameter list')  
    parser.add_argument('-mc', '--mid_channel', type=int, default=16)  # Mid channel dimension
    parser.add_argument('-gcdim1', '--global_conv1_dim', type=int, default=12)  # First global convolution dimension
    parser.add_argument('-gcdim2', '--global_conv2_dim', type=int, default=4)  # Second global convolution dimension
    
    return parser



def classify_test(args):
    expr_npz = args.expr
    netname = args.netname
    save_folder = args.outdir
    HVGs_num = args.high_var_genes  
    addname = args.addname 
    n_splits=args.n_splits

    cuda_flag = args.cuda
    batch_size = args.batch_size

    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]      

    _, logExpr1 = get_logExpr3(expr_npz) 

    device = torch.device('cuda' if torch.cuda.is_available() and cuda_flag  else 'cpu')
    seq_folder = pathjoin('dataset/5fold_data/', base_filename)
    models_folder = pathjoin(seq_folder, 'wgcna_models')
    os.makedirs(pathjoin(save_folder, 'wgcna_preds'),exist_ok=True)
    preds_folder = pathjoin(save_folder, 'wgcna_preds')

    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')    
    seq_dict = np.load(seq_dict_file, allow_pickle=True) 
    label = seq_dict['label']
    str_labels = seq_dict['str_labels']
    barcodes = seq_dict['barcode']

    all_filtered_genes_file = pathjoin(seq_folder, f'{base_filename}_filtered_hvgs{HVGs_num}.npy')

    all_filtered_genes_array = np.load(all_filtered_genes_file, allow_pickle=True)

    pred_probability = []
    cell_type_all = []
    cell_embedding= []

    n_splits = args.n_splits
    for k in range(n_splits):
        k_fold = k + 1
        print("k_fold: ", k_fold)
        test_index = seq_dict[f'test_index_{k_fold}']
        barcodes_test = barcodes[test_index]
        label_test = label[test_index]
        
        filtered_genes_index = all_filtered_genes_array[k]
        filtered_genes_index = filtered_genes_index.astype(int)

        net_file = os.path.join(seq_folder, netname, f"{netname}_f{k_fold}.tsv")
        net_df = pd.read_csv(net_file, sep='\t', header=None, dtype={0: int, 1: int})

        net_df['gene1'] = net_df[0]  
        net_df['gene2'] = net_df[1]     
        net_df['weight'] = net_df[2]
        net_df = net_df[['gene1', 'gene2', 'weight']]

        logExpr1_test = logExpr1[np.ix_(test_index, filtered_genes_index)] 

        cell_test_folder = os.path.join(seq_folder, netname, f"test_f{k_fold}")
        os.makedirs(cell_test_folder, exist_ok=True)
        processed_dir = os.path.join(cell_test_folder, 'processed')
        os.makedirs(processed_dir, exist_ok=True)

        test_dataset = MyDataset(root=cell_test_folder, com_net=net_df, expr=logExpr1_test, label=label_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        model_file = pathjoin(models_folder, f'{base_filename}_hvgs{HVGs_num}_model{addname}{k_fold}.pth')

        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        model = torch.load(model_file)
        model.to(device)


        test_acc, test_f1, curr_y_out, curr_y_pred, cell_latent = test(model, test_loader, device, predicts=True, latent=True) 
        print('Acc: %.03f, F1: %.03f' %(test_acc, test_f1))   
        cell_latent = pd.DataFrame(cell_latent, index=barcodes_test) 
        cell_embedding.append(cell_latent)   
        curr_y_out = pd.DataFrame(curr_y_out, index=barcodes_test) 
        pred_probability.append(curr_y_out)
        sub_test_cell_type = pd.DataFrame(index=barcodes_test)
        sub_test_cell_type["pred_cell_type"] = curr_y_pred
        sub_test_cell_type["true_cell_type"] = label_test
        cell_type_all.append(sub_test_cell_type)     

    pred_probability_df = pd.concat(pred_probability, ignore_index=False)  
    cell_type_all_df = pd.concat(cell_type_all, ignore_index=False) 
    cell_embedding_df = pd.concat(cell_embedding, ignore_index=False) 

    label_true = cell_type_all_df['true_cell_type'].to_numpy()
    label_pred = cell_type_all_df["pred_cell_type"].to_numpy()
    test_acc = accuracy_score(label_true, label_pred)
    test_f1 = f1_score(label_true, label_pred, average='macro')
    test_f1_all = f1_score(label_true, label_pred, average=None)
    print('Final Acc: %.03f, F1: %.03f'%(test_acc, test_f1))   


    pred_save_file = pathjoin(preds_folder, f'{base_filename}_hvgs{HVGs_num}{addname}_prediction.h5')
    cell_type_all_df.to_hdf(pred_save_file, key='cell_type', mode='a')
    pred_probability_df.to_hdf(pred_save_file, key='pred_prob', mode='a')
    cell_embedding_df.to_hdf(pred_save_file, key='embedding', mode='a')


    f1_all = pd.DataFrame(index=str_labels)
    f1_all['F1'] = test_f1_all
    f1_all.loc['acc'] = [test_acc]
    f1_all.loc['macro_f1'] = [test_f1]
    f1_file = pathjoin(preds_folder, f'{base_filename}_hvgs{HVGs_num}{addname}_F1.csv')
    print(f1_all)
    f1_all.to_csv(f1_file, index=True)   



def test(model, loader, device, predicts=False, latent=False):
    model.eval()
    y_pred = []
    y_true = []
    y_out = []
    cell_latent = []
    for data in loader:
        data = data.to(device)
        if latent:
            latent_varaible, y_output = model(data, get_latent_varaible=True)
            latent_var = latent_varaible.cpu().data.numpy()
            cell_latent.append(latent_var)
        else:
            y_output = model(data)
        y_softmax = F.softmax(y_output, dim=1).cpu().detach().numpy() # Convert to probabilities
        y_out.extend(y_softmax)
        
        pred = y_output.argmax(dim=1).cpu().numpy()
        y = data.y.cpu().data.numpy()
        y_pred.extend(pred) 
        y_true.extend(y) #(64,)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    if predicts:
        if latent:
            cell_latent = np.vstack(cell_latent)
            return acc, f1, y_out, y_pred, cell_latent
        else:
            return acc, f1, y_out, y_pred
    else:
        if latent:
            cell_latent = np.vstack(cell_latent)
            return acc, f1, cell_latent
        else:
            return acc, f1


def get_logExpr3(expr_npz):
    data = np.load(expr_npz, allow_pickle=True)
    countExpr = data['count']  # count: row-cell, column-gene
    print("raw (cells, genes): ", countExpr.shape)

    row_sums = countExpr.sum(axis=1, keepdims=True)
    normalized_data = 1e6 * countExpr / row_sums
    normalized_data = normalized_data.astype(np.float32)

    logExpr0 = np.log1p(normalized_data)  
    logExpr1 = np.log1p(normalized_data + 1e-5) 
    return logExpr0, logExpr1


if __name__ == '__main__':
    start_time = time.time()  
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)
    classify_test(args)

    end_time = time.time()
    print(f"Code run time: {end_time - start_time} s")

