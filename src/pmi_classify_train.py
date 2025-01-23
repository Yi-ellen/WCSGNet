import numpy as np
import pandas as pd
import os
import torch
import argparse
import time
from torch.optim.lr_scheduler import ExponentialLR

import torch.nn.functional as F
from sklearn.metrics import precision_score,f1_score, accuracy_score
from torch_geometric.loader import DataLoader
import copy

from model import CSGNet
from datasets_wgcna import MyDataset

pathjoin = os.path.join

def get_parser(parser=None):

    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr', type=str, default='dataset/pre_data/scRNAseq_datasets/Muraro.npz')  # Input file (NPZ format)
    parser.add_argument('-outdir', type=str, default='result/pca_pmi_models')  # Output directory
    parser.add_argument('-netname','--netname',type=str, default='pca_pmi')     # different network construction method
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


def classify_train(args):
    expr_npz = args.expr
    netname = args.netname
    models_folder = args.outdir 
    HVGs_num = args.high_var_genes
    addname = args.addname 
    n_splits=args.n_splits
    cuda_flag = args.cuda
    batch_size = args.batch_size
    
    mid_channel = args.mid_channel
    global_conv1_dim = args.global_conv1_dim
    global_conv2_dim = args.global_conv2_dim

    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]      

    device = torch.device('cuda' if torch.cuda.is_available() and cuda_flag  else 'cpu')

    _, logExpr1 = get_logExpr3(expr_npz) 


    seq_folder = pathjoin('dataset/5fold_data/', base_filename)    
    
    os.makedirs(models_folder,exist_ok=True)
    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')    
    seq_dict = np.load(seq_dict_file, allow_pickle=True) 
    label = seq_dict['label']
    str_labels = seq_dict['str_labels']
    class_num = len(np.unique(str_labels))

    all_filtered_genes_file = pathjoin(seq_folder, f'{base_filename}_filtered_hvgs{HVGs_num}.npy')
    all_filtered_genes_array = np.load(all_filtered_genes_file, allow_pickle=True)
    genes_num_all = [len(row) for row in all_filtered_genes_array]

    train_index_imputed_file = pathjoin(seq_folder, f'{base_filename}_train_index_imputed.npy')
    train_index_imputed = np.load(train_index_imputed_file, allow_pickle=True)



    init_lr =0.01
    max_epoch= args.epoch 
    weight_decay  = 1e-4  
    dropout_ratio = 0.1

    print('use wegithed cross entropy.... ')
    label_type = np.unique(label.reshape(-1))
    alpha = np.array([np.sum(label == x) for x in label_type])
    alpha = np.max(alpha) / alpha
    alpha = np.clip(alpha,1,50)
    alpha = alpha/ np.sum(alpha)
    loss_fn = torch.nn.CrossEntropyLoss(weight = torch.tensor(alpha).float())
    loss_fn = loss_fn.to(device)

    n_splits = args.n_splits

    for k in range(n_splits):
        k_fold = k + 1
        print("k_fold: ", k_fold)
        train_index = seq_dict[f'train_index_{k_fold}'] 
        train_label = label[train_index]

        net_file = os.path.join(seq_folder, netname, f"{netname}_f{k_fold}.tsv")
        print(net_file)
        net_df = pd.read_csv(net_file, sep='\t', header=None, dtype={0: int, 1: int})
        filtered_genes_index = all_filtered_genes_array[k]
        filtered_genes_index = filtered_genes_index.astype(int)


        net_df['gene1'] = net_df[0]  
        net_df['gene2'] = net_df[1]                    
        net_df['weight'] = net_df[2]
        net_df = net_df[['gene1', 'gene2', 'weight']]

        logExpr1_train = logExpr1[np.ix_(train_index, filtered_genes_index)] 
        cell_train_folder = os.path.join(seq_folder, netname, f"train_f{k_fold}")
        os.makedirs(cell_train_folder, exist_ok=True)
        processed_dir = os.path.join(cell_train_folder, 'processed')
        os.makedirs(processed_dir, exist_ok=True)

        train_dataset = MyDataset(root=cell_train_folder, com_net=net_df, expr=logExpr1_train, label=train_label, my_indices=train_index_imputed[k])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4) 

        gene_num = genes_num_all[k]
        channel_list = copy.deepcopy(args.channel_list)
        model = CSGNet(in_channel=1, mid_channel=mid_channel, num_nodes=gene_num, out_channel=class_num, dropout_ratio=dropout_ratio, \
                       channel_list=channel_list, global_conv1_dim=global_conv1_dim, global_conv2_dim=global_conv2_dim
                       ).to(device)
        print(model)

        optimizer = torch.optim.Adam(model.parameters(),lr=init_lr ,weight_decay=weight_decay,)
        scheduler = ExponentialLR(optimizer, gamma=0.8)

        for epoch in range(1, max_epoch):
            train_loss, train_acc, train_f1 = train(model,optimizer,train_loader,epoch,device,loss_fn, scheduler=scheduler)
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print('epoch\t%03d,lr : %.06f,loss: %.06f,Train-acc: %.04f,Train-f1: %.04f'%(
                        epoch,lr,train_loss,train_acc,train_f1))

            print(f'Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()[0]}')
        model_file = pathjoin(models_folder, f'{base_filename}_hvgs{HVGs_num}_model{addname}{k_fold}.pth')
        torch.save(model, model_file)




def train(model,optimizer,train_loader,epoch,device,loss_fn=None,scheduler=None,verbose=False):
    model.train()
    loss_all = 0
    iters = len(train_loader)
    y_pred = []
    y_true = []

    for idx, data in enumerate(train_loader):
        data = data.to(device)
        if verbose:
            print(data.x.shape, data.y.shape, data.edge_index.shape, data.edge_weight.shape)
        optimizer.zero_grad()
        output = model(data)
        if loss_fn is None:
            loss = F.cross_entropy(output, data.y.reshape(-1), weight=None,)
        else:
            loss = loss_fn(output, data.y.reshape(-1))        
        loss.backward()
        optimizer.step()
        
        loss_all += loss.item()
       
        with torch.no_grad():  # Disable gradient calculation for inference
            pred = output.argmax(dim=1).cpu().numpy()  # Predicted labels
            y = data.y.cpu().numpy()  # True labels   

        y = data.y.cpu().data.numpy()
        y_pred.extend(pred) 
        y_true.extend(y) #(64,)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    return loss_all / iters, acc, f1


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
    classify_train(args)
    end_time = time.time()
    print(f"Code run time: {end_time - start_time} s")