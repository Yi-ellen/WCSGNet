import numpy as np
import os
import torch
import argparse
import time
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.loader import DataLoader
import copy

from model import CSGNet
from datasets_wcsn import MyDataset

pathjoin = os.path.join


# Function to parse command line arguments
def get_parser(parser=None):

    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr', type=str, default='dataset/pre_data/scRNAseq_datasets/Muraro.npz')  # Input file (NPZ format)
    parser.add_argument('-outdir', type=str, default='result/models')  # Output directory
    parser.add_argument('-cuda', '--cuda', type=bool, default=True)  # Use CUDA (GPU) for computation
    parser.add_argument('-bs', '--batch_size', type=int, default=32)  # Batch size for training
    parser.add_argument('-epoch', type=int, default=30)  # Number of epochs
    parser.add_argument('-hvgs', '--high_var_genes', type=int, default=2000)  # Number of high variance genes
    parser.add_argument('-ca', '--csn_alpha', type=float, default=0.01)   # Significance level for WCSN construction
    parser.add_argument('-addname', type=str, default="")  # Additional name for model saving
    parser.add_argument('-nsp', '--n_splits', type=int, default=5)  # Number of splits for cross-validation
    parser.add_argument('-clist', '--channel_list', nargs='+', type=int, default=[256, 64], help='Model parameter list')  
    parser.add_argument('-mc', '--mid_channel', type=int, default=16)  # Mid channel dimension
    parser.add_argument('-gcdim1', '--global_conv1_dim', type=int, default=12)  # First global convolution dimension
    parser.add_argument('-gcdim2', '--global_conv2_dim', type=int, default=4)  # Second global convolution dimension
    
    return parser


def classify_train(args):
    expr_npz = args.expr
    models_folder = args.outdir
    csn_alpha = args.csn_alpha
    HVGs_num = args.high_var_genes
    addname = args.addname 
    n_splits = args.n_splits
    cuda_flag = args.cuda
    batch_size = args.batch_size
    mid_channel = args.mid_channel
    global_conv1_dim = args.global_conv1_dim
    global_conv2_dim = args.global_conv2_dim

    # Process file names for input expression and network files
    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]      

    device = torch.device('cuda' if torch.cuda.is_available() and cuda_flag else 'cpu')

    # Load the 5-fold split data and prepare directories
    seq_folder = pathjoin('dataset/5fold_data/', base_filename)
    csn_data_folder = pathjoin(seq_folder, f"wcsn_a{csn_alpha}_hvgs{HVGs_num}")

    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')    
    seq_dict = np.load(seq_dict_file, allow_pickle=True) 
    label = seq_dict['label']
    str_labels = seq_dict['str_labels']
    class_num = len(np.unique(str_labels))

    # Load filtered high-variance genes
    all_filtered_genes_file = pathjoin(seq_folder, f'{base_filename}_filtered_hvgs{HVGs_num}.npy')
    all_filtered_genes_array = np.load(all_filtered_genes_file, allow_pickle=True)
    genes_num_all = [len(row) for row in all_filtered_genes_array]

    # Load training indices
    train_index_imputed_file = pathjoin(seq_folder, f'{base_filename}_train_index_imputed.npy')
    train_index_imputed = np.load(train_index_imputed_file, allow_pickle=True)

    # Initialize hyperparameters for model training
    init_lr = 0.01
    max_epoch = args.epoch 
    weight_decay = 1e-4  
    dropout_ratio = 0.1

    # Weighting cross-entropy loss based on class imbalance
    label_type = np.unique(label.reshape(-1))
    alpha = np.array([np.sum(label == x) for x in label_type])
    alpha = np.max(alpha) / alpha
    alpha = np.clip(alpha, 1, 50)
    alpha = alpha / np.sum(alpha)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(alpha).float()).to(device)

    # Cross-validation loop
    for k in range(n_splits):
        k_fold = k + 1
        print("k_fold:", k_fold)

        # Prepare the training dataset for this fold
        cell_train_folder = os.path.join(csn_data_folder, f"train_f{k_fold}")
        
        # Prepare the training dataset and loader, get WCSN
        train_dataset = MyDataset(root=cell_train_folder, my_indices=train_index_imputed[k])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)  

        gene_num = genes_num_all[k]
        # Prepare model configuration for the current fold
        channel_list = copy.deepcopy(args.channel_list)
        model = CSGNet(in_channel=1, mid_channel=mid_channel, num_nodes=gene_num, out_channel=class_num, dropout_ratio=dropout_ratio, 
                       channel_list=channel_list, global_conv1_dim=global_conv1_dim, global_conv2_dim=global_conv2_dim).to(device)
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)

        # Learning rate scheduler using ExponentialLR
        scheduler = ExponentialLR(optimizer, gamma=0.8)

        # Training loop
        for epoch in range(1, max_epoch):
            train_loss, train_acc, train_f1 = train(model, optimizer, train_loader, device, loss_fn)
            scheduler.step()  # Update learning rate
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}, lr: {lr:.6f}, loss: {train_loss:.6f}, Train-acc: {train_acc:.4f}, Train-f1: {train_f1:.4f}')
            print(f'Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()[0]}')

        # Save the trained model
        model_file = pathjoin(models_folder, f'wcsn_{base_filename}_a{csn_alpha}_hvgs{HVGs_num}_model{addname}{k_fold}.pth')
        torch.save(model, model_file)


def train(model, optimizer, train_loader, device, loss_fn=None, verbose=False):
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
            loss = F.cross_entropy(output, data.y.reshape(-1), weight=None)
        else:
            loss = loss_fn(output, data.y.reshape(-1))
        loss.backward()
        optimizer.step()

        loss_all += loss.item()

        with torch.no_grad():  # Disable gradient calculation for inference
            pred = output.argmax(dim=1).cpu().numpy()  # Predicted labels
            y = data.y.cpu().numpy()  # True labels

        y_pred.extend(pred)
        y_true.extend(y)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    return loss_all / iters, acc, f1


if __name__ == '__main__':
    start_time = time.time()  
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)
    classify_train(args)

    end_time = time.time()
    print(f"Code run time: {end_time - start_time} s")
