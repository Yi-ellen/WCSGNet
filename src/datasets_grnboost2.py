import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root, com_net=None, expr=None, label=None, my_indices=None):
        super(MyDataset, self).__init__(root)
        self.my_indices = my_indices if my_indices is not None else range(len(label))
        self.expr = expr
        self.label = label
        indices = np.array([com_net['gene1'], com_net['gene2']], dtype=np.int64)

        self.edge_index = torch.tensor(indices, dtype=torch.int64) 
        self.edge_weight = torch.tensor(com_net['weight'], dtype=torch.float)  

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        cell_files = os.listdir(self.processed_dir)
        return cell_files

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.my_indices)

    def get(self, idx):
        absolute_idx = self.my_indices[idx]

        x = self.expr[absolute_idx]   
        x = x.reshape(-1, 1)  
        x = x.astype(np.float32)
        x = torch.from_numpy(x)     
        y = torch.tensor(self.label[absolute_idx], dtype=torch.long) 

        data = Data(x=x, edge_index=self.edge_index, edge_weight=self.edge_weight, y=y)

        return data 


