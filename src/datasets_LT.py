import os
import torch
from torch_geometric.data import Dataset


class MyDataset2(Dataset):
    def __init__(self, root, my_indices=None):
        super(MyDataset2, self).__init__(root)
        # If my_indices is provided, use it; otherwise, use all indices from the processed data
        self.my_indices = my_indices if my_indices is not None else range(len(self.processed_file_names))

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
        """
        Loads and returns the processed data for a given index. This includes 
        loading the edge indices and weights, performing necessary transformations 
        (such as log1p on edge weights), and ensuring the symmetry of the graph's 
        adjacency matrix by adding reverse edges.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            torch_geometric.data.Data: The data object containing node features, 
            edge indices, and edge weights, with symmetric edges and transformed weights. (WCSN_logarithmic transformation)
        """
        # Retrieve the absolute index from the list of indices
        absolute_idx = self.my_indices[idx]
        # Load the preprocessed data from the disk (edges, features, weights)
        data = torch.load(os.path.join(self.processed_dir, f'cell_{absolute_idx}.pt'))
        data.edge_index = data.edge_index.to(torch.int64)
        
        # Get the current edge indices and edge weights from the loaded data
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        
        # Apply log1p transformation to the edge weights (log(1 + x))
        edge_weight = torch.log1p(edge_weight)

        # Generate symmetric edge indices (add reverse edges, i.e., (i, j) and (j, i))
        row, col = edge_index
        symmetric_edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        
        # Generate symmetric edge weights (duplicate the weights for both directions)
        symmetric_edge_weight = torch.cat([edge_weight, edge_weight])

        # Update the data object with the symmetric edge indices and weights
        data.edge_index = symmetric_edge_index
        data.edge_weight = symmetric_edge_weight

        return data
