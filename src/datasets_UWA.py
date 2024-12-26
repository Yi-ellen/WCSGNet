import os
import torch
from torch_geometric.data import Dataset


class MyDataset2(Dataset):
    def __init__(self, root, my_indices=None, net_adj=None):
        super(MyDataset2, self).__init__(root)
        # If my_indices is provided, use it; otherwise, use all indices from the processed data
        self.my_indices = my_indices if my_indices is not None else range(len(self.processed_file_names))
        # Store the network adjacency as a set of tuples (row, col)
        self.net_adj_set = set(zip(net_adj.row, net_adj.col))

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
        Loads and returns the processed data for a given index. This includes loading the edge indices, 
        setting edge weights to 1, adding missing edges from the network adjacency matrix with a weight of 0.1, 
        and ensuring the symmetry of the graph's adjacency matrix by adding reverse edges.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            torch_geometric.data.Data: The data object containing node features, 
            edge indices, and edge weights, with symmetric edges and updated weights. (EWCSN_UWA)
        """
        # Retrieve the absolute index from the list of indices
        absolute_idx = self.my_indices[idx]
        # Load the preprocessed data from the disk (edges, features, weights)
        data = torch.load(os.path.join(self.processed_dir, f'cell_{absolute_idx}.pt'))
        data.edge_index = data.edge_index.to(torch.int64)
        
        # Get the current edge indices
        edge_index = data.edge_index
        
        # Set the edge weights to 1 for the existing edges
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

        # Merge the network adjacency (net_adj) to include missing edges
        # For edges that exist in net_adj but not in edge_index, add them with a weight of 0.1
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        # Use set difference to find new edges that are in net_adj but not in edge_index
        new_edges = self.net_adj_set - existing_edges
        
        # If new edges are found, add them to the graph's edge_index with a weight of 0.1
        if new_edges:
            new_edges = torch.tensor(list(new_edges), dtype=torch.int64).T  # Convert to [2, N] shape
            new_weights = torch.tensor([1] * new_edges.size(1), dtype=torch.float32)
            # new_weights = torch.tensor([0.1] * new_edges.size(1), dtype=torch.float32)

            # Update edge_index and edge_weight with the new edges and their weights
            edge_index = torch.cat([edge_index, new_edges], dim=1)
            edge_weight = torch.cat([edge_weight, new_weights])

        # Generate symmetric edge indices by adding reverse edges (i.e., (i, j) and (j, i))
        row, col = edge_index
        symmetric_edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        
        # Generate symmetric edge weights by duplicating the weights
        symmetric_edge_weight = torch.cat([edge_weight, edge_weight])

        # Update the data object with the symmetric edge indices and weights
        data.edge_index = symmetric_edge_index
        data.edge_weight = symmetric_edge_weight

        return data







