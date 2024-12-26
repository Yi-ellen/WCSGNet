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
        Loads and returns the processed data for a given index. This includes 
        loading the edge indices and weights, performing necessary transformations 
        (such as log1p on edge weights), and ensuring that missing edges from the 
        network adjacency matrix are added to the graph data.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            torch_geometric.data.Data: The data object containing node features, 
            edge indices, and edge weights, with symmetric edges and transformed weights. (EWCSN_LWT)
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

        # Merge the network adjacency (net_adj) to include missing edges
        # For edges that exist in net_adj but not in edge_index, add them with the mean weight
        mean_weight = edge_weight.mean().item()

        # Convert the existing edge indices to a set of tuples (row, col) for comparison
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        # Use set difference to find new edges that are in net_adj but not in edge_index
        new_edges = self.net_adj_set - existing_edges
        
        # If new edges are found, add them to the graph's edge_index and assign them the mean weight
        if new_edges:
            new_edges = torch.tensor(list(new_edges), dtype=torch.int64).T  # Convert to [2, N] shape
            new_weights = torch.tensor([mean_weight] * new_edges.size(1), dtype=torch.float32)

            # Update edge_index and edge_weight with the new edges and their weights
            edge_index = torch.cat([edge_index, new_edges], dim=1)
            edge_weight = torch.cat([edge_weight, new_weights])

        # Generate symmetric edge indices (add reverse edges, i.e., (i, j) and (j, i))
        row, col = edge_index
        symmetric_edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        
        # Generate symmetric edge weights (duplicate the weights for both directions)
        symmetric_edge_weight = torch.cat([edge_weight, edge_weight])

        # Update the data object with the symmetric edge indices and weights
        data.edge_index = symmetric_edge_index
        data.edge_weight = symmetric_edge_weight

        return data
