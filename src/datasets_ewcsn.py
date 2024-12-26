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
        Loads and returns the processed data for a specific index.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            data (torch_geometric.data.Data): EWCSN, The data object containing node features, edge indices, and edge weights. 
        """
        # Retrieve the absolute index from the list of indices
        absolute_idx = self.my_indices[idx]
        data = torch.load(os.path.join(self.processed_dir, f'cell_{absolute_idx}.pt'))
        data.edge_index = data.edge_index.to(torch.int64)
        
        # Get the current edge indices and weights from the data
        edge_index = data.edge_index
        edge_weight = data.edge_weight

        # Merge the network adjacency (net_adj) by filling in missing edges
        # For edges that are in net_adj but not in edge_index, add them with the mean weight
        mean_weight = edge_weight.mean().item()
        
        # Convert existing edge indices to a set for easy comparison
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        # Use set difference to find new edges in net_adj that are not in edge_index
        new_edges = self.net_adj_set - existing_edges
        # If there are new edges, add them to edge_index and set their weights to the mean weight
        if new_edges:
            new_edges = torch.tensor(list(new_edges), dtype=torch.int64).T  # Convert to [2, N] shape
            new_weights = torch.tensor([mean_weight] * new_edges.size(1), dtype=torch.float32)

            # Update edge_index and edge_weight
            edge_index = torch.cat([edge_index, new_edges], dim=1)
            edge_weight = torch.cat([edge_weight, new_weights])

        # Generate symmetric edge indices (both (i, j) and (j, i) edges)
        row, col = edge_index
        symmetric_edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        
        # Generate symmetric edge weights (same weight for both directions)
        symmetric_edge_weight = torch.cat([edge_weight, edge_weight])

        # Update the data object with the new symmetric edges and weights
        data.edge_index = symmetric_edge_index
        data.edge_weight = symmetric_edge_weight

        return data
