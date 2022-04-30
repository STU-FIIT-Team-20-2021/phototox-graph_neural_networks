import torch
from torch import nn
import torch.optim as optim
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GATConv, Linear, to_hetero
from torch_geometric.data import HeteroData

from sklearn.metrics import classification_report



gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATv2Conv,
    "GraphConv": geom_nn.GraphConv
}


class GNNModel(torch.nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GAT", dp_rate=0.1, **kwargs):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(kwargs['layers_soft']):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          heads=2),
                          # **kwargs),
                geom_nn.norm.BatchNorm(out_channels*2),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = out_channels * 2
            out_channels = in_channels
        # layers += [gnn_layer(in_channels=in_channels,
        #                      out_channels=out_channels,
        #                      heads=2)]
                             # **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class GraphGNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, dp_gnn=0.1, **kwargs):
        super().__init__()
        self.trial = None
        self.n_classes=1
        self.GNN = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden,
                            dp_rate=dp_gnn,
                            **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden * 2**kwargs['layers_soft'], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_rate_linear),
            nn.Linear(256, 1)
        )

    def start_trial(self, trial):
        self.trial = trial

    def forward(self, x, edge_index, batch_idx):
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx) # TODO: learn more about pooling options
        x = self.head(x)
        return x


class MSPModel(geom_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j