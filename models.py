import torch
from torch import nn
import torch_geometric.nn as geom_nn
from torch_geometric.nn import MemPooling


gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATv2Conv,
    "GraphConv": geom_nn.GraphConv
}


# Model architecture of the Graph part of our network
# Basic, but fully optimizeable by Optuna
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
                geom_nn.norm.BatchNorm(out_channels*2),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = out_channels * 2
            out_channels = in_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


# Appends Head to Graph model, interacts with Optuna directly
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
        x = geom_nn.global_mean_pool(x, batch_idx)
        x = self.head(x)
        return x


# Ignore this - unfinished implementation of a Memory network
class GATMem(torch.nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GAT", dp_rate=0.1, dp_rate_linear=0.3, **kwargs):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        self.n_classes=1

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(kwargs['layers_soft']):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          heads=2),
                          # **kwargs),
                geom_nn.norm.BatchNorm(out_channels*2),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = out_channels * 2
            out_channels = in_channels

        out_channels /= 2
        layers += [MemPooling(in_channels=in_channels,
                             out_channels=int(out_channels),
                             heads=2,
                             num_clusters=1),
                    geom_nn.norm.BatchNorm(int(out_channels)),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(dp_rate)]

        in_channels = int(out_channels)
        out_channels /= 2

        layers += [MemPooling(in_channels=in_channels,
                             out_channels=int(out_channels),
                             heads=2,
                             num_clusters=1),
                    geom_nn.norm.BatchNorm(int(out_channels)),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(dp_rate)]

        self.layers = nn.ModuleList(layers)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden * 2 ** kwargs['layers_soft'], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_rate_linear),
            nn.Linear(256, 1)
        )

    def forward(self, x, edge_index, batch_idx):
        for l in self.layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            elif isinstance(x, tuple):
                x = l(torch.reshape(x[0], (1, -1)))
            else:
                x = l(x)
        x = self.head(x)
        return x