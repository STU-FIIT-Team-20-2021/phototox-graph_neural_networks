import torch
from torch import nn
import torch.optim as optim
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GATConv, Linear, to_hetero
from torch_geometric.data import HeteroData

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from sklearn.metrics import classification_report
from ray import tune



gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

class GNNModel(torch.nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class GraphGNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        super().__init__()
        self.GNN = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden,
                            **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, x, edge_index, batch_idx):
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx) # TODO: learn more about pooling options
        x = self.head(x)
        return x



class GraphLevelGNN(pl.LightningModule):

    def __init__(self, config, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = GraphGNNModel(**model_kwargs)
        self.config = config
        self.loss_module = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3)) if self.hparams.c_out == 1 else nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)

        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)
        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        if mode == 'test':
          try:
            sensitivity = classification_report(y_true=data.y.cpu().detach().numpy(), y_pred=preds.cpu().detach().numpy(), output_dict=True)['1.0']['recall']
          except KeyError:
            sensitivity = 0

          try:
            specificity = classification_report(y_true=data.y.cpu().detach().numpy(), y_pred=preds.cpu().detach().numpy(), output_dict=True)['0.0']['recall']
          except KeyError:
            specificity = 0

          return loss, acc, sensitivity, specificity
        else:
          return loss, acc

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config['optim'])(self.parameters(), lr=self.config['lr'])
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="val")
        # tune.report(val_loss=loss, val_acc=acc, val_sensitivity=sensitivity, val_specificity=specificity)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        loss, acc, sensitivity, specificity = self.forward(batch, mode="test")
        # tune.report(test_loss=loss, test_acc=acc, test_sensitivity=sensitivity, test_specificity=specificity)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_sensitivity', sensitivity)
        self.log('test_specificity', specificity)