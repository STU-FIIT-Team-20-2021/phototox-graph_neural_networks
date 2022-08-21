import pandas as pd
import numpy as np
import torch
import torch_geometric.data as geom_data

from models import GATMem, GraphGNNModel
from data_utils import create_data_list

config_default = {
    'c_hidden_soft': 345,
    'layers_soft': 2,
    'drop_rate_soft_dense': 0.369310585790777,
    'drop_rate_soft': 0.369310585790777,
    'drop_rate_hard_dense_1': 0.4997282537547015,
    'drop_rate_hard_dense_2': 0.4997282537547015,
    'dense_input_head': 256,
    'dense_input_hidden': 128,
    'pos_weight': 1.3,
    'optim': "RMSProp",
    'lr': 3.1804358785861437e-03,
    'batch_size': 512
}

net = GraphGNNModel(80, config_default['c_hidden_soft'], config_default['c_hidden_soft'],
                        dp_rate_linear=config_default['drop_rate_soft_dense'],
                        dp_gnn=config_default['drop_rate_soft'], **config_default)
# net = GATMem(80, 345, 345, num_layers=4, dp_rate=0.369310585790777,
#              dp_rate_linear=0.4997282537547015, **config_default)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

state_dict = torch.load(f'F:/Halinkovic/gnns/outputs/final_f1b/Strong_Mutagenicity__chs=345_ls=2_drsd=0.3899208485746232_drs=0.369310585790777_drhd1=0.4997282537547015_drhd2=0.4985505573688271_dih=128_dihidden=64_optim=RMSprop_lr=6.181155368363278e-05_batchsize=128_posweight=1.4805839905064306/checkpoint.pth',
                        map_location=device)

net.load_state_dict(state_dict)
net.to(device)
net.eval()

mol_df = pd.read_csv('./mutagenicity_pseudolabels.csv', delimiter=',')
print(mol_df.shape)
mol_df = mol_df.dropna()
print(mol_df.shape)

batch_size = 512
data_list = create_data_list(mol_df['Smiles'].values.tolist(), mol_df['LGB_label'].values.tolist())
loader = geom_data.DataLoader(data_list, batch_size=batch_size, shuffle=True, num_workers=0)


gnn_labels = np.array([])

for batch in iter(loader):
    batch = batch.to(device=device)
    x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

    with torch.cuda.amp.autocast(enabled=False):
        preds = net(x, edge_index, batch_idx)

        if net.n_classes == 1:
            preds = preds.squeeze(dim=-1)
    y = batch.y.cpu().numpy()
    preds = (torch.sigmoid(preds).data.float() > 0.5).float().cpu().numpy()
    gnn_labels = np.append(gnn_labels, preds)

out_df = mol_df.copy()
out_df['GNN_label'] = gnn_labels
out_df = out_df.astype({"Smiles": str, "RF_label": int, "LGB_label": int, "GNN_label": int})
out_df.to_csv('./mutagenicity_pseudolabels_gnn.csv', index=False)

