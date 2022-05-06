import optuna
import numpy as np
import warnings

from rdkit import RDLogger
from models import GATMem
from train_utils import setup


warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

np.random.seed(42)

config_default = {
    'c_hidden_soft': 256,
    'layers_soft': 2,
    'drop_rate_soft_dense': 0.18157566871830347,
    'drop_rate_soft': 0.17834544408336056,
    'drop_rate_hard_dense_1': 0.30640872696383925,
    'drop_rate_hard_dense_2': 0.38842013955429855,
    'dense_input_head': 256,
    'dense_input_hidden': 128,
    'pos_weight': 1.3,
    'optim': "Adam",
    'lr': 3.1804358785861437e-03,
    'batch_size': 512
}



name = f"GAT_Memory"
net = GATMem(80, config_default['c_hidden_soft'], config_default['c_hidden_soft'],  dp_rate=config_default['drop_rate_soft'],
             dp_rate_linear=config_default['drop_rate_soft_dense'], **config_default)


# Default setup runs Optuna optimization to find best model architecture
# Uncomment lines 40-43 to train a custom model specified in config_default
if __name__ == "__main__":
    # sensitivity, specificity = setup_training(config_default, f'./outputs/memory/Strong_{name}', df,
    #                                           wandb_name='Strong_' + name, net=net)
    #
    # print(f'Sensitivity: {sensitivity}\nSpecificity: {specificity}')
    study = optuna.create_study(
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        sampler=optuna.samplers.TPESampler(),
        directions=["maximize"])

    study.optimize(setup, n_trials=100, timeout=None)
