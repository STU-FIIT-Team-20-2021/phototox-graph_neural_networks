import glob

import torch
import torch.nn as nn
from ray import tune
import optuna
import wandb

from models import GraphGNNModel
from train import setup_training


import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import warnings
import os

os.environ["PATH"] += r";C:\Users\Lukas\Anaconda3\envs\Halinkovic_GNNs"

from rdkit import RDLogger


warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

np.random.seed(42)
pl.seed_everything(42)



df = pd.read_csv('./alvadesc_full_cleaned.csv', index_col=0)
df = df.drop([345, 346], axis=0)
df = df.drop_duplicates(subset=["Smiles"], keep=False)

pre_df = pd.read_csv('./10k_smiles_scored.csv', sep='\t')
pre_df = pre_df.drop_duplicates(subset=["Smiles"], keep=False)

df = df[['Phototoxic', 'Smiles']]
pre_df = pre_df[['Phototoxic', 'Smiles']]


config = {
    'c_hidden_soft': tune.randint(256, 1024),
    'layers_soft': tune.randint(3, 10),
    'drop_rate_soft_dense': tune.quniform(0.1, 0.5, 0.1),
    'drop_rate_soft': tune.quniform(0.1, 0.5, 0.1),
    'drop_rate_hard_dense': tune.quniform(0.1, 0.5, 0.1),
    'optim': tune.choice(["Adam", "RMSprop", "SGD"]),
    'type': tune.choice(["GAT", "GCN", "GraphConv"]),
    'lr': tune.loguniform(1e-5, 1e-1),
}

config_default = {
    'c_hidden_soft': 256,
    'layers_soft': 3,
    'drop_rate_soft_dense': 0.4,
    'drop_rate_soft': 0.2,
    'drop_rate_hard_dense': 0.4,
    'c_hidden_hard': 256,
    'optim': "Adam",
    'type': 'GAT',
    'pos_weight': 1.5,
    'lr': 1e-3,
    'batch_size': 1024
}

# setup_training(config_default, './outputs/test_run', pre_df, wandb_name='test_run')

pre_trained = glob.glob('./outputs/test_run/*.pth')
last = max(pre_trained, key=os.path.getctime)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
net = GraphGNNModel(79, config_default['c_hidden_soft'], config_default['c_hidden_soft'], dp_rate_linear=config_default['drop_rate_soft_dense'],
                            dp_gnn=config_default['drop_rate_soft'],**config_default)
net.load_state_dict(torch.load(last, map_location=device))

new_head = nn.Sequential(
    nn.Dropout(config_default['drop_rate_hard_dense']),
    nn.Linear(config_default['c_hidden_soft'] * 2 ** config_default['layers_soft'], 256),
    nn.ReLU(inplace=True),
    nn.Dropout(config_default['drop_rate_hard_dense']),
    nn.Linear(256, 1)
)

for param in net.parameters():
    param.requires_grad = False

list(net.children())[0].head = new_head

setup_training(config_default, './outputs/test_run_strong', pre_df, wandb_name='test_run_strong', net=net)
wandb.finish()

def setup(trial: optuna.trial.Trial):

    a = trial.suggest_int('dense_input_head', 5, 11)
    b = trial.suggest_int('dense_input_hidden', 3, a)

    config = {
        'c_hidden_soft': trial.suggest_int('c_hidden_soft', 128, 512),
        'layers_soft': trial.suggest_int('layers_soft', 2, 5),
        'drop_rate_soft_dense': trial.suggest_uniform('drop_rate_soft_dense', 0.05, 0.5),
        'drop_rate_soft': trial.suggest_uniform('drop_rate_soft', 0.05, 0.5),
        'drop_rate_hard_dense_1': trial.suggest_uniform('drop_rate_hard_dense_1', 0.05, 0.5),
        'drop_rate_hard_dense_2': trial.suggest_uniform('drop_rate_hard_dense_2', 0.05, 0.5),
        'dense_input_head': 2 ** a,
        'dense_input_hidden': 2 ** b,
        'optim': trial.suggest_categorical('optim', ["Adam", "RMSprop", "SGD"]),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
    }

    model, result, trainer = train_model_both(config, trial)

    return result['test_sensitivity'], result['test_specificity']




# study = optuna.create_study(pruner=optuna.pruners.SuccessiveHalvingPruner(), sampler=optuna.samplers.TPESampler(), directions=["maximize", "maximize"])
# study.optimize(setup, n_trials=100, timeout=300)

# print(f"Train accuracy: {100.0*result['train_acc']:4.2f}%")
# print(f"Train sensitivity: {100.0*result['train_sensitivity']:4.2f}%")
# print(f"Train specificity: {100.0*result['train_specificity']:4.2f}%")
# print('------')
# print(f"Test accuracy: {100.0*result['test_acc']:4.2f}%")
# print(f"Test sensitivity: {100.0*result['test_sensitivity']:4.2f}%")
# print(f"Test specificity: {100.0*result['test_specificity']:4.2f}%")