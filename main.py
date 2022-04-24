import glob

import torch
import torch.nn as nn
import optuna
import wandb

from models import GraphGNNModel
from train import setup_training

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


df = pd.read_csv('./alvadesc_full_cleaned.csv', index_col=0)
df = df.drop([345, 346], axis=0)
df = df.drop_duplicates(subset=["Smiles"], keep=False)

pre_df = pd.read_csv('./10k_smiles_scored.csv', sep='\t')
pre_df = pre_df.drop_duplicates(subset=["Smiles"], keep=False)

df = df[['Phototoxic', 'Smiles']]
pre_df = pre_df[['Phototoxic', 'Smiles']]


config_default = {
    'c_hidden_soft': 234,
    'layers_soft': 2,
    'drop_rate_soft_dense': 0.18157566871830347,
    'drop_rate_soft': 0.17834544408336056,
    'drop_rate_hard_dense_1': 0.30640872696383925,
    'drop_rate_hard_dense_2': 0.38842013955429855,
    'dense_input_head': 256,
    'dense_input_hidden': 128,
    'pos_weight': 1.2686536617518058,
    'optim': "Adam",
    'lr': 1.1804358785861437e-05,
    'batch_size': 512
}

def train_no_pretrain(trial: optuna.trial.Trial, config: dict):
    name = f"Optuna__chs={config['c_hidden_soft']}_ls={config['layers_soft']}_drsd={config['drop_rate_soft_dense']}_" \
           f"drs={config['drop_rate_soft']}_drhd1={config['drop_rate_hard_dense_1']}_drhd2={config['drop_rate_hard_dense_2']}_" \
           f"dih={config['dense_input_head']}_dihidden={config['dense_input_hidden']}_optim={config['optim']}_" \
           f"lr={config['lr']}_batchsize={config['batch_size']}_posweight={config['pos_weight']}"
    net = GraphGNNModel(80, config['c_hidden_soft'], config['c_hidden_soft'],
                        dp_rate_linear=config['drop_rate_soft_dense'],
                        dp_gnn=config['drop_rate_soft'], **config)

    net.start_trial(trial)

    try:
        sensitivity, specificity = setup_training(config, f'./outputs/no_pretrain/Strong_{name}', df,
                                                  wandb_name='Strong_'+name, net=net)
        wandb.finish()
    except optuna.TrialPruned:
        raise optuna.TrialPruned()

    return sensitivity, specificity

def train_model_both(trial: optuna.trial.Trial, config: dict):
    name = f"Optuna__chs={config['c_hidden_soft']}_ls={config['layers_soft']}_drsd={config['drop_rate_soft_dense']}_" \
           f"drs={config['drop_rate_soft']}_drhd1={config['drop_rate_hard_dense_1']}_drhd2={config['drop_rate_hard_dense_2']}_" \
           f"dih={config['dense_input_head']}_dihidden={config['dense_input_hidden']}_optim={config['optim']}_" \
           f"lr={config['lr']}_batchsize={config['batch_size']}_posweight={config['pos_weight']}"
    setup_training(config, f'./outputs/{name}', pre_df, wandb_name=name, pretrain=True)
    wandb.finish()
    pre_trained = glob.glob(f'./outputs/{name}/*.pth')
    last = max(pre_trained, key=os.path.getctime)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    net = GraphGNNModel(80, config['c_hidden_soft'], config['c_hidden_soft'],
                        dp_rate_linear=config['drop_rate_soft_dense'],
                        dp_gnn=config['drop_rate_soft'], **config)
    net.load_state_dict(torch.load(last, map_location=device))

    new_head = nn.Sequential(
        nn.Dropout(config['drop_rate_hard_dense_1']),
        nn.Linear(config['c_hidden_soft'] * 2 ** config['layers_soft'], 256),
        nn.ReLU(inplace=True),
        nn.Dropout(config['drop_rate_hard_dense_2']),
        nn.Linear(256, 1)
    )

    for param in net.parameters():
        param.requires_grad = False

    list(net.children())[0].head = new_head
    net.start_trial(trial)

    try:
        sensitivity, specificity = setup_training(config, f'./outputs/Strong_{name}', df,
                                                  wandb_name=name, net=net)
        wandb.finish()
    except optuna.TrialPruned:
        raise optuna.TrialPruned()

    return sensitivity, specificity


def setup(trial: optuna.trial.Trial):

    a = trial.suggest_int('dense_input_head', 5, 11)
    b = trial.suggest_int('dense_input_hidden', 3, a)

    config = {
        'c_hidden_soft': trial.suggest_int('c_hidden_soft', 128, 384),
        'layers_soft': trial.suggest_int('layers_soft', 2, 5),
        'drop_rate_soft_dense': trial.suggest_uniform('drop_rate_soft_dense', 0.05, 0.5),
        'drop_rate_soft': trial.suggest_uniform('drop_rate_soft', 0.05, 0.5),
        'drop_rate_hard_dense_1': trial.suggest_uniform('drop_rate_hard_dense_1', 0.05, 0.5),
        'drop_rate_hard_dense_2': trial.suggest_uniform('drop_rate_hard_dense_2', 0.05, 0.5),
        'dense_input_head': 2 ** a,
        'dense_input_hidden': 2 ** b,
        'pos_weight': trial.suggest_uniform('pos_weight', 0.8, 1.5),
        'optim': trial.suggest_categorical('optim', ["Adam", "RMSprop", "SGD"]),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    }

    sensitivity, specificity = train_no_pretrain(trial, config)

    return sensitivity, specificity


name = f"Optuna__chs={config_default['c_hidden_soft']}_ls={config_default['layers_soft']}_drsd={config_default['drop_rate_soft_dense']}_" \
           f"drs={config_default['drop_rate_soft']}_drhd1={config_default['drop_rate_hard_dense_1']}_drhd2={config_default['drop_rate_hard_dense_2']}_" \
           f"dih={config_default['dense_input_head']}_dihidden={config_default['dense_input_hidden']}_optim={config_default['optim']}_" \
           f"lr={config_default['lr']}_batchsize={config_default['batch_size']}_posweight={config_default['pos_weight']}"
net = GraphGNNModel(80, config_default['c_hidden_soft'], config_default['c_hidden_soft'],
                    dp_rate_linear=config_default['drop_rate_soft_dense'],
                    dp_gnn=config_default['drop_rate_soft'], **config_default)


sensitivity, specificity = setup_training(config_default, f'./outputs/no_pretrain/Strong_{name}', df,
                                          wandb_name='Strong_'+name, net=net)

# study = optuna.create_study(pruner=optuna.pruners.SuccessiveHalvingPruner(), sampler=optuna.samplers.TPESampler(), directions=["maximize", "maximize"])
# study.optimize(setup, n_trials=100, timeout=None)

# print(f"Train accuracy: {100.0*result['train_acc']:4.2f}%")
# print(f"Train sensitivity: {100.0*result['train_sensitivity']:4.2f}%")
# print(f"Train specificity: {100.0*result['train_specificity']:4.2f}%")
# print('------')
# print(f"Test accuracy: {100.0*result['test_acc']:4.2f}%")
# print(f"Test sensitivity: {100.0*result['test_sensitivity']:4.2f}%")
# print(f"Test specificity: {100.0*result['test_specificity']:4.2f}%")