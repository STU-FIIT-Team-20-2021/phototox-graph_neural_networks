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
    'c_hidden_soft': 256,
    'layers_soft': 3,
    'drop_rate_soft_dense': 0.4,
    'drop_rate_soft': 0.2,
    'drop_rate_hard_dense': 0.4,
    'c_hidden_hard': 256,
    'optim': "Adam",
    'type': 'GAT',
    'pos_weight': 1,
    'lr': 1e-3,
    'batch_size': 1024
}


def f1b_score(sensitivity, specificity, beta=0.5):
    return (1 + beta ** 2) * (sensitivity * specificity) / (beta ** 2 * sensitivity + specificity)


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
        'pos_weight': trial.suggest_uniform('pos_weight', 1.2, 2.5),
        'optim': trial.suggest_categorical('optim', ["Adam", "RMSprop", "SGD"]),
        'lr': trial.suggest_loguniform('lr', 1e-4, 1e-1),
        'batch_size': trial.suggest_categorical('batch_size', [512, 1024])
    }

    sensitivity, specificity = train_model_both(trial, config)

    return f1b_score(sensitivity, specificity)


if __name__ == "__main__":
    study = optuna.create_study(
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        sampler=optuna.samplers.TPESampler(),
        directions=["maximize"])

    study.optimize(setup, n_trials=100, timeout=None)

# print(f"Train accuracy: {100.0*result['train_acc']:4.2f}%")
# print(f"Train sensitivity: {100.0*result['train_sensitivity']:4.2f}%")
# print(f"Train specificity: {100.0*result['train_specificity']:4.2f}%")
# print('------')
# print(f"Test accuracy: {100.0*result['test_acc']:4.2f}%")
# print(f"Test sensitivity: {100.0*result['test_sensitivity']:4.2f}%")
# print(f"Test specificity: {100.0*result['test_specificity']:4.2f}%")