from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

import torch
import torch_geometric.data as geom_data


import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import warnings
import os

os.environ["PATH"] += r";C:\Users\Lukas\Anaconda3\envs\Halinkovic_GNNs"

from rdkit import RDLogger
from data_utils import create_data_list
from train_utils import train_model_both


# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

np.random.seed(42)
pl.seed_everything(42)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



df = pd.read_csv('./alvadesc_full_cleaned.csv', index_col=0)
df = df.drop([345, 346], axis=0)

pre_df = pd.read_csv('./10k_smiles_scored.csv', sep='\t')

df = df.drop_duplicates(subset=["Smiles"], keep=False)
pre_df = pre_df.drop_duplicates(subset=["Smiles"], keep=False)

df = df[['Phototoxic', 'Smiles']]
pre_df = pre_df[['Phototoxic', 'Smiles']]


data_list = create_data_list(pre_df['Smiles'], pre_df['Phototoxic'])
train, test = train_test_split(data_list, test_size=0.1, stratify=[t.y for t in data_list])
train, valid = train_test_split(train, test_size=0.15, stratify=[t.y for t in train])


strong_data_list = create_data_list(df['Smiles'], df['Phototoxic'])
strong_train, strong_test = train_test_split(strong_data_list, test_size=0.2, stratify=[t.y for t in strong_data_list])
strong_train, strong_valid = train_test_split(strong_train, test_size=0.13, stratify=[t.y for t in strong_train])




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

train_loader = geom_data.DataLoader(train, batch_size=128, shuffle=True, num_workers=4)
val_loader = geom_data.DataLoader(valid, batch_size=128, num_workers=4)
test_loader = geom_data.DataLoader(test, batch_size=128, num_workers=4)

strong_train_loader = geom_data.DataLoader(strong_train, batch_size=128, shuffle=True, num_workers=4)
strong_val_loader = geom_data.DataLoader(strong_valid, batch_size=128, num_workers=4)
strong_test_loader = geom_data.DataLoader(strong_test, batch_size=128, num_workers=4)


trainable = tune.with_parameters(
    train_model_both,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    strong_train_loader=strong_train_loader,
    strong_val_loader=strong_val_loader,
    strong_test_loader=strong_test_loader)

algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=4)
scheduler = AsyncHyperBandScheduler()
analysis = tune.run(trainable,
                    resources_per_trial={
                        "cpu": 20,
                        "gpu": 1
                    },
                    search_alg=algo,
                    scheduler=scheduler,
                    metric="val_loss",
                    mode="min",
                    verbose=2,
                    config=config,
                    num_samples=30,
                    name='tune_graph')


# print(f"Train accuracy: {100.0*result['train_acc']:4.2f}%")
# print(f"Train sensitivity: {100.0*result['train_sensitivity']:4.2f}%")
# print(f"Train specificity: {100.0*result['train_specificity']:4.2f}%")
# print('------')
# print(f"Test accuracy: {100.0*result['test_acc']:4.2f}%")
# print(f"Test sensitivity: {100.0*result['test_sensitivity']:4.2f}%")
# print(f"Test specificity: {100.0*result['test_specificity']:4.2f}%")