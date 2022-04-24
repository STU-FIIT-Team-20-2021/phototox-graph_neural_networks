import argparse
import glob
import logging
import sys
from pathlib import Path
import os
import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as geom_data
import wandb
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data_utils import create_data_list

from models import GraphGNNModel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def f1b_score(sensitivity, specificity, beta=0.5):
    return (1 + beta ** 2) * (sensitivity * specificity) / (beta ** 2 * sensitivity + specificity)


def setup_training(config: dict, save_checkpoint: str, df: pd.DataFrame, wandb_name: str, net: nn.Module = None, pretrain: bool = False):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if net is None:
        net = GraphGNNModel(80, config['c_hidden_soft'], config['c_hidden_soft'], dp_rate_linear=config['drop_rate_soft_dense'],
                            dp_gnn=config['drop_rate_soft'],**config)
    net.to(device)

    if not os.path.exists(f'./outputs/no_pretrain/{wandb_name}'):
        os.mkdir(f'./outputs/no_pretrain/{wandb_name}')

    experiment = wandb.init(project='TP-GNNs', resume='allow', name=wandb_name, group='no_pretrain',
                            config={
                                "learning_rate": config['lr'],
                                "batch_size": config['batch_size'],
                                "epochs": 500,
                                "save_checkpoint": save_checkpoint,
                            })

    return train_net(net, device, epochs=500, batch_size=config['batch_size'], learning_rate=config['lr'], experiment=experiment,
                     df=df, opt=config['optim'], dir_checkpoint=f'./outputs/no_pretrain/{wandb_name}', pos_weight=config['pos_weight'],
                     pretrain=pretrain)


def train_net(net,
              device,
              epochs: int = 50,
              amp: bool = False,
              batch_size: int = 1024,
              learning_rate: float = 1e-4,
              save_checkpoint: bool = True,
              df: pd.DataFrame = None,
              opt: str = 'Adam',
              dir_checkpoint: str = 'outputs',
              experiment = None,
              pos_weight: float = 1.0,
              early_stopping_patience: int = 5,
              pretrain: bool = False,
              l1_lambda: float = 0):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # experiment.watch(net, log_freq=10, log='all')

    data_list = create_data_list(df['Smiles'], df['Phototoxic'])
    if not pretrain:
        train, test = train_test_split(data_list, test_size=0.2, stratify=[t.y for t in data_list], random_state=42)
    else:
        train, test = train_test_split(data_list, test_size=0.1, stratify=[t.y for t in data_list], random_state=42)
    train, valid = train_test_split(train, test_size=0.15, stratify=[t.y for t in train], random_state=42)

    n_train = len(train)
    n_val = len(valid)
    n_test = len(test)


    train_loader = geom_data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = geom_data.DataLoader(valid, batch_size=batch_size, num_workers=0)
    test_loader = geom_data.DataLoader(test, batch_size=batch_size, num_workers=0)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Test size:       {n_test}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    if opt == 'Adam':
        optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=learning_rate)
    elif opt == 'RMSprop':
        optimizer = optim.RMSprop([p for p in net.parameters() if p.requires_grad], lr=learning_rate)
    elif opt == 'SGD':
        optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad], lr=learning_rate)

    # TODO: Decide if we want this
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if net.n_classes == 1:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    else:
        criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    prev_best = np.inf
    epochs_since_improvement = 0
    val_sensitivities, val_specificities = [], []
    for epoch in range(epochs):
        train_loss_total = 0
        val_loss_total = 0
        test_loss_total = 0
        net.train()
        epoch_score = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='mol') as pbar:
            for batch in iter(train_loader):
                batch = batch.to(device=device)
                x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

                with torch.cuda.amp.autocast(enabled=amp):
                    pred = net(x, edge_index, batch_idx)

                    if net.n_classes == 1:
                        pred = pred.squeeze(dim=-1)

                    # if not pretrain:
                    #     pred.requires_grad = True
                    loss = criterion(pred, batch.y)


                # l1_norm = 0
                # for layer in [net.medium_spring_1, net.medium_spring_2, net.up_2, net.large_spring_1,
                #               net.large_spring_2, net.fc1, net.fc2]:
                #     l1_norm += l1_lambda * sum(p.abs().sum() for p in layer.parameters())
                #
                # loss += l1_norm

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(x.shape[0])
                global_step += 1
                train_loss_total += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Monitoring model parameters
            histograms = {}
            for tag, value in net.named_parameters():
                tag = tag.replace('/', '.')
                if value.grad is not None:
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            net.eval()
            with tqdm(total=n_val, desc=f'Epoch {epoch + 1}/{epochs}', unit='mol') as pbar:
                count = 0
                with torch.no_grad():
                    val_sensitivity_total, val_specificity_total = 0, 0
                    for batch in iter(val_loader):
                        batch = batch.to(device=device)
                        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

                        with torch.cuda.amp.autocast(enabled=amp):
                            preds = net(x, edge_index, batch_idx)

                            if net.n_classes == 1:
                                preds = preds.squeeze(dim=-1)

                            val_loss = criterion(preds, batch.y)

                        # l1_norm = 0
                        # for layer in [net.medium_spring_1, net.medium_spring_2, net.up_2, net.large_spring_1,
                        #               net.large_spring_2, net.fc1, net.fc2]:
                        #     l1_norm += l1_lambda * sum(p.abs().sum() for p in layer.parameters())

                        val_loss_total += val_loss.item()

                        preds = (torch.sigmoid(preds).data.float() > 0.5).float().cpu().numpy()
                        if count == 0:
                            pred, true = preds, batch.y.data.cpu()
                            count += 1
                        else:
                            pred = np.concatenate((pred, preds), axis=0)
                            true = torch.cat([true, batch.y.data.cpu()])

                        try:
                            sensitivity = classification_report(y_true=batch.y.cpu().detach().numpy(),
                                                                y_pred=preds, output_dict=True)[
                                '1.0']['recall']
                        except KeyError:
                            sensitivity = 0

                        try:
                            specificity = classification_report(y_true=batch.y.cpu().detach().numpy(),
                                                                y_pred=preds, output_dict=True)[
                                '0.0']['recall']
                        except KeyError:
                            specificity = 0

                        val_sensitivity_total += sensitivity
                        val_specificity_total += specificity

            experiment.log({
                'train loss': train_loss_total/len(train_loader),
                'val loss': val_loss_total/len(val_loader),
                'learning rate': optimizer.param_groups[0]['lr'],
                'val sensitivity': val_sensitivity_total/len(val_loader),
                'val specificity': val_specificity_total/len(val_loader),
                **histograms
            })
            val_sensitivities.append(val_sensitivity_total/len(val_loader))
            val_specificities.append(val_specificity_total/len(val_loader))
            if net.trial:
                net.trial.report(f1b_score(sum(val_sensitivities)/len(val_sensitivities),
                                           sum(val_specificities)/len(val_specificities)), epoch)
                if net.trial.should_prune():
                    raise optuna.TrialPruned()

        # scheduler.step()
        epochs_since_improvement += 1
        if save_checkpoint and val_loss < prev_best:
            epochs_since_improvement = 0
            prev_best = val_loss
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            best_model = net
            torch.save(net.state_dict(), f'{dir_checkpoint}/checkpoint.pth')
            logging.info(f'Checkpoint {epoch + 1} saved!')

        if epochs_since_improvement > early_stopping_patience:
            break

    best_model.load_state_dict(torch.load(f'{dir_checkpoint}/checkpoint.pth', map_location=device))
    best_model.eval()

    with torch.no_grad():
        test_sensitivity_total, test_specificity_total = 0, 0

        count = 0
        for batch in iter(test_loader):
            batch = batch.to(device=device)
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

            with torch.cuda.amp.autocast(enabled=amp):
                preds = best_model(x, edge_index, batch_idx)

                if net.n_classes == 1:
                    preds = preds.squeeze(dim=-1)

            test_loss = criterion(preds, batch.y)
            test_loss_total += test_loss.item()

            preds = (torch.sigmoid(preds).data.float() > 0.5).float().cpu().numpy()
            if count == 0:
                pred, true = preds, batch.y.data.cpu()
                count += 1
            else:
                pred = np.concatenate((pred, preds), axis=0)
                true = torch.cat([true, batch.y.data.cpu()])

            try:
                sensitivity = classification_report(y_true=batch.y.cpu().detach().numpy(),
                                                    y_pred=preds, output_dict=True)[
                    '1.0']['recall']
            except KeyError:
                sensitivity = 0

            try:
                specificity = classification_report(y_true=batch.y.cpu().detach().numpy(),
                                                    y_pred=preds, output_dict=True)[
                    '0.0']['recall']
            except KeyError:
                specificity = 0

            test_sensitivity_total += sensitivity
            test_specificity_total += specificity

        experiment.log({
            'test loss': test_loss_total / len(test_loader),
            'test sensitivity': test_sensitivity_total / len(test_loader),
            'test specificity': test_specificity_total / len(test_loader)
        })
    return test_sensitivity_total / len(test_loader), test_specificity_total / len(test_loader)