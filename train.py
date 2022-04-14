import argparse
import logging
import sys
from pathlib import Path
import os

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


def setup_training(config: dict, save_checkpoint: str, df: pd.DataFrame, wandb_name: str, net: nn.Module = None):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if net is None:
        net = GraphGNNModel(79, config['c_hidden_soft'], config['c_hidden_soft'], dp_rate_linear=config['drop_rate_soft_dense'],
                            dp_gnn=config['drop_rate_soft'],**config)

    os.mkdir(f'./outputs/{wandb_name}')

    experiment = wandb.init(project='TP-GNNs', resume='allow', name=wandb_name,
                            config={
                                "learning_rate": config['lr'],
                                "batch_size": config['batch_size'],
                                "epochs": 250,
                                "save_checkpoint": save_checkpoint,
                            })

    train_net(net, device, epochs=10, batch_size=config['batch_size'], learning_rate=config['lr'], experiment=experiment,
              df=df, opt=config['optim'], dir_checkpoint=f'./outputs/{wandb_name}', pos_weight=config['pos_weight'])


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
              l1_lambda: float = 0):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # experiment.watch(net, log_freq=10, log='all')

    data_list = create_data_list(df['Smiles'], df['Phototoxic'])
    train, test = train_test_split(data_list, test_size=0.1, stratify=[t.y for t in data_list])
    train, valid = train_test_split(train, test_size=0.15, stratify=[t.y for t in train])

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
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    elif opt == 'RMSProp':
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
    elif opt == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)

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
    for epoch in range(epochs):
        train_loss_total = 0
        val_loss_total = 0
        test_loss_total = 0
        net.train()
        epoch_score = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='mol') as pbar:
            for batch in iter(train_loader):
                x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

                with torch.cuda.amp.autocast(enabled=amp):
                    pred = net(x, edge_index, batch_idx)

                    if net.n_classes == 1:
                        pred = pred.squeeze(dim=-1)

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

                        preds = (torch.sigmoid(preds).data.float() > 0.5).float()
                        if count == 0:
                            pred, true = preds, batch.y
                            count += 1
                        else:
                            pred = np.concatenate((pred, preds), axis=0)
                            true = torch.cat([true, batch.y])

                        try:
                            sensitivity = classification_report(y_true=batch.y.cpu().detach().numpy(),
                                                                y_pred=preds.cpu().detach().numpy(), output_dict=True)[
                                '1.0']['recall']
                        except KeyError:
                            sensitivity = 0

                        try:
                            specificity = classification_report(y_true=batch.y.cpu().detach().numpy(),
                                                                y_pred=preds.cpu().detach().numpy(), output_dict=True)[
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

        # scheduler.step()

        if save_checkpoint and val_loss < prev_best:
            prev_best = val_loss
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            best_model = net
            torch.save(net.state_dict(), f'{dir_checkpoint}/checkpoint_epoch{epoch+1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved!')

    best_model.eval()

    with torch.no_grad():
        test_sensitivity_total, test_specificity_total = 0, 0

        count = 0
        for batch in iter(test_loader):
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

            with torch.cuda.amp.autocast(enabled=amp):
                preds = net(x, edge_index, batch_idx)

                if net.n_classes == 1:
                    preds = preds.squeeze(dim=-1)

            test_loss = criterion(preds, batch.y)
            test_loss_total += test_loss.item()

            preds = (torch.sigmoid(preds).data.float() > 0.5).float()
            if count == 0:
                pred, true = preds, batch.y
                count += 1
            else:
                pred = np.concatenate((pred, preds), axis=0)
                true = torch.cat([true, batch.y])

            try:
                sensitivity = classification_report(y_true=batch.y.cpu().detach().numpy(),
                                                    y_pred=preds.cpu().detach().numpy(), output_dict=True)[
                    '1.0']['recall']
            except KeyError:
                sensitivity = 0

            try:
                specificity = classification_report(y_true=batch.y.cpu().detach().numpy(),
                                                    y_pred=preds.cpu().detach().numpy(), output_dict=True)[
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