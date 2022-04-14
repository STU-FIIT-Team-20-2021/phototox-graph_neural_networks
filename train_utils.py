from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torch import nn

from models import GraphLevelGNN


def train_graph_classifier(model_name, tr_loader, te_loader, v_loader, model=None, config=None, **model_kwargs):
    pl.seed_everything(42)

    if model is None:
        model = GraphLevelGNN(c_in=79, c_out=1, config=config, **model_kwargs)

    callbacks = [ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                 EarlyStopping(monitor="val_loss", mode="min")]

    if model:
        callbacks = callbacks.append(TuneReportCallback(
            {
                "loss": "val_loss",
                "mean_accuracy": "val_acc",
                "sensitivity": "val_sensitivity",
                "specificity": "val_specificity"
            },
            on="validation_end"))

    root_dir = os.path.join('.', "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=callbacks,
                         accelerator="gpu",
                         devices=1,
                         max_epochs=200,
                         progress_bar_refresh_rate=0)
    trainer.logger._default_hp_metric = None

    pl.seed_everything(42)
    print(model)

    trainer.fit(model, tr_loader, v_loader)
    model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    print('---------')
    print(f'Testing {config}')
    print('---------')
    train_result = trainer.test(model, dataloaders=tr_loader, verbose=False)
    val_result = trainer.test(model, dataloaders=v_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=te_loader, verbose=False)

    tune.report(test_loss=test_result[0]['test_loss'], test_acc=test_result[0]['test_acc'], test_sensitivity=test_result[0]['test_sensitivity'], test_specificity=test_result[0]['test_specificity'],
                val_loss=val_result[0]['test_loss'], val_acc=val_result[0]['test_acc'], val_sensitivity=val_result[0]['test_sensitivity'], val_specificity=val_result[0]['test_specificity'],)

    result = {"test_acc": test_result[0]['test_acc'],
              "test_sensitivity": test_result[0]['test_sensitivity'],
              "test_specificity": test_result[0]['test_specificity'],
              "train_acc": train_result[0]['test_acc'],
              "train_sensitivity": train_result[0]['test_sensitivity'],
              "train_specificity": train_result[0]['test_specificity']}
    return model, result, trainer


def train_model_both(config, train_loader, test_loader, val_loader, strong_train_loader, strong_val_loader, strong_test_loader):

    model, result, trainer = train_graph_classifier(model_name="Tune",
                                                    model=None,
                                                    config=config,
                                                    tr_loader=train_loader,
                                                    te_loader=test_loader,
                                                    v_loader=val_loader,
                                                    c_hidden=config['c_hidden_soft'],
                                                    layer_name=config['type'],
                                                    num_layers=config['layers_soft'],
                                                    dp_rate_linear=config['drop_rate_soft_dense'],
                                                    dp_rate=config['drop_rate_soft'])

    new_head = nn.Sequential(
        nn.Dropout(config['drop_rate_hard_dense']),
        nn.Linear(config['c_hidden_soft'], 1)
    )

    for param in model.parameters():
        param.requires_grad = False

    list(model.children())[0].head = new_head

    model, result, trainer = train_graph_classifier(model_name="Tune_strong",
                                                    model=model,
                                                    tr_loader=strong_train_loader,
                                                    te_loader=strong_test_loader,
                                                    v_loader=strong_val_loader,
                                                    c_hidden=config['c_hidden_soft'],
                                                    layer_name=config['type'],
                                                    num_layers=config['layers_soft'],
                                                    dp_rate_linear=config['drop_rate_soft_dense'],
                                                    dp_rate=config['drop_rate_soft'])
    return model, result, trainer