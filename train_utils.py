import torch
import torch.nn as nn
import optuna
import wandb
import pandas as pd

from models import GraphGNNModel
from train import setup_training, f1b_score

df = pd.read_csv('./mutagenicity.csv', sep=',')
# df = df.drop([345, 346], axis=0)
# df = df.drop_duplicates(subset=["Smiles"], keep=False)

# pre_df = pd.read_csv('./10k_smiles_scored.csv', sep='\t')
# pre_df = pre_df.drop_duplicates(subset=["Smiles"], keep=False)

df = df[['Phototoxic', 'Smiles']]
# pre_df = pre_df[['Phototoxic', 'Smiles']]



# Specify this in setup to run an experiment without pretraining on weakly annotated data
def train_no_pretrain(trial: optuna.trial.Trial, config: dict):
    name = f"Mutagenicity__chs={config['c_hidden_soft']}_ls={config['layers_soft']}_drsd={config['drop_rate_soft_dense']}_" \
           f"drs={config['drop_rate_soft']}_drhd1={config['drop_rate_hard_dense_1']}_drhd2={config['drop_rate_hard_dense_2']}_" \
           f"dih={config['dense_input_head']}_dihidden={config['dense_input_hidden']}_optim={config['optim']}_" \
           f"lr={config['lr']}_batchsize={config['batch_size']}_posweight={config['pos_weight']}"
    net = GraphGNNModel(80, config['c_hidden_soft'], config['c_hidden_soft'],
                        dp_rate_linear=config['drop_rate_soft_dense'],
                        dp_gnn=config['drop_rate_soft'], **config)

    net.start_trial(trial)

    try:
        sensitivity, specificity = setup_training(config, f'./outputs/mutagenicity/Strong_{name}', df,
                                                  wandb_name='Strong_'+name, net=net)
        wandb.finish()
    except optuna.TrialPruned:
        raise optuna.TrialPruned()

    return sensitivity, specificity



# Default optimization mode
# The model is first pre-trained by letting it run for 3 epochs on the weakly annotated dataset, then we the LR is
# increased by a factor of 10 and the training is finished on the strongly annotated data
def train_model_both(trial: optuna.trial.Trial, config: dict):
    name = f"GATv2_final_f1b__chs={config['c_hidden_soft']}_ls={config['layers_soft']}_drsd={config['drop_rate_soft_dense']}_" \
           f"drs={config['drop_rate_soft']}_drhd1={config['drop_rate_hard_dense_1']}_drhd2={config['drop_rate_hard_dense_2']}_" \
           f"dih={config['dense_input_head']}_dihidden={config['dense_input_hidden']}_optim={config['optim']}_" \
           f"lr={config['lr']}_batchsize={config['batch_size']}_posweight={config['pos_weight']}"
    setup_training(config, f'./outputs/final_f1b/{name}', pre_df, wandb_name=name, pretrain=True)
    wandb.finish()
    last = f'./outputs/final_f1b/{name}/checkpoint.pth'
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

    # Not freezing the graph layers worked out better than just replacing the head
    # for param in net.parameters():
    #     param.requires_grad = False

    list(net.children())[0].head = new_head
    net.start_trial(trial)

    try:
        sensitivity, specificity = setup_training(config, f'./outputs/final_f1b/Strong_{name}', df,
                                                  wandb_name='Strong_'+name, net=net)
        wandb.finish()
    except optuna.TrialPruned:
        raise optuna.TrialPruned()

    return sensitivity, specificity


# Optuna controller - finds best parameters from specified ranges - also optimizes number of layers
# Trials are pruned based on f1b_score if optuna doesn't think the run will result in an improvement, to reduce
# training time
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
        'pos_weight': trial.suggest_uniform('pos_weight', 1.0, 1.7),
        'optim': trial.suggest_categorical('optim', ["Adam", "RMSprop", "SGD"]),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    }

    sensitivity, specificity = train_no_pretrain(trial, config)

    return f1b_score(sensitivity, specificity)