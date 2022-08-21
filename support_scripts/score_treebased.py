import pandas as pd
import numpy as np
import joblib

from functools import partial
from multiprocessing import Pool
from rfc_chem import populate
from rdkit import Chem
from rdkit.Chem import AllChem


def get_pseudolabel(smiles, model):
    descriptors = populate(smiles)
    pseudo_label = model.predict(descriptors)
    return pseudo_label


def score(rf_model, lgb_model, smiles):
    try:
        return get_pseudolabel(smiles, rf_model)[0], get_pseudolabel(smiles, lgb_model)[0], smiles
    except ValueError:
        return np.nan, np.nan, np.nan


if __name__ == '__main__':
    mol_df = pd.read_csv('F:/Halinkovic/asgn/data/mutagenicity_no_labels/raw/support.csv', delimiter=';')
    mol_df = mol_df.loc[[len(sm) < 100 if sm is not np.nan else False for sm in mol_df['Smiles']]].sample(n=100000, random_state=42)

    lgb_model = joblib.load('./mutagenicity_lgbm.joblib')
    rf_model = joblib.load('./mutagenicity_rf.joblib')

    rf_labels = []
    lgb_labels = []
    out_smiles = []

    partial_score = partial(score, rf_model, lgb_model)

    with Pool(22) as p:
        results = np.array(p.map(partial_score, mol_df['Smiles'].values))

    # for smiles in mol_df['Smiles']:
    #     try:
    #         rf_labels.append(get_pseudolabel(smiles, rf_model))
    #         lgb_labels.append(get_pseudolabel(smiles, lgb_model))
    #         out_smiles.append(smiles)
    #     except ValueError:
    #         return np.nan, np.nan, np.nan

    out_df = pd.DataFrame(results, columns=['RF_label', 'LGB_label', 'Smiles'])
    out_df.to_csv('./mutagenicity_pseudolabels.csv', index=False)

