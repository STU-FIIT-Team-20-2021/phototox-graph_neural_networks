from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.rdMolDescriptors as Mol
import rdkit.Chem.GraphDescriptors as Graph
import rdkit.Chem.Crippen as Crip
import rdkit.Chem.Descriptors as Desc
import numpy as np
import pandas as pd
from math import log


def populate(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        raise ValueError("The Molecule can't be created. Either the Smiles is malformed or it's a RDKit issue.")

    try:
        mol = Chem.AddHs(mol)
    except:
        raise ValueError("The Molecule can't have Hydrogens added to it. Smiles is probably malformed.")

    df = {}

    AllChem.ComputeGasteigerCharges(mol)

    cycles = len(mol.GetRingInfo().BondRings())
    count_valence = sum([x.GetExplicitValence() for x in mol.GetAtoms()])
    gesteiger_charges = [float(x.GetProp("_GasteigerCharge")) for x in mol.GetAtoms()]
    gesteiger_charges = np.nan_to_num(gesteiger_charges)
    positive_gesteiger_sum = np.sum([y for y in gesteiger_charges if y >= 0])
    positive_gesteiger_sum = np.nan_to_num(positive_gesteiger_sum)
    negative_gesteiger_sum = np.sum([y for y in gesteiger_charges if y < 0])
    negative_gesteiger_sum = np.nan_to_num(negative_gesteiger_sum)
    positive_gesteiger_mean = np.mean([y for y in gesteiger_charges if y >= 0])
    positive_gesteiger_mean = np.nan_to_num(positive_gesteiger_mean)
    negative_gesteiger_mean = np.mean([y for y in gesteiger_charges if y < 0])
    negative_gesteiger_mean = np.nan_to_num(negative_gesteiger_mean)
    total_gesteiger_sum = np.sum([y for y in gesteiger_charges]) * 10 ** 15
    total_gesteiger_sum = np.nan_to_num(total_gesteiger_sum)
    total_gesteiger_mean = np.mean([y for y in gesteiger_charges]) * 10 ** 15
    total_gesteiger_mean = np.nan_to_num(total_gesteiger_mean)
    hydrogen_count = sum([x.GetAtomicNum() for x in mol.GetAtoms() if x.GetAtomicNum() == 1])
    carbon_count = sum([x.GetAtomicNum() for x in mol.GetAtoms() if x.GetAtomicNum() == 6])
    csp3 = Mol.CalcFractionCSP3(mol)
    aha = len([x for x in mol.GetAtoms() if x.GetAtomicNum() != 1 and x.GetIsAromatic()])
    drug = sum([0 if -0.4 <= Crip.MolLogP(mol, True) <= 5.6 else 1,
                 0 if 160 <= Desc.ExactMolWt(mol) <= 480 else 1,
                 0 if 20 <= len(mol.GetAtoms()) <= 70 else 1,
                 0 if 40 <= Crip.MolMR(mol) <= 130 else 1,
                 0 if Mol.CalcNumRotatableBonds(mol) <= 10 else 1])
    kappa = Mol.CalcKappa1(mol)
    balban_j = Graph.BalabanJ(mol)
    hallkier = Mol.CalcHallKierAlpha(mol)
    labute_asa = Mol.CalcLabuteASA(mol)
    h_donors = Mol.CalcNumHBD(mol)
    h_acceptors = Mol.CalcNumHBA(mol)
    tpsa = Mol.CalcTPSA(mol)
    vsa = np.array(Mol.SlogP_VSA_(mol))

    df['cycles'] = cycles
    df['atom_valence'] = count_valence
    df['negative_gesteiger_sum'] = negative_gesteiger_sum
    df['positive_gesteiger_sum'] = positive_gesteiger_sum
    df['negative_gesteiger_mean'] = negative_gesteiger_mean
    df['positive_gesteiger_mean'] = positive_gesteiger_mean
    df['sum_gesteiger_fromsum'] = (df['negative_gesteiger_sum'] + df['positive_gesteiger_sum']) * 10 ** 15
    df['sum_gesteiger_frommean'] = (df['negative_gesteiger_mean'] + df['positive_gesteiger_mean'])
    df['sum_gesteiger_totalsum'] = total_gesteiger_sum
    df['sum_gesteiger_totalmean'] = total_gesteiger_mean
    df['hydrogen_count'] = hydrogen_count
    df['csp3'] = csp3
    df['balban_j'] = balban_j
    df['aromatic_heavy_atoms'] = aha
    df['hallkier'] = hallkier
    df['kappa_1'] = kappa
    df['labute_asa'] = labute_asa
    df['h_donors'] = h_donors
    df['h_acceptors'] = h_acceptors
    df['carbon_count'] = carbon_count
    df['tpsa'] = tpsa
    df['druglikeliness'] = drug

    for i, vsa_vals in enumerate(vsa.transpose()):
        df[f'vsa_logP_{i}'] = vsa_vals

    # rings = mol.GetRingInfo().BondRings()
    # binarystr = '0000000000000000'
    #
    # #TODO Fix issue if len(cycle) > 16
    # for cycle in rings:
    #     binarystr = binarystr[:16 - (len(cycle) - 2)] + '{:x}'.format(
    #         int(binarystr[16 - (len(cycle) - 2)], 16) + 1) + binarystr[16 - (len(cycle) - 3):]
    #
    # converted = int(binarystr, 16)

    # df['cycle_type_counts'] = 0 if converted == 0 else log(converted, 16)

    return pd.DataFrame.from_dict(df, orient='index').transpose()


if __name__ == "__main__":
    df = pd.DataFrame([{'Phototoxic': 1, "Smiles": "[Ti](=O)=O"}])
    populate(df)
    print(df)