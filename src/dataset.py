import copy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import rdkit.Chem as Chem
import numpy as np
from tdc.single_pred import ADME, Tox
import torch

from src.splitters import ScaffoldSplitter
from src.featurizers import GraphFeaturizer


def create_synthetic_dataset(smiles_for_class_task="c1ccccc1", n_samples=1000):
    with open("data/ChEMBL_filtered.txt", "r") as f:
        smiles = [next(f).strip() for _ in range(n_samples)]
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles]
    # Set target values.
    targets = create_synthetic_target(molecules, smiles_for_class_task)
    for i, mol in enumerate(molecules):
        mol.SetProp("y", str(targets[i]))
    return molecules


def create_synthetic_target(data, smiles_for_class_task):
    y_synthetic = []
    for mol in data:
        target_value = 0
        if mol.HasSubstructMatch(Chem.MolFromSmiles(smiles_for_class_task)):
            target_value = 1
        y_synthetic.append(target_value)
    return np.array(y_synthetic)


def get_graph_data_with_substituted_target(data, target):
    data_copy = copy.deepcopy(data)
    for i in range(len(data_copy)):
        data_copy[i].y = torch.Tensor([target[i]])
    return data_copy


def load_synthetic_data_split():
    data = create_synthetic_dataset()
    splitter = ScaffoldSplitter()
    train, test = splitter.train_test_molecules_split(data, "y")
    train, val = splitter.train_test_molecules_split(train, "y")
    return train, val, test


def load_cyp_data_split():
    data = ADME(name='CYP3A4_Veith')
    train, valid, test = get_tdc_data_split_components(data.get_split())
    return train, valid, test


def load_herg_data_split():
    data = Tox(name='hERG')
    train, valid, test = get_tdc_data_split_components(data.get_split())
    return train, valid, test


def load_pampa_data_split():
    data = ADME(name='PAMPA_NCATS')
    train, valid, test = get_tdc_data_split_components(data.get_split())
    return train, valid, test


def get_tdc_data_split_components(tdc_data):
    train, valid, test = tdc_data['train'], tdc_data['valid'], tdc_data['test']
    train = create_mol_data_from_tdc_data(train)
    valid = create_mol_data_from_tdc_data(valid)
    test = create_mol_data_from_tdc_data(test)
    return train, valid, test


def create_mol_data_from_tdc_data(tdc_data):
    mols = []
    for _, row in tdc_data.iterrows():
        mol = Chem.MolFromSmiles(row['Drug'])
        mol.SetProp('y', str(row['Y']))
        mols.append(mol)
    return mols
