import copy

import rdkit.Chem as Chem
import numpy as np
import torch


def create_synthetic_dataset(n_samples=10000):
    with open("data/ChEMBL_filtered.txt", "r") as f:
        smiles = [next(f).strip() for _ in range(n_samples)]
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles]
    # Set target values.
    targets = create_synthetic_target(molecules, {"c1ccccc1": (1, 0)}, default_mean=0.0, default_std=0.0)
    for i, mol in enumerate(molecules):
        mol.SetProp("y", str(targets[i]))
    return molecules


def create_synthetic_target(
        data,
        mol_mean_and_std,
        default_mean=0.0,
        default_std=1.0
):
    y_synthetic = []
    for mol in data:
        target_value = 0.0
        for mol_to_check in mol_mean_and_std:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(mol_to_check)):
                target_value += np.random.normal(loc=mol_mean_and_std[mol_to_check][0],
                                                 scale=mol_mean_and_std[mol_to_check][1])
        if target_value == 0.0:
            target_value = np.random.normal(loc=default_mean, scale=default_std)
        y_synthetic.append(target_value)
    return np.array(y_synthetic)


def get_graph_data_with_substituted_target(
        data,
        target
):
    data_copy = copy.deepcopy(data)
    for i in range(len(data_copy)):
        data_copy[i].y = torch.Tensor([target[i]])
    return data_copy
