import copy

import rdkit.Chem as Chem
import numpy as np
import torch


def create_synthetic_dataset(smiles_for_class_task, n_samples=10000):
    with open("data/ChEMBL_filtered.txt", "r") as f:
        smiles = [next(f).strip() for _ in range(n_samples)]
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles]
    # Set target values.
    targets = create_synthetic_target(molecules, smiles_for_class_task)
    for i, mol in enumerate(molecules):
        mol.SetProp("y", str(targets[i]))
    return molecules


def create_synthetic_target(
        data,
        smiles_for_class_task,
):
    y_synthetic = []
    for mol in data:
        target_value = 0
        if mol.HasSubstructMatch(Chem.MolFromSmiles(smiles_for_class_task)):
            target_value = 1
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
