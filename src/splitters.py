import deepchem as dc
import numpy as np
from rdkit import Chem


class ScaffoldSplitter:
    def __init__(self):
        self.splitter = dc.splits.ScaffoldSplitter()

    def train_test_split(self, X, y, smiles):
        dc_dataset = dc.data.DiskDataset.from_numpy(X, y, ids=smiles)
        train, test = self.splitter.train_test_split(dc_dataset)
        return train.X, test.X, train.y, test.y
    
    
    def train_test_molecules_split(self, data, target_name):
        X = np.zeros(len(data))
        y = []
        for mol in data:
            y.append(mol.GetProp(target_name))
        smiles = [Chem.MolToSmiles(mol) for mol in data]
        dc_dataset = dc.data.DiskDataset.from_numpy(X, y, ids=smiles)
        train, test = self.splitter.train_test_split(dc_dataset)

        train_molecules = [Chem.MolFromSmiles(smiles) for smiles in train.ids]
        for i, mol in enumerate(train_molecules):
            mol.SetProp(target_name, train.y[i])
        test_molecules = [Chem.MolFromSmiles(smiles) for smiles in test.ids]
        for i, mol in enumerate(test_molecules):
            mol.SetProp(target_name, test.y[i])        
        return train_molecules, test_molecules
    