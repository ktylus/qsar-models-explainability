import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data


# from class notebooks
class Featurizer:
    def __init__(self, y_col, smiles_col="Drug", log_target_transform=False, **kwargs):
        self.y_col = y_col
        self.smiles_col = smiles_col
        self.log_target_transform = log_target_transform
        self.__dict__.update(kwargs)
    
    def __call__(self, df):
        raise NotImplementedError()


# from class notebooks
class ECFPFeaturizer(Featurizer):
    def __init__(self, y_col, radius=2, length=1024, **kwargs):
        self.radius = radius
        self.length = length
        super().__init__(y_col, **kwargs)
    
    def __call__(self, molecules: list):
        fingerprints = []
        targets = []
        for mol in molecules:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.length)
            fingerprints.append(fp)
            targets.append(mol.GetProp(self.y_col))
        fingerprints = np.array(fingerprints)
        targets = np.array(targets).astype(np.float64)
        if self.log_target_transform:
            targets = np.log(targets + 1e-8)
        return fingerprints, targets


# from class notebooks
class GraphFeaturizer(Featurizer):
    def __call__(self, molecules: list):
        graphs = []
        labels = []
        for mol in molecules:
            edges = []
            for bond in mol.GetBonds():
                edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
            edges = np.array(edges)
            
            nodes = []
            for atom in mol.GetAtoms():
                atom_encoding = self.one_of_k_encoding_unk(atom.GetSymbol(), ["C", "N", "O", "S", "Cl", "Br", "F", "I", "P", "unknown"])
                hydrogen_atoms = Chem.Atom.GetTotalNumHs(atom)
                results = [*atom_encoding, hydrogen_atoms]
                nodes.append(results)
            nodes = np.array(nodes)

            graphs.append((nodes, edges.T))
            labels.append(mol.GetProp(self.y_col))
        labels = np.array(labels).astype(np.float64)
        if self.log_target_transform:
            labels = np.log(labels + 1e-8)
        return [Data(
            x=torch.FloatTensor(x), 
            edge_index=torch.LongTensor(edge_index), 
            y=torch.FloatTensor([y])
        ) for ((x, edge_index), y) in zip(graphs, labels)]
    
    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise ValueError("input {0} not in allowable set{1}:".format(
                x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))