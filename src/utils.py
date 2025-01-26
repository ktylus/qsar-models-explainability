import copy

import matplotlib.pyplot as plt
import pubchempy as pcp
from sklearn.tree import DecisionTreeRegressor, plot_tree
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
import torch_geometric as tg


def save_decision_tree_graph(model: DecisionTreeRegressor, filename: str):
    assert filename.split(".")[-1] in ["png", "pdf"]
    plt.figure(figsize=(20, 10))
    plot_tree(model)
    plt.savefig(filename)
    plt.close()


def draw_morgan_bit_many_molecules(molecules: list, bit_id, radius=2, length=1024):
    target_mols = []
    bit_infos = []
    mols_found = 0
    for i, mol in enumerate(molecules):
        if mols_found == 10:
            break
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=length, radius=radius, bitInfo=bit_info)
        if fp[bit_id]:
            mols_found += 1
            target_mols.append(mol)
            bit_infos.append(bit_info)
    imgs = []
    for mol, bit_info in zip(target_mols, bit_infos):
        imgs.append(Draw.DrawMorganBit(mol, bit_id, bit_info))
    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    for i, img in enumerate(imgs[:10]):
        ax[i // 5, i % 5].imshow(img)
        ax[i // 5, i % 5].axis('off')
        ax[i // 5, i % 5].set_title(f'mol {i}')


def load_synthetic_target_data():
    with open("data/ChEMBL_filtered.txt", "r") as f:
        smiles = f.readlines()
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles]
    return molecules


def get_iupac_name_of_smiles(smiles_string):
    """
    Returns the IUPAC name of a compound given its SMILES string.

    Parameters:
    smiles_string (str): The SMILES string of the compound.

    Returns:
    str: The IUPAC name of the compound.
    """
    # Search PubChem for the compound.
    compound = pcp.get_compounds(smiles_string, 'smiles')
    iupac_name = compound[0].iupac_name if compound else 'Not found'
    return iupac_name


def get_fingerprint_with_zeroed_top_bits(X, top_bits):
    X_inverted = copy.deepcopy(X)
    X_inverted[:, top_bits] = 0
    return X_inverted


def get_morgan_fragment(molecule, bit, radius=2, nBits=2048):
    """
    Returns the molecule fragment corresponding to a given Morgan fingerprint bit.

    Parameters:
    molecule (rdkit.Chem.rdchem.Mol): The molecule to analyze.
    bit (int): The bit of the Morgan fingerprint.
    radius (int): The radius of the Morgan fingerprint. Default is 2.
    nBits (int): The size of the Morgan fingerprint. Default is 2048.

    Returns:
    rdkit.Chem.rdchem.Mol: The fragment corresponding to the given bit.
    """
    # Generate the Morgan fingerprint
    info = {}
    _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, radius, nBits=nBits, bitInfo=info)

    # Check if the bit is in the fingerprint
    if bit not in info:
        raise ValueError(f"Bit {bit} is not present in the fingerprint.")

    # Get the atom indices and radius for the bit
    atom_indices, bit_radius = info[bit][0]

    # Get the fragment corresponding to the bit
    if bit_radius == 0:
        frag = Chem.MolFromSmarts(molecule.GetAtomWithIdx(atom_indices).GetSmarts())
    else:
        env = Chem.FindAtomEnvironmentOfRadiusN(molecule, bit_radius, atom_indices)
        amap = {}
        frag = Chem.PathToSubmol(molecule, env, atomMap=amap)
    return frag


def get_graph_batch_with_inverted_target(batch):
    batch_copy = copy.deepcopy(batch)
    batch_copy.y = (~batch_copy.y.bool()).float()
    return batch_copy


def min_max_scale_graph_batch(data, batch):
    max_exp_values = tg.utils.scatter(data, batch, dim=0, reduce="max")
    max_exp_values_expanded = max_exp_values[batch]
    max_exp_values_expanded[max_exp_values_expanded == 0] = 1
    return data / max_exp_values_expanded


def get_data_partition_on_substructure_presence(data, substructure_smiles):
    has_substructure = []
    no_substructure = []
    for mol in data:
        if mol.HasSubstructMatch(Chem.MolFromSmiles(substructure_smiles)):
            has_substructure.append(mol)
        else:
            no_substructure.append(mol)
    return has_substructure, no_substructure
