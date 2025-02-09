import copy
import requests

from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors, rdFingerprintGenerator
import torch
import torch_geometric as tg

from src.featurizers import GraphFeaturizer
from src.models.gnn import GraphConvolutionalNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_morgan_bit(molecules, bit_id, radius=2, length=2048):
    for mol in molecules:
        additional_output = rdFingerprintGenerator.AdditionalOutput()
        additional_output.AllocateBitInfoMap()
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=length)
        fp = fp_gen.GetFingerprint(mol, additionalOutput=additional_output)
        if bit_id in additional_output.GetBitInfoMap():
            break
    return Draw.DrawMorganBit(mol, bit_id, additional_output.GetBitInfoMap())


def draw_many_morgan_bits(molecules, bit_ids, radius=2, length=2048):
    results = []
    for bit_id in bit_ids:
        for mol in molecules:
            additional_output = rdFingerprintGenerator.AdditionalOutput()
            additional_output.AllocateBitInfoMap()
            fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=length)
            fp = fp_gen.GetFingerprint(mol, additionalOutput=additional_output)
            if bit_id in additional_output.GetBitInfoMap():
                break
        results.append((mol, bit_id, additional_output.GetBitInfoMap()))
    return Draw.DrawMorganBits(results, molsPerRow=5, legends=[str(bit_id) for bit_id in bit_ids])


def get_iupac_name_of_smiles(smiles_string):
    """
    Returns the IUPAC name of a compound given its SMILES string.

    Parameters:
    smiles_string (str): The SMILES string of the compound.

    Returns:
    str: The IUPAC name of the compound.
    """
    # Search PubChem for the compound.
    try:
        compound = pcp.get_compounds(smiles_string, 'smiles')
        found = True
    except Exception as e:
        found = False
    if found:
        iupac_name = compound[0].iupac_name if compound else 'Not found'
    else:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles_string}/iupac_name"
        response = requests.get(url)
        response.raise_for_status()
        iupac_name = response.text
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
    batch_copy.y = 1 - batch_copy.y
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


def load_gnn_model(data, dataset_name, best_params):
    graph_featurizer = GraphFeaturizer(y_col="y", log_target_transform=False)
    graph_test = graph_featurizer(data)

    model_path = f"models/gnn_tuned_{dataset_name}.pth"
    model = GraphConvolutionalNetwork(
        input_dim=graph_test[0].x.shape[1],
        hidden_size=best_params["hidden_size"],
        n_layers=best_params["num_layers"],
        dropout=best_params["dropout"]
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    return model


def load_catboost_model(dataset_name, best_params):
    model = CatBoostClassifier(
        iterations=9999,
        learning_rate=best_params["lr"],
        max_depth=best_params["max_depth"],
        l2_leaf_reg=best_params["l2_leaf_reg"]
    )
    model = model.load_model(f"models/catboost_tuned_{dataset_name}.cbm")
    return model


def get_sub_molecule(mol, atom_indices):
    # Create an editable molecule
    editable_mol = Chem.EditableMol(Chem.Mol())

    # Map old atom indices to new atom indices
    old_to_new = {}
    for i, atom_idx in enumerate(atom_indices):
        atom = mol.GetAtomWithIdx(atom_idx)
        new_idx = editable_mol.AddAtom(atom)
        old_to_new[atom_idx] = new_idx

    # Add bonds between the new atoms
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in old_to_new and end_idx in old_to_new:
            editable_mol.AddBond(old_to_new[begin_idx], old_to_new[end_idx], bond.GetBondType())

    # Get the new molecule
    sub_mol = editable_mol.GetMol()
    return sub_mol