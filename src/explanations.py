from collections import deque
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lime import lime_tabular
import torch
import torch.nn.functional as F
import torch_geometric as tg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import rdkit.Chem as Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
import matplotlib.cm as cm
from skimage.io import imread
# a way to make it work on Windows, it's very bad, and I had to install UniConvertor
os.environ["path"] += r";C:\\Program Files\\UniConvertor-2.0rc5\dlls"
from cairosvg import svg2png

from src.utils import is_duplicate_in_list_of_lists

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# from https://github.com/ndey96/GCNN-Explainability/blob/master/explain.py
def img_for_mol(mol, atom_weights):
    highlight_kwargs = {}
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    cmap = cm.get_cmap('bwr')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {
        i: plt_colors.to_rgba(atom_weights[i]) for i in range(len(atom_weights))
    }
    highlight_kwargs = {
        'highlightAtoms': list(range(len(atom_weights))),
        'highlightBonds': [],
        'highlightAtomColors': atom_colors
    }

    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(6)

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, **highlight_kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')
    svg2png(bytestring=svg, write_to='tmp.png', dpi=100)
    img = imread('tmp.png')
    os.remove('tmp.png')
    return img


# from https://github.com/ndey96/GCNN-Explainability/blob/master/explain.py
def grad_cam(model, featurized_mol, invert_gradients=False):
    model.eval()
    featurized_mol = featurized_mol.to(device)
    output = model(featurized_mol)
    loss = F.binary_cross_entropy(output, featurized_mol.y.reshape(-1, 1))
    loss.backward()
    node_heat_map = []
    if invert_gradients:
        model.final_conv_grads = (-1) * model.final_conv_grads
    alphas = torch.mean(model.final_conv_grads, axis=0)
    for n in range(model.final_conv_activations.shape[0]):
        node_heat = F.relu(alphas @ model.final_conv_activations[n]).item()
        node_heat_map.append(node_heat)
    model.final_conv_grads = None
    return np.array(node_heat_map)


def plot_grad_cam_explanation(model, mol, featurized_mol, invert_gradients=False):
    grad_cam_weights = grad_cam(model, featurized_mol, invert_gradients)
    scaled_grad_cam_weights = MinMaxScaler().fit_transform(grad_cam_weights.reshape(-1, 1)).squeeze()
    plt.imshow(img_for_mol(mol, scaled_grad_cam_weights))


# from https://github.com/ndey96/GCNN-Explainability/blob/master/explain.py
def saliency_map(model, featurized_mol):
    model.eval()
    featurized_mol = featurized_mol.to(device)
    output = model(featurized_mol)
    loss = F.binary_cross_entropy(output, featurized_mol.y.reshape(-1, 1))
    loss.backward()
    input_grads = model.input.grad
    node_saliency_map = []
    for n in range(input_grads.shape[0]):
        node_grads = input_grads[n, :]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        node_saliency_map.append(node_saliency)
    model.input.grad = None
    return np.array(node_saliency_map)


def plot_saliency_map_explanation(model, mol, featurized_mol):
    saliency_map_weights = saliency_map(model, featurized_mol)
    scaled_saliency_map_weights = MinMaxScaler().fit_transform(saliency_map_weights.reshape(-1, 1)).squeeze()
    plt.imshow(img_for_mol(mol, scaled_saliency_map_weights))


def generate_lime_explanations(train_data, model, instances_to_explain):
    explainer = lime_tabular.LimeTabularExplainer(train_data, feature_names=list(range(len(train_data[0]))),
                                                  categorical_features=list(range(len(train_data[0]))))
    result = []
    for instance in instances_to_explain:
        exp = explainer.explain_instance(instance, model.predict_proba, num_features=10, top_labels=2)
        exp = exp.as_list()
        for i in range(len(exp)):
            exp_bit_number = exp[i][0][:-2]
            exp_coefficient = exp[i][1]
            # Convert all coefficients to correspond to existence of a bit.
            # 0 - bit didn't appear in molecule, 1 - bit appeared in molecule.
            if int(exp[i][0][-1]) == 0:
                exp_coefficient = -exp_coefficient
            exp[i] = (exp_bit_number, exp_coefficient)
        result.append(exp)
    return result


def batch_grad_cam(model, batch, invert_gradients=False):
    model.eval()
    batch = batch.to(device)
    output = model(batch)
    loss = F.binary_cross_entropy(output, batch.y.view(-1, 1))
    loss.backward()
    if invert_gradients:
        model.final_conv_grads = (-1) * model.final_conv_grads
    alphas = tg.utils.scatter(model.final_conv_grads, index=batch.batch, reduce="mean") * len(batch)
    # Grad-CAM scores for each node in the batch.
    # Nodes are assigned to graphs by the batch.batch tensor.
    node_heat_map = F.relu(model.final_conv_activations @ alphas.T).gather(1, batch.batch.view(-1, 1)).squeeze()
    return node_heat_map


def batch_saliency_map(model, batch):
    model.eval()
    batch = batch.to(device)
    output = model(batch)
    loss = F.binary_cross_entropy(output, batch.y.view(-1, 1))
    loss.backward()
    node_saliency = torch.norm(F.relu(model.input.grad), dim=1) * len(batch)
    model.input.grad = None
    return node_saliency


def get_connected_components_for_explanation(mol, exp_score, threshold):
    visited = set()
    components = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in visited and exp_score[atom.GetIdx()] > threshold:
            component = []
            queue = deque([atom.GetIdx()])
            visited.add(atom.GetIdx())

            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx not in visited and exp_score[neighbor_idx] > threshold:
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)
            components.append(component)
    return components


def get_n_atom_connected_components(mol, exp_score, threshold, n):
    scaled_exp_score = MinMaxScaler().fit_transform(exp_score.reshape(-1, 1)).squeeze()
    def dfs(atom_idx, current_component):
        if len(current_component) == n:
            component_set = set(current_component[:])
            if component_set not in components:
                components.append(component_set)
            return
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in current_component and scaled_exp_score[neighbor_idx] > threshold:
                current_component.append(neighbor_idx)
                dfs(neighbor_idx, current_component)
                current_component.pop()

    components = []
    for atom in mol.GetAtoms():
        if scaled_exp_score[atom.GetIdx()] > threshold:
            dfs(atom.GetIdx(), [atom.GetIdx()])
    components = [list(component) for component in components]
    return components


def complete_rings(mol, atom_indices):
    def dfs(atom_idx, current_component):
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in current_component and neighbor.GetIsAromatic():
                current_component.append(neighbor_idx)
                result.append(neighbor_idx)
                dfs(neighbor_idx, current_component)
                current_component.pop()
    
    result = atom_indices.copy()
    for atom_index in atom_indices:
        if mol.GetAtomWithIdx(atom_index).GetIsAromatic():
            dfs(atom_index, [atom_index])
    return list(set(result))


def complete_rings_in_components(mol, components):
    result = []
    for component in components:
        completed_rings = complete_rings(mol, component)
        if not is_duplicate_in_list_of_lists(result, completed_rings):
            result.append(complete_rings(mol, component))
    return result


def rate_explanation_on_synthetic_data(model, mols, data, explanation_fn, scaled_weight_threshold=0.5, smiles_to_match="c1ccccc1"):
    total_score = 0
    for i in range(len(data)):
        mol = mols[i]
        featurized_mol = data[i]
        explanation_weights = explanation_fn(model, featurized_mol)
        scaled_explanation_weights = MinMaxScaler().fit_transform(explanation_weights.reshape(-1, 1)).squeeze()
        substructure_match = mol.GetSubstructMatch(Chem.MolFromSmiles(smiles_to_match))
        belongs_to_substructure = np.zeros(len(featurized_mol.x))
        belongs_to_substructure[list(substructure_match)] = 1
        # Intersection over union (IoU).
        intersection = np.logical_and(belongs_to_substructure, scaled_explanation_weights > scaled_weight_threshold)
        union = np.logical_or(belongs_to_substructure, scaled_explanation_weights > scaled_weight_threshold)
        if np.sum(union) == 0:
            continue
        total_score += np.sum(intersection) / np.sum(union)
    return total_score / len(data)


def get_dataframe_from_lime_results(lime_results):
    bit_numbers = []
    coefficients = []
    for mol_index in range(len(lime_results)):
        for bit_index in range(len(lime_results[mol_index])):
            bit_numbers.append(lime_results[mol_index][bit_index][0])
            coefficients.append(lime_results[mol_index][bit_index][1])
    lime_results_df = pd.DataFrame(data={"bit_number": bit_numbers, "coefficient": coefficients})
    return lime_results_df


def abs_feature_importance_and_lime_difference(feature_importances, lime_importances):
    # Join top 10 feature importances with top 10 LIME importances on bit number to compare.
    importance_lime_join = feature_importances.merge(lime_importances, on="bit_number", how="outer")
    importance_lime_join = importance_lime_join.fillna(0)
    importance_lime_join = importance_lime_join.rename(columns={"importance": "feature_importance", "coefficient": "lime_importance"})
    importance_lime_join["feature_importance"] = importance_lime_join["feature_importance"] / importance_lime_join["feature_importance"].max()
    importance_lime_join["lime_importance"] = importance_lime_join["lime_importance"] / importance_lime_join["lime_importance"].max()
    # Mean of absolute scaled differences between feature importance and LIME importance. Baseline for comparing with real datasets.
    return (importance_lime_join["feature_importance"] - importance_lime_join["lime_importance"]).abs().mean()