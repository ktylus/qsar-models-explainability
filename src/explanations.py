import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
import matplotlib.cm as cm
from skimage.io import imread
import os
# a way to make it work on Windows, it's very bad, and I had to install UniConvertor
os.environ["path"] += r";C:\\Program Files\\UniConvertor-2.0rc5\dlls"
from cairosvg import svg2png


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
def grad_cam(model, featurized_mol):
    model.train()
    featurized_mol = featurized_mol.to(device)
    output = model(featurized_mol)
    loss = F.binary_cross_entropy(output, featurized_mol.y.reshape(-1, 1))
    loss.backward()
    node_heat_map = []
    alphas = torch.mean(model.final_conv_grads, axis=0)
    for n in range(model.final_conv_activations.shape[0]):
        node_heat = F.relu(alphas @ model.final_conv_activations[n]).item()
        node_heat_map.append(node_heat)
    return np.array(node_heat_map)


def plot_grad_cam_explanation(model, mol, featurized_mol):
    grad_cam_weights = grad_cam(model, featurized_mol)
    scaled_grad_cam_weights = MinMaxScaler().fit_transform(grad_cam_weights.reshape(-1, 1)).squeeze()
    plt.imshow(img_for_mol(mol, scaled_grad_cam_weights))


# from https://github.com/ndey96/GCNN-Explainability/blob/master/explain.py
def saliency_map(model, featurized_mol):
    model.train()
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
    return node_saliency_map


def plot_saliency_map_explanation(model, mol, featurized_mol):
    saliency_map_weights = saliency_map(model, featurized_mol)
    scaled_saliency_map_weights = MinMaxScaler().fit_transform(np.array(saliency_map_weights).reshape(-1, 1)).squeeze()
    plt.imshow(img_for_mol(mol, scaled_saliency_map_weights))
