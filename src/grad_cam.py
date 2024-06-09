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
    x = featurized_mol.x.to(device)
    y = featurized_mol.y.to(device)
    edge_index = featurized_mol.edge_index.to(device)
    output = model(x, edge_index)
    loss = F.mse_loss(output, y)
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
