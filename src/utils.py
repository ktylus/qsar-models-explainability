import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from IPython.core.display import display, HTML


def save_decision_tree_graph(model: DecisionTreeRegressor, filename: str):
    assert filename.split(".")[-1] in ["png", "pdf"]
    plt.figure(figsize=(20, 10))
    plot_tree(model)
    plt.savefig(filename)
    plt.close()


def load_data():
    data = Chem.SDMolSupplier("data/caco2_permeability_from_Chembl_permeability.sdf")
    data2 = Chem.SDMolSupplier("data/CYP2D6_IC50_CHEMBL_data.sdf")
    data3 = Chem.SDMolSupplier("data/CYP3A4_IC50_CHEMBL_data.sdf")
    data4 = Chem.SDMolSupplier("data/genotoxicity_from_chembl_data.sdf")
    data5 = Chem.SDMolSupplier("data/hERG_IC50_CHEMBL_data.sdf")
    datasets = (data, data2, data3, data4, data5)
    return datasets


def get_data_target_field_names():
    return ("Field 11", "Standard Value", "Standard Value", "Field 11", "Standard Value")


def draw_morgan_bit_many_molecules(molecules: list, bit_id, radius=2, length=1024):
    target_mols = []
    bit_infos = []
    for i, mol in enumerate(molecules):
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=length, radius=radius, bitInfo=bit_info)
        if fp[bit_id]:
            target_mols.append(mol)
            bit_infos.append(bit_info)
    imgs = []
    for mol, bit_info in zip(target_mols, bit_infos):
        imgs.append(Draw.DrawMorganBit(mol, bit_id, bit_info))
    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    for i, img in enumerate(imgs[:10]):
        ax[i // 5, i % 5].imshow(img)
        ax[i // 5, i % 5].axis('off')
        ax[i // 5, i % 5].set_title(f'bit {i}')


def display_lime_explanation(explanation):
    explanation.save_to_file("lime/ex.html")
    display(HTML('<iframe src=lime/ex.html width=900 style="background: #FFFFFF;" height=400></iframe>'))
