#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 10:52:47 2026

@author: amit
"""

from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles("CCO")
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
print(fp.GetNumBits())


from rdkit import Chem
from rdkit.Chem import Draw

mol = Chem.MolFromSmiles("CCO")  # ethanol
Draw.MolToFile(mol, "ethanol.png")


import os
print(os.getcwd())

from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

mol = Chem.MolFromSmiles("CCO")
img = Draw.MolToImage(mol, size=(300,300))

plt.imshow(img)
plt.axis("off")
plt.show()


from rdkit import Chem
from rdkit.Chem import Draw

mol = Chem.MolFromSmiles("CCO")  # ethanol

save_path = "/home/amit/Drug_discovery/ethanol.png"
Draw.MolToFile(mol, save_path)

print("Saved to:", save_path)




## now drawing molecule , its graph visualization and machine learning understanding 




from rdkit import Chem
from rdkit.Chem import Draw

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
mol = Chem.MolFromSmiles(smiles)

Draw.MolToFile(mol, "aspirin.png")



import matplotlib.pyplot as plt
img = Draw.MolToImage(mol, size=(400,400))
plt.imshow(img)
plt.axis("off")
plt.show()


print("Atoms (Nodes):\n")
for atom in mol.GetAtoms():
    print(
        f"Atom {atom.GetIdx()} | "
        f"Symbol: {atom.GetSymbol()} | "
        f"AtomicNum: {atom.GetAtomicNum()} | "
        f"Hybridization: {atom.GetHybridization()} | "
        f"Aromatic: {atom.GetIsAromatic()}"
    )

print("\nBonds (Edges):\n")
for bond in mol.GetBonds():
    print(
        f"Bond {bond.GetBeginAtomIdx()} - {bond.GetEndAtomIdx()} | "
        f"Type: {bond.GetBondType()} | "
        f"Aromatic: {bond.GetIsAromatic()}"
    )


## to visualize the molecule as graph 
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

# Add atoms as nodes
for atom in mol.GetAtoms():
    G.add_node(atom.GetIdx(), label=atom.GetSymbol())

# Add bonds as edges
for bond in mol.GetBonds():
    G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

labels = nx.get_node_attributes(G, 'label')
pos = nx.spring_layout(G, seed=42)

nx.draw(G, pos, with_labels=True, labels=labels,
        node_size=2000, node_color="lightblue",
        font_size=10, font_weight="bold")
plt.show()


### generate molecular fingerprint 
from rdkit.Chem import AllChem

fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
print("Fingerprint length:", fp.GetNumBits())

import numpy as np


## to see the fingerprints
arr = np.zeros((2048,), dtype=int)
Chem.DataStructs.ConvertToNumpyArray(fp, arr)

print("First 50 bits:")
print(arr[:50])











