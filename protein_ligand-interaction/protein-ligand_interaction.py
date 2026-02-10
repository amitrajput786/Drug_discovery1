#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 22:14:41 2026

@author: amit
"""

# notebooks/03_protein_ligand_basics.ipynb

"""
Protein-Ligand Interactions for Drug Discovery
Author: Amit Kumar
Purpose: Understand binding pockets and docking concepts
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import py3Dmol
import requests
import numpy as np

# =============================================================================
# 1. UNDERSTANDING PROTEIN STRUCTURE
# =============================================================================

print("=== PROTEIN STRUCTURE BASICS ===\n")

# Fetch a protein structure from PDB
pdb_id = "1FJS"  # HIV-1 Protease with inhibitor

def fetch_pdb(pdb_id):
    """Fetch protein structure from RCSB PDB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    return response.text

pdb_data = fetch_pdb(pdb_id)

# Save locally
with open(f'data/{pdb_id}.pdb', 'w') as f:
    f.write(pdb_data)

print(f"Downloaded {pdb_id}: HIV-1 Protease (Drug Target for AIDS)")
print("This enzyme is essential for HIV replication")
print("Drugs that block this enzyme stop the virus from maturing\n")

# =============================================================================
# 2. VISUALIZE PROTEIN-LIGAND COMPLEX
# =============================================================================

def visualize_complex(pdb_data):
    """Interactive 3D visualization of protein-ligand complex."""
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    
    # Protein surface
    view.setStyle({'protein': True}, {'cartoon': {'color': 'spectrum'}})
    
    # Ligand
    view.setStyle({'hetflag': True}, {'stick': {'colorscheme': 'greenCarbon'}})
    
    # Binding site residues
    view.addSurface(py3Dmol.VDW, 
                    {'opacity': 0.3, 'color': 'white'},
                    {'hetflag': True, 'expand': 5})
    
    view.zoomTo()
    return view


from IPython.display import display

view = visualize_complex(pdb_data)
view.show() 
view = visualize_complex(pdb_data)
html = view._make_html()
with open('protein_view.html', 'w') as f:
    f.write(html)
print("Visualization saved to protein_view.html")
print("Open this file in your web browser to see the 3D structure")


## it save to the file .html file 








# Display (in Jupyter notebook)
# visualize_complex(pdb_data)

# =============================================================================
# 3. UNDERSTANDING BINDING POCKETS
# =============================================================================

print("=== BINDING POCKETS ===\n")

print("""
What is a Binding Pocket?
------------------------
A binding pocket is a cavity on the protein surface where small
molecules (drugs/ligands) bind.

Key Characteristics:
1. SHAPE: Complementary to ligand shape (lock-and-key)
2. CHEMISTRY: Mix of hydrophobic and polar regions
3. SIZE: Typically 300-1000 Å³ for small molecule drugs
4. DEPTH: Deeper pockets provide better binding

Types of Pockets:
- Orthosteric: Natural binding site (blocks normal function)
- Allosteric: Alternative site (modulates function)
- Cryptic: Only visible when protein changes shape

For Drug Discovery:
- AlphaFold predicts protein structure
- Pocket detection tools find druggable sites
- Docking predicts how molecules bind
""")

# =============================================================================
# 4. MOLECULAR DOCKING CONCEPTS
# =============================================================================

print("=== MOLECULAR DOCKING ===\n")

print("""
What is Molecular Docking?
--------------------------
Computational method to predict:
1. WHERE a ligand binds in a protein (pose)
2. HOW WELL it binds (score)

The Process:
1. Prepare protein (add hydrogens, define binding site)
2. Prepare ligand (generate 3D conformation)
3. Search algorithm explores possible poses
4. Scoring function evaluates each pose
5. Return best poses ranked by score

Scoring Functions:
- Vina Score: Combines empirical + knowledge-based terms
- Lower score = Better binding (typically -6 to -12 kcal/mol)

Common Tools:
- AutoDock Vina (open source, fast)
- Glide (Schrödinger, commercial)
- GOLD (CCDC, commercial)
- DiffDock (ML-based, new)
""")

# =============================================================================
# 5. SIMPLE DOCKING EXAMPLE WITH DEEPCHEM
# =============================================================================

print("=== FEATURIZING PROTEIN-LIGAND COMPLEX ===\n")

import deepchem as dc

# Use DeepChem's complex featurizer
featurizer = dc.feat.AtomicConformationFeaturizer()

# Example: Featurize a simple molecule
smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'  # Aspirin
features = featurizer.featurize([smiles])

print(f"Number of atoms: {features[0].num_atoms}")
print(f"Atomic numbers: {features[0].atomic_number}")
print(f"Positions shape: {features[0].positions.shape}")

# =============================================================================
# 6. CONNECTING TO PATTERN'S PIPELINE
# =============================================================================

print("\n" + "="*60)
print("CONNECTION TO PATTERN'S PIPELINE")
print("="*60)
print("""
In the AI Drug Discovery pipeline at Pattern:

1. TARGET STRUCTURE:
   └── Use AlphaFold/OpenFold to predict protein structure
   └── Identify binding pockets using fpocket/P2Rank

2. MOLECULE GENERATION:
   └── Generate molecules that fit the pocket
   └── Use RL/Diffusion models

3. DOCKING & SCORING:
   └── Dock generated molecules to target
   └── Rank by binding affinity
   └── Filter by ADMET properties

4. CHEMIST REVIEW:
   └── Medicinal chemist reviews top candidates
   └── Provides feedback for model improvement

This notebook demonstrates understanding of:
✓ Protein structure and binding pockets
✓ Protein-ligand interactions
✓ Docking concepts
✓ Feature extraction for ML
""")