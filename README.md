AI-Driven Drug Discovery: From Protein Structure to Molecular Design

Python
DeepChem
RDKit
AlphaFold
PyMOL

Drug Discovery Pipeline

Molecular Descriptors Analysis for Drug-likeness Assessment (Lipinski's Rule of Five)

Similarity Matrix

Molecular Similarity Matrix using Morgan Fingerprints (Tanimoto Similarity)

GNN Training

Graph Convolutional Network Training for Toxicity Prediction
🎯 Project Overview

A comprehensive exploration of AI-driven drug discovery, covering the complete pipeline from protein structure prediction to molecular property prediction. This repository demonstrates practical implementation of modern computational drug discovery techniques combining deep learning with chemistry domain knowledge.

Author: Amit Kumar
Affiliation: NIT Rourkela, Integrated M.Sc. Chemistry
Email: amitrajput51169@gmail.com
🏆 Key Achievements

    🧬 AlphaFold Integration: Protein structure prediction and visualization workflow
    🔬 Molecular Representations: SMILES, fingerprints, and graph-based encoding
    📊 GNN for ADMET: Graph Convolutional Networks achieving ~0.82 ROC-AUC on Tox21
    🔗 Protein-Ligand Analysis: Binding pocket concepts and featurization
    📝 Published Research: First author paper in Elsevier journal

📁 Repository Structure

text

Drug_discovery1/
├── alphafold_pymol/
│   └── AlphaFold → PyMOL workflow for protein structure
├── molecular_representation/
│   ├── molecular_representation.py
│   ├── molecule_description.png
│   ├── similarity_matrix.png
│   └── molecular_graph.png
├── properties_prediction/
│   ├── properties_prediction.py
│   ├── GNC_model_training.png
│   ├── GNC_model_training_graph.png
│   └── GNC_model_testing.png
├── protein_ligand_interaction/
│   ├── protein_ligand_interaction.py
│   ├── loaded_protein.png
│   ├── featuring_protein_ligand.png
│   └── protein_visualization.png
├── requirements.txt
└── README.md

🧬 Part 1: AlphaFold → PyMOL Workflow
Protein Structure Prediction Pipeline

text

Amino Acid Sequence
        ↓
Multiple Sequence Alignment (MSA)
        ↓
Co-evolutionary Analysis
        ↓
Deep Neural Network (Evoformer + Structure Module)
        ↓
3D Atomic Structure + Confidence Scores

Input Example: Lysozyme Protein Sequence

text

KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL

Property	Value
Length	129 amino acids
Function	Enzyme that breaks bacterial cell walls
Use Case	Classic example for structure prediction
AlphaFold Output Files
File	Description
ranked_0.pdb	Best predicted structure
ranked_1-4.pdb	Alternative models
scores.json	Confidence scores (pLDDT, PAE, pTM)
pae.json	Pairwise alignment error matrix
PyMOL Visualization Commands

Python

# Load structure
load ranked_0.pdb

# Show secondary structure
show cartoon
color yellow, ss h  # helices
color cyan, ss s    # sheets

# Highlight active site (Lysozyme: Glu35, Asp52)
select active_site, resi 35+52
show sticks, active_site
color red, active_site
zoom

🔬 Part 2: Molecular Representations

Understanding how molecules are represented for machine learning is fundamental to drug discovery.
Key Representations
Representation	Description	Use Case
SMILES	String notation for molecules	Data storage, parsing
Molecular Descriptors	Numerical properties (MW, LogP, etc.)	Drug-likeness filtering
Fingerprints	Binary vectors encoding substructures	Similarity search
Graphs	Atoms as nodes, bonds as edges	Graph Neural Networks
Drug-likeness Analysis (Lipinski's Rule of Five)

Molecular Descriptors
Property	Threshold	Significance
Molecular Weight	< 500 Da	Absorption
LogP	< 5	Membrane permeability
H-Bond Donors	< 5	Solubility
H-Bond Acceptors	< 10	Binding interactions
Molecular Similarity Matrix

Similarity Matrix

Key Insight: The Structure-Activity Relationship (SAR) principle states that structurally similar molecules often have similar biological activities.
Molecular Graph Representation

Molecular Graph
Component	Representation	Features
Nodes	Atoms	Element, formal charge, hybridization, aromaticity
Edges	Bonds	Bond type, conjugation, ring membership
📊 Part 3: Property Prediction with Graph Neural Networks
What is ADMET?

text

A - Absorption:    Will the drug get into the bloodstream?
D - Distribution:  Will it reach the target organ/tissue?
M - Metabolism:    How will the body break it down?
E - Excretion:     How will the body eliminate it?
T - Toxicity:      Will it cause harm?

~60% of drug failures in clinical trials are due to poor ADMET properties.
Dataset: Tox21
Property	Value
Dataset	Tox21 (NIH toxicity screening)
Tasks	12 toxicity endpoints
Molecules	~8,000 compounds
Model	Graph Convolutional Network (GCN)
Framework	DeepChem
Model Architecture

text

Input: Molecular Graph (atoms + bonds)
        ↓
Graph Convolution Layer 1 (64 units, ReLU)
        ↓
Graph Convolution Layer 2 (64 units, ReLU)
        ↓
Global Mean Pooling (graph → vector)
        ↓
Dense Layer (128 units, ReLU)
        ↓
Dropout (0.2)
        ↓
Output: 12 Toxicity Predictions (sigmoid)

Training Results
Metric	Visualization
Training Epochs	GCN Training
Loss Curve	GCN Loss
Evaluation	GCN Eval
Performance Comparison
Model	Representation	Test ROC-AUC
Random Forest	ECFP Fingerprints	~0.75
Graph Convolutional Network	Molecular Graph	~0.82

Key Insight: GNNs outperform fingerprint-based methods by learning task-specific molecular representations.
🔗 Part 4: Protein-Ligand Interactions
Binding Pocket Concepts

text

┌─────────────────────────────────────────────────────────────┐
│                      BINDING POCKET                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Definition: Cavity on protein surface where drugs bind     │
│                                                             │
│  Key Characteristics:                                        │
│  ├── Volume: 300-1000 Å³ (typical for small molecules)      │
│  ├── Shape: Complementary to ligand (lock-and-key)          │
│  ├── Chemistry: Mix of hydrophobic + polar regions          │
│  └── Depth: Deeper pockets = better, more specific binding  │
│                                                             │
│  Types of Binding Sites:                                     │
│  ├── Orthosteric: Natural substrate binding site            │
│  ├── Allosteric: Alternative site that modulates function   │
│  └── Cryptic: Only appears upon conformational change       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Molecular Docking

text

Protein (Target) + Ligand (Drug Candidate)
                ↓
         DOCKING ENGINE
    (AutoDock Vina / DiffDock)
                ↓
    Binding Pose + Affinity Score

Score Interpretation:
  -6 kcal/mol → Weak binding (μM affinity)
  -9 kcal/mol → Good binding (nM affinity)
  -12 kcal/mol → Very strong binding

Example: HIV-1 Protease (PDB: 1FJS)
Visualization	Image
Loaded Protein	Loaded Protein
3D Visualization	Protein Viz
Featurization	Featurization
Features Extracted
Feature	Description	Use
Atomic Positions	3D coordinates (x, y, z)	Spatial relationships
Atomic Numbers	Element identity	Atom type embedding
Formal Charges	Integer charge	Electrostatic interactions
Partial Charges	Gasteiger charges	Binding affinity prediction
🎯 AI Drug Discovery Pipeline

text

┌─────────────────────────────────────────────────────────────────────────────┐
│                      AI DRUG DISCOVERY PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: TARGET IDENTIFICATION & STRUCTURE                                  │
│  └── AlphaFold predicts protein 3D structure ──────────► [Part 1: AlphaFold]│
│  └── Identify druggable binding pockets                                     │
│                                                                             │
│  STEP 2: MOLECULAR REPRESENTATION                                           │
│  └── Encode molecules for ML (SMILES, fingerprints, graphs) ► [Part 2]      │
│  └── Filter by drug-likeness (Lipinski's Rule)                              │
│                                                                             │
│  STEP 3: PROPERTY PREDICTION (ADMET)                                        │
│  └── Predict toxicity, absorption, metabolism ──────────► [Part 3: GNN]     │
│  └── Filter problematic compounds early                                     │
│                                                                             │
│  STEP 4: PROTEIN-LIGAND INTERACTION                                         │
│  └── Understand binding mechanisms ─────────────────────► [Part 4]          │
│  └── Dock molecules to target, score binding                                │
│                                                                             │
│  STEP 5: MOLECULE GENERATION (Future Work)                                  │
│  └── De novo design with RL/Diffusion models                                │
│  └── Optimize for target binding + drug-likeness                            │
│                                                                             │
│  STEP 6: CHEMIST REVIEW & SYNTHESIS                                         │
│  └── Expert review of AI-generated candidates                               │
│  └── Synthesis and experimental validation                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

🛠️ Installation

Bash

# Clone repository
git clone https://github.com/amitrajput786/Drug_discovery1.git
cd Drug_discovery1

# Create conda environment
conda create -n drugdiscovery python=3.9
conda activate drugdiscovery

# Install dependencies
pip install -r requirements.txt

Requirements

text

# Core ML Frameworks
deepchem>=2.7.0
torch>=2.0.0
torch-geometric>=2.3.0

# Chemistry Libraries
rdkit>=2023.03
py3Dmol>=2.0.0

# Data Science
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0

# Visualization & Utilities
networkx>=3.0
requests>=2.28.0

Additional Tools
Tool	Installation	Purpose
PyMOL	conda install -c schrodinger pymol	Protein visualization
ColabFold	Google Colab notebook	AlphaFold predictions
AutoDock Vina	conda install -c conda-forge vina	Molecular docking
📚 Key Learnings
1. Protein Structure (AlphaFold)

    AlphaFold revolutionizes structure prediction from sequence alone
    MSA provides evolutionary information for accurate folding
    Confidence scores (pLDDT, PAE) indicate prediction reliability

2. Molecular Representations

    SMILES provides compact, parseable string representation
    Fingerprints enable fast similarity search
    Graph representations preserve topology for neural networks

3. Property Prediction (GNN)

    Graph Neural Networks outperform traditional fingerprint methods
    Multi-task learning improves predictions
    Early ADMET/toxicity filtering saves resources

4. Protein-Ligand Interactions

    Binding pockets determine druggability of targets
    Docking predicts binding pose and affinity
    3D features are essential for accurate interaction modeling

📖 References
Foundational Papers

    AlphaFold 2 - Jumper, J., et al. (2021). Nature, 596, 583-589. DOI

    Neural Message Passing - Gilmer, J., et al. (2017). ICML. arXiv:1704.01212

    Graph Convolutional Networks - Duvenaud, D., et al. (2015). NeurIPS. arXiv:1509.09292

    DeepChem - Ramsundar, B., et al. (2019). O'Reilly Media. Documentation

    REINVENT - Olivecrona, M., et al. (2017). J. Cheminformatics. DOI

    DiffDock - Corso, G., et al. (2022). ICLR 2023. arXiv:2210.01776

Tools & Libraries
Tool	Documentation
RDKit	https://www.rdkit.org/docs/
DeepChem	https://deepchem.io/
PyTorch Geometric	https://pytorch-geometric.readthedocs.io/
ColabFold	https://github.com/sokrypton/ColabFold
PyMOL	https://pymol.org/2/
AutoDock Vina	https://vina.scripps.edu/
Datasets
Dataset	Description	Source
Tox21	Toxicity screening	MoleculeNet
PDB	Protein structures	RCSB PDB
ChEMBL	Bioactivity data	ChEMBL
ZINC	Purchasable compounds	ZINC
🔗 Related Work
Published Research

    DM-FNet: Lightweight CNN for Medical Image Classification
    Kumar, A., et al. (2025). Biomedical Signal Processing and Control (Elsevier)
    Published Paper

Live Demos

    WBC Classifier: Hugging Face Spaces

Achievements

    Deep-ML Profile: Global Rank < 200 — 70+ ML/DL/CV problems solved

🚀 Future Work

    Implement molecular generation using REINVENT (RL-based)
    Add docking workflow with AutoDock Vina
    Integrate DiffDock for ML-based pose prediction
    Build ADMET prediction models on larger datasets
    Create end-to-end pipeline from target to ranked candidates
    Add binding affinity prediction with 3D GNNs

📜 License
Component	License
AlphaFold parameters	CC BY 4.0
AlphaFold source code	Apache 2.0
ColabFold	MIT
DeepChem	MIT
This repository	MIT
🙏 Acknowledgments

    DeepMind AlphaFold Team - Revolutionary protein structure prediction
    ColabFold Team (Mirdita, Ovchinnikov, Steinegger) - Accessible AlphaFold
    DeepChem Development Team - Comprehensive ML for life sciences
    RDKit Community - Open-source cheminformatics
    NIT Rourkela Machine Learning Lab - Research guidance and support

📧 Contact

Amit Kumar
Integrated M.Sc. Chemistry (Computational Chemistry & AI Focus)
National Institute of Technology, Rourkela
Platform	Link
📧 Email	amitrajput51169@gmail.com
💼 LinkedIn	Amit Kumar
🐙 GitHub	amitrajput786
📊 Deep-ML	Profile
