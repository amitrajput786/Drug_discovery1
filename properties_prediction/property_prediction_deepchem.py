#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:12:56 2026

@author: amit
"""

# notebooks/02_property_prediction.ipynb

"""
Molecular Property Prediction for Drug Discovery
Author: Amit Kumar
Purpose: Build and evaluate ML models for ADMET property prediction
"""





import deepchem as dc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

# =============================================================================
# 1. LOAD TOX21 DATASET (Toxicity Prediction)
# =============================================================================

print("Loading Tox21 dataset...")
tasks, datasets, transformers = dc.molnet.load_tox21(
    featurizer='GraphConv',
    splitter='random'
)
train_dataset, valid_dataset, test_dataset = datasets

print(f"Number of tasks: {len(tasks)}")
print(f"Tasks: {tasks}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# another way to load the data 




# Step 2 — Check what models are actually available
print(dir(dc.models))







# Pass featurizer object directly instead of string
featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

tasks, datasets, transformers = dc.molnet.load_tox21(
    featurizer=featurizer,    # ← pass object, not string
    splitter='random'
)
train_dataset, valid_dataset, test_dataset = datasets

print(f"Number of tasks: {len(tasks)}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Test samples: {len(test_dataset)}")




tasks, datasets, transformers = dc.molnet.load_tox21(
    featurizer='MolGraphConv',
    splitter='random'
)








# =============================================================================
# 2. BUILD GRAPH CONVOLUTIONAL NETWORK
# =============================================================================

print("\nBuilding Graph Convolutional Model...")

# model = dc.models.GCNModel(
#     n_tasks=len(tasks),
#     mode='classification',
#     dropout=0.2,
#     learning_rate=0.001,
#     batch_size=64
# )








#======== few changes  GCN model build  succesfully 
model = dc.models.GCNModel(
    n_tasks=len(tasks),
    mode='classification',
    dropout=0.2,
    learning_rate=0.001,
    batch_size=64
)
print("GCN Model built successfully!")




# =============================================================================
# 3. TRAIN THE MODEL



# =============================================================================

# here is the training 


# Train
print("\nTraining...")
losses = []
for epoch in range(50):
    loss = model.fit(train_dataset, nb_epoch=1, checkpoint_interval=0)
    losses.append(loss)
    if epoch % 1== 0:
        print(f"Epoch {epoch:>3}: Loss = {loss:.4f}")


plt.figure(figsize=(10, 5))
plt.plot(losses, color='steelblue', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GCN Training Loss on Tox21')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()




#####============Evaluate GCN 



# Evaluate GCN
print("\nEvaluating...")
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
test_scores  = model.evaluate(test_dataset,  [metric], transformers)

print(f"Train ROC-AUC: {train_scores['roc_auc_score']:.4f}")
print(f"Valid ROC-AUC: {valid_scores['roc_auc_score']:.4f}")
print(f"Test  ROC-AUC: {test_scores['roc_auc_score']:.4f}")



#================Comparison with baseline model



from sklearn.ensemble import RandomForestClassifier

# Load with ECFP for RF
tasks_rf, datasets_rf, transformers_rf = dc.molnet.load_tox21(
    featurizer='ECFP',
    splitter='random'
)
train_rf, valid_rf, test_rf = datasets_rf

rf_model = dc.models.SklearnModel(
    RandomForestClassifier(n_estimators=100, n_jobs=-1),
    model_dir='rf_model'
)
rf_model.fit(train_rf)
rf_test_score = rf_model.evaluate(test_rf, [metric], transformers_rf)

print("\nComparison Results:")
print(f"{'Model':<25} {'Test ROC-AUC':<15}")
print("-" * 40)
print(f"{'Random Forest (ECFP)':<25} {rf_test_score['roc_auc_score']:.4f}")
print(f"{'GCN':<25} {test_scores['roc_auc_score']:.4f}")






#=======predictions 


print("\n=== Predict Toxicity for Drug Candidates ===")

new_molecules = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',        # Aspirin
    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',   # Ibuprofen
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',    # Caffeine
]
mol_names = ['Aspirin', 'Ibuprofen', 'Caffeine']

# Must use same featurizer as GCN training!
featurizer_pred = dc.feat.MolGraphConvFeaturizer(use_edges=True)
features = featurizer_pred.featurize(new_molecules)

new_dataset = dc.data.NumpyDataset(X=features)
predictions = model.predict(new_dataset)

print("\nToxicity Predictions (probability of being toxic):")
for i, name in enumerate(mol_names):
    print(f"\n{name}:")
    for j, task in enumerate(tasks):
        prob = predictions[i][j][1]   # [1] = positive class probability
        flag = "HIGH" if prob > 0.5 else "low"
        print(f"  {task:<35} {prob:.3f}  {flag}")


















