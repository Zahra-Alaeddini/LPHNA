<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-orange" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-GNN-blue" alt="PyTorch Geometric">
  <img src="https://img.shields.io/badge/Explainability-XAI-purple" alt="Explainable AI">
</p>

# Explainable Multi-Module Framework for Drug Repurposing  
### Link Prediction & Explanation over a Heterogeneous Biomedical Knowledge Graph

This repository contains the official implementation of an **explainable graph-based drug repurposing framework**, integrating:

- A **relation-aware GNN encoder**
- A **scoring-based MLP decoder**
- A neuro-inspired **Hierarchical Neuro-Attention (HNA) explanation module**

The system performs **link prediction** among biomedical entities using a heterogeneous knowledge graph (BKG) derived from **HetioNet**, and generates **causal, counterfactual, and contrastive explanations** for each prediction.

<p align="center">
  <img width="2000" height="1500" alt="Model Architecture" src="https://github.com/user-attachments/assets/a0133598-864e-4739-9906-63b8fe549577" />
</p>

# 🧠 1. Overview

Drug repurposing requires models capable of:
- Learning heterogeneous biological relations  
- Handling graph sparsity and imbalance  
- Generating **explainable biomedical reasoning**  

Our framework addresses these needs by combining:

### ✔️ Graph Neural Networks  
### ✔️ Similarity-based graph augmentation  
### ✔️ Neuro-inspired explainability  
### ✔️ Semantic understanding using LLM embeddings  
### ✔️ Multi-level attention  

This system predicts new **Compound → Disease** therapeutic associations and produces human-interpretable explanations.

---

# 🧬 2. Dataset: Customized HetioNet Subgraph

We extract and refine a biomedical knowledge graph including:

- **1,689 nodes**
  - 1,552 compounds  
  - 137 diseases  
- **Relations**
  - Compound–treats–Disease (CtD)  
  - Compound–palliates–Disease (CpD)  
  - Compound–resembles–Compound (CrC)  
  - Disease–resembles–Disease (DrD)

## Similarity Matrices
Four similarity channels:
- Gene-based similarity  
- Symptom-based similarity  
- Anatomical involvement  
- Side-effect profiles  

These similarities enrich message passing and improve representation learning.

---

# 🏗️ 3. Architecture

The full pipeline includes **three core modules**:

---

## 🔹 3.1 Multi-Behaviour GNN Encoder (Mb-GNN)

The encoder processes heterogeneous biomedical relations using:

- **Relation-specific transformation matrices**  
- **Multi-head attention over edge types**  
- **Message passing enhanced with similarity priors**  
- **BatchNorm & Dropout for stability**

### Output:
- Compound embeddings  
- Disease embeddings  
- Relation embeddings 

---

## 🔹 3.2 MLP Decoder

The decoder predicts links via:

- Concatenated embeddings  
- 2 fully connected layers  
- ReLU / LeakyReLU  
- Sigmoid scoring  

Used to compute link probabilities for:
- **Compound → Disease**

---

## 🔹 3.3 Hierarchical Neuro-Attention (HNA) Explainer

### Key Components:
- **Local attention** over neighborhood  
- **Global attention** over multi-hop paths  
- **Semantic LLM embeddings** (BERT/ BioBERT / PubMedBERT)  
- **Structural graph features**  
  - degree  
  - common neighbors  
  - similarity score  
  - multi-hop connectivity  
- **Hebbian learning rule** to amplify biologically meaningful patterns  

### Explanation Outputs:
- 🧠 Causal  
- 🔄 Counterfactual  
- ⚖️ Contrastive  

<p align="center">
  <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/8a952e21-ca94-4051-8692-836aba580d83" />
<p>
---

# 📊 4. Performance

## ✔️ Link Prediction

| Setting | ROC-AUC | AUPR |
|--------|---------|-------|
| **In-Distribution** | **0.9882** | **0.9921** |
| **Out-of-Distribution (20%)** | **0.9748** | **0.9754** |

The model outperforms several state-of-the-art baselines, including:
- DRHGCN  
- DRPreter  
- TxGNN  
- DBR-x  
- rd-explainer  

---

## ✔️ Explanation Quality

| Metric | Score |
|--------|--------|
| Comprehensiveness | **0.0837** |
| Robustness | **0.9210** |
| Description Accuracy | **0.9306** |
| Stability | **0.9803** |
| Consistency | **0.9646** |

---

# 🙋‍♀️ 5. Contact

👩‍💻 **Zahra Alaeddini**

📧 alaeddini.zahra@gmail.com

---

# 📦 6. Requirements
Make sure you have the following dependencies installed:

```bash
Python >= 3.8
PyTorch
PyTorch Geometric
Transformers
NumPy
Pandas
NetworkX
scikit-learn
