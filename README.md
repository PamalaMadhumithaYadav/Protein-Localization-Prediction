# Protein Localization Prediction using HMM & PSSM

Multi-label protein subcellular localization using evolutionary features and deep learning.

---

## 🔬 Overview
This project predicts **Protein Localization Prediction** using **evolutionary information** derived from **Hidden Markov Models (HMM)** and **Position-Specific Scoring Matrices (PSSM)**.

The task is formulated as a **multi-label classification problem**, since a protein may localize to multiple cellular compartments simultaneously.

We compare:
- Classical machine learning models on pooled evolutionary features
- A CNN-based deep learning model operating directly on raw HMM and PSSM matrices

---

## 🧬 Localization Labels
Each protein may belong to one or more of the following compartments:

- `envelope`
- `lumen`
- `plastoglobule`
- `stroma`
- `thylakoid_membrane`

---

## 🗂 Dataset Structure
```
 Dataset/
├── Benchmark_BinaryML.csv
├── Novel_BinaryML.csv
└── parse_files.py 
```

- **Benchmark** → training dataset  
- **Novel** → independent test dataset  

> ⚠️ HMM and PSSM matrices have **different sequence lengths** and **must not be aligned position-wise**.

---

## 🧠 Problem Formulation
- **Task**: Multi-label classification  
- **Input**:
  - HMM matrix (L₁ × 30)
  - PSSM matrix (L₂ × 42)
- **Output**: Binary vector of 5 labels  
- **Evaluation Metrics**:
  - Micro-F1 (primary)
  - Macro-F1
  - Hamming Accuracy
  - Subset Accuracy
  - ROC-AUC and PR-AUC

---

## 🏗 Methods

### 1️⃣ Classical Machine Learning (Baseline)
- Independent pooling of HMM and PSSM matrices:
  - Mean, Max, Standard Deviation
- Feature fusion into a 216-dimensional vector
- Models:
  - Logistic Regression (One-vs-Rest)
  - XGBoost (One-vs-Rest)
- Per-label threshold tuning to handle class imbalance

---

### 2️⃣ CNN on Raw HMM + PSSM Matrices
- Two-branch CNN architecture:
  - **HMM branch**: 1D CNN + masked global pooling
  - **PSSM branch**: 1D CNN + masked global pooling
- Late fusion of learned representations
- Binary cross-entropy loss with class imbalance handling
- Per-label threshold optimization

---

## 📊 Results (Summary)

| Model | Micro-F1 | Macro-F1 | Hamming Accuracy |
|------|---------|----------|------------------|
| Logistic Regression (tuned) | ~0.57 | ~0.35 | ~0.78 |
| XGBoost (tuned) | **~0.61** | ~0.38 | ~0.79 |
| CNN (raw matrices) | ~0.59 | **~0.54** | ~0.73 |

### Key Observations
- XGBoost achieves the best **overall Micro-F1**
- CNN significantly improves **Macro-F1**, benefiting rare labels
- Evolutionary features (HMM + PSSM) are highly informative for localization

---

## 📈 Visual Analysis
The notebook includes:
- Label distribution (class imbalance)
- Multi-label histogram
- ROC curves (micro + per label)
- Precision–Recall curves
- Threshold vs F1 plots
- Training loss and F1 curves
- Label co-occurrence heatmap

---

## 🛠 Tech Stack
- **Python**
- NumPy, Pandas
- Scikit-learn
- PyTorch
- XGBoost
- Google Colab (NVIDIA T4 GPU)

---

## 🚀 How to Run
1. Open `Protein_Localization_Prediction.ipynb`
2. Mount Google Drive containing the dataset
3. Run all cells sequentially:
   - Data parsing
   - Feature extraction
   - Model training
   - Evaluation and visualization

---

## ⚠️ Limitations
- Limited dataset size
- Severe class imbalance for some compartments
- Threshold sensitivity for rare labels

---

## 🔮 Future Work
- Integrate pretrained protein language models (ESM, ProtBERT, ProtT5)
- Attention-based feature fusion
- Larger and more diverse datasets
- Multi-task learning approaches

---

## 👤 Author
**Madhu**  
M.S. in Artificial Intelligence  
University at Buffalo
