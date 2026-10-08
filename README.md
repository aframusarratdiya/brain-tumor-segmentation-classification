# 🧠 Multi-Task Brain Tumor Segmentation & Classification (BRISC 2025)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aframusarratdiya/brain-tumor-segmentation-classification/blob/main/Brain_Tumor_Segmentation_and_Classification.ipynb)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![SMP](https://img.shields.io/badge/SMP-Segmentation_Models_PyTorch-blue)

An end-to-end deep learning framework for joint pixel-level tumor segmentation ($256 \times 256$) and 4-class multi-class pathological classification on MRI scans from the BRISC 2025 challenge[cite: 21].

---

## 📌 Executive Summary

Automated neuro-oncological analysis requires solving two clinical tasks simultaneously:
1. **Localization & Delineation (Segmentation):** Determining the precise boundary and spatial extent of the lesion[cite: 21].
2. **Pathological Staging (Classification):** Categorizing the lesion into one of the four clinical profiles: `no_tumor`, `meningioma`, `pituitary`, or `glioma`[cite: 21].

This project implements a **Multi-Task Joint U-Net** architecture where the encoder features are shared between a pixel-level segmentation decoder head and a bottleneck classification head, trained via a unified composite loss[cite: 21].

---

## 📊 Dataset & Verification (BRISC 2025)

* **Dataset Source:** Downloaded programmatically via `kagglehub` (`briscdataset/brisc2025`)[cite: 21].
* **Volume:** 6,000 total high-contrast clinical MRI scans[cite: 21].
* **Class Profiles:**
  * `no_tumor`: Healthy control tissue (empty segmentation mask)[cite: 21, 21].
  * `meningioma`: Extra-axial meningeal tumors[cite: 21].
  * `pituitary`: Sellar / pituitary gland tumors[cite: 21].
  * `glioma`: Glial cell intra-axial malignancies[cite: 21].
* **Partitioning (80/20 Stratified Split):**
  * **Training Set:** 4,000 image-mask pairs[cite: 21]
  * **Validation Set:** 1,000 image-mask pairs[cite: 21]
  * **Unseen Test Set:** 1,000 image-mask pairs[cite: 21]

### Augmentation & Preprocessing
* Applied Albumentations pipelines: `Resize(256, 256)`, `HorizontalFlip`, `Rotate(15°)`, `RandomBrightnessContrast`, and ImageNet z-score normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)[cite: 21].

---

## 🔬 Model Architecture
Input MRI (3 x 256 x 256)
│
▼
ResNet-34 Encoder (ImageNet Pretrained)
├── Bottleneck Features ──► AdaptiveAvgPool ──► Dense(512->256) ──► Dropout(0.3) ──► 4-Class Output
│
└── Feature Skip Connections ──► U-Net Decoder ──► Segmentation Head ──► Binary Mask (1 x 256 x 256)

* **Joint Loss Formulation:**
  $$\mathcal{L}_{total} = \mathcal{L}_{Dice}(\hat{M}, M) + \mathcal{L}_{CrossEntropy}(\hat{Y}, Y)$$
 [cite: 21]
* **Optimization:** Adam optimizer ($\text{lr} = 10^{-4}$) coupled with a `ReduceLROnPlateau` learning rate scheduler ($\text{factor} = 0.1$, $\text{patience} = 2$) monitoring validation loss[cite: 21].

---

## 📈 Benchmark Results

Evaluated on the 1,000 unseen test MRI samples[cite: 21]:

### 1. Classification Performance
* **Overall Test Accuracy:** **99.00%**[cite: 21]
* **Multi-Class One-vs-Rest ROC-AUC:** **0.9990**[cite: 21]

| Diagnostic Class | Precision | Recall (Sensitivity) | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **no_tumor** | 0.99 | **1.00** | 1.00 | 140[cite: 21] |
| **meningioma** | 0.98 | 0.99 | 0.99 | 306[cite: 21] |
| **pituitary** | 0.99 | 0.99 | 0.99 | 300[cite: 21] |
| **glioma** | 0.99 | 0.99 | 0.99 | 254[cite: 21] |
| **Macro Average** | **0.99** | **0.99** | **0.99** | **1,000**[cite: 21] |

### 2. Segmentation Performance
* **Base U-Net (Joint Multi-Task Training):** **75.31% Mean Dice Score**[cite: 21]
* **scSE Attention U-Net (Segmentation-Only):** **75.85% Mean Dice Score** / **80.62% mIoU**[cite: 21]

---

## 💡 Key Findings

1. **Multi-Task Synergies:** Joint multi-task learning achieves top-tier classification performance (99.00%) without requiring separate backbone encoders, drastically reducing parameter overhead[cite: 21].
2. **Zero False Positives on Controls:** The model achieved 100% sensitivity on non-tumorous control scans (140/140), preventing false alarms on healthy tissue[cite: 21].
3. **Attention Gates vs. Base U-Net:** Adding spatial and channel squeeze-and-excitation (scSE) blocks yielded a marginal +0.54% Dice score gain on single-seed evaluation, suggesting baseline U-Net with ResNet-34 encoder provides sufficient representational capacity for standard clinical contrast[cite: 21].

---

## 🛠️ Tech Stack

* **Deep Learning Framework:** PyTorch & `segmentation_models_pytorch`[cite: 21]
* **Augmentation:** `albumentations`[cite: 21]
* **Computer Vision:** `opencv-python`[cite: 21]
* **Metrics & Evaluation:** `scikit-learn`, `matplotlib`, `seaborn`[cite: 21]
* **Dataset Management:** `kagglehub`[cite: 21]
