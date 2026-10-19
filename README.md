# High-Frequency Aware Restormer for Document Deshadowing

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Benchmark](https://img.shields.io/badge/RDD%20PSNR-33.21%20dB-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)

An enhanced **Dual-Stream Restormer** architecture tailored specifically for **Document Deshadowing** (shadow removal on scanned and camera-captured text documents), preserving sharp character ink strokes while generating clean, artifact-free paper backgrounds.

---

## 🌟 Key Innovations & Contributions

1. **Lightweight High-Frequency Branch (HFB)**:
   - Uses non-parametric high-pass filtering ($x - \text{LowPass}(x)$) followed by Depthwise-Separable convolutions to isolate and restore fine character strokes from broad shadow illumination gradients.
2. **Adaptive Gated Spatial-Frequency Fusion (GSFF)**:
   - Dynamically learns a pixel-wise Spatial Gate Map $\mathbf{G} \in [0, 1]$:
     $$\mathbf{G} = \sigma\Big(\text{Conv}_{1\times 1}\big(\text{LReLU}(\text{Conv}_{3\times 3}([\mathbf{F}_{\text{main}}, \mathbf{F}_{\text{hf}}]))\big)\Big)$$
     $$\hat{\mathbf{I}} = \mathbf{F}_{\text{main}} + \mathbf{G} \odot \mathbf{F}_{\text{hf}}$$
   - Injects sharp high-frequency details strictly at character boundaries while suppressing chromatic noise on uniform paper backgrounds.
3. **Dual-Domain Fourier Loss ($\mathcal{L}_{\text{freq}}$)**:
   - Supervises spectral consistency between reconstructions and ground truth targets in the 2D Fast Fourier Transform (FFT) domain.

---

## 📊 Benchmark Results on RDD Dataset (545 Test Images)

| Method | Type / Venue | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | RMSE $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| **Input Degraded** | Camera Capture | 18.24 | 0.8120 | 32.14 |
| **AOD-Net** | ICCV 2017 | 23.41 | 0.8845 | 18.92 |
| **DHAN** | AAAI 2020 | 27.12 | 0.9234 | 12.05 |
| **Restormer Baseline** | CVPR 2022 | 28.15 | 0.9410 | 9.87 |
| 🏆 **Ours (Full Model)** | **Proposed (Dual-Stream + GSFF)** | **33.21** | **0.9712** | **5.64** |

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Lloyd2103/Deep_Learning.git
cd Deep_Learning
pip install -r requirements.txt
python setup.py develop
```

### 2. Inference / Demo
To deshadow a single document image:
```bash
python3 Deshadowing/demo.py --input path/to/shadowed_image.jpg --output clean_result.png
```

To process an entire directory of document images:
```bash
python3 Deshadowing/demo.py --input path/to/input_folder/ --output path/to/output_folder/
```

### 3. Training
To train the model on the full Real Document Deshadowing (RDD) benchmark:
```bash
python3 basicsr/train.py -opt Deshadowing/Options/Deshadowing_Restormer.yml
```

---

## 📑 IEEE Report & Documentation
The complete formal academic report formatted under the IEEE conference standard (`IEEEtran`) is available at:
- `Deshadowing/report_ieee.tex`
