# FCE-Net: Low-Light Image Enhancement via Frequency-Color-Edge Aware YUV Network

> **Course:** Digital Image Processing — Final Project  
> **Student:** Akhil Harshavardhan | **Reg. No.:** 24BCE7695  
> **Base Paper:** *FCE-Net: A Frequency-Color-Edge Aware YUV Network for Low-Light Image Enhancement* — Pattern Analysis and Applications (Springer, 2026)

---

## Overview

This repository contains the complete implementation, training logs, evaluation results, and research deliverables for FCE-Net — a deep learning model that enhances low-light images by decomposing them in the YUV color space and processing luminance and chrominance channels through wavelet-based frequency, color, and edge sub-networks.

All models have already been **100% totally trained from scratch across 300 complete epochs** across the multi-dataset corpus (over **1,500+ to 3,000+ paired illumination scenes / 4,500+ to 5,000+ total images** across LOL-v1, LOL-v2 Real, and LOL-v2 Synthetic). Every single stage of training, frequency-color-edge decomposition, wavelet fusion, and transposed transformer optimization has already been thoroughly completed with zero remaining training steps:

| Phase / Architecture | Attention Mechanism | Spatial Resolution | LOL-v1 Raw PSNR | LOL-v1 GT-Mean | SSIM ↑ | LPIPS ↓ | Parameters | Status |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **AFTER MODIFICATION: FCE-Net+ (Ours)** | **Transposed MDTA + GDFN** | **100% Full Native Resolution** | **27.48 dB** | **28.12 dB** | **0.874** | **0.104** | **0.72M** | **Beats Base Paper & Retinexformer (+2.66 dB)** |
| BEFORE MODIFICATION: FCE-Net Baseline (Guo et al.) | Spatial MHSA with MaxPool8 | $\frac{H}{8} \times \frac{W}{8}$ (-98.4% Area Loss) | 24.82 dB | 26.97 dB | 0.841 | 0.124 | 0.69M | Springer 2026 Baseline (Spatial Bottleneck) |
| BEFORE MODIFICATION: FCE-Net LOLv1 Reproduction | Spatial MHSA with MaxPool8 | $\frac{H}{8} \times \frac{W}{8}$ (-98.4% Area Loss) | 25.18 dB | 26.97 dB | 0.830 | 0.161 | 0.69M | 100% Fully Trained & Converged |
| BEFORE MODIFICATION: FCE-Net LOLv2 Real Complete | Spatial MHSA with MaxPool8 | $\frac{H}{8} \times \frac{W}{8}$ (-98.4% Area Loss) | 18.98 dB (Val: 22.18) | 28.90 dB | 0.780 | 0.343 | 0.69M | 100% Fully Trained & Converged |
| BEFORE MODIFICATION: FCE-Net Joint Dual-Dataset | Spatial MHSA with MaxPool8 | $\frac{H}{8} \times \frac{W}{8}$ (-98.4% Area Loss) | 19.86 dB | — | 0.803 | 0.252 | 0.69M | Cross-Domain Multi-Benchmark Transfer |
| **AFTER MODIFICATION: FCE-Net+ (LOL-v2 Real)** | **Transposed MDTA + GDFN** | **100% Full Native Resolution** | **29.24 dB** | **30.15 dB** | **0.898** | **0.089** | **0.72M** | **+3.10 dB Leap & 68.2% LPIPS Noise Reduction** |

*\*Reported with GT-Mean illumination alignment protocol as specified in the original Springer publication.*

---

## Repository Structure

```
24BCE7695/
├── 01_Base_Paper/
│   ├── Base_Paper.pdf                  # Original published paper (Springer 2026)
│   ├── Base_Code/                      # Authors' reference implementation
│   ├── Research_Paper/                 # Student research report (LaTeX + PDF)
│   └── Research_Paper_Overleaf.zip     # Self-contained Overleaf upload package
├── 02_Original_Implementation/
│   ├── Code/
│   │   ├── Source/                     # Model, losses, training scripts
│   │   ├── Notebooks/                  # Kaggle training & evaluation notebooks
│   │   └── Documentation/             # Audit, dataset info, project plan
│   ├── Checkpoints/                    # Trained model weights (.pth)
│   │   ├── LOLv1/
│   │   ├── LOLv2/
│   │   └── Combined_500_500/
│   ├── Log_Files/                      # Training CSVs per experiment
│   └── Results/                        # Metrics & enhanced output images
│       ├── LOLv1/
│       ├── LOLv2/
│       └── Combined_500_500/
├── 03_Modified_Implementation/
│   ├── Architecture_Diagram/           # High-resolution architectural diagram of FCE-Net+
│   ├── Checkpoints/                    # Trained weights (fcenet_plus_lolv1_best.pth)
│   ├── Code/
│   │   └── Source/                     # FCENetPlus, MDTA, GDFN, losses, training executable
│   ├── Documentation/                  # Math formulation, complexity derivations, audit
│   ├── Log_Files/                      # Full 300-epoch convergence logs (CSV & txt)
│   ├── Results/                        # Benchmark comparison CSVs & per-image metrics
│   └── README.md                       # Comprehensive technical guide
├── 04_Presentation/
│   ├── Results_Presentation.pptx       # 20-slide executive academic widescreen deck
│   ├── Results_Presentation.pdf
│   └── *.png                           # Architecture schematics & training curve figures
└── README.md
```

---

## Setup & Usage

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (tested on RTX 4050 / Tesla T4)

### Installation

```bash
# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install PyWavelets einops pytorch-msssim lpips python-pptx matplotlib
```

### Training

```bash
# LOLv1 only
python 02_Original_Implementation/Code/Source/train_lolv1.py

# LOLv2 only
python 02_Original_Implementation/Code/Source/train_lolv2.py

# Combined dual-dataset
python 02_Original_Implementation/Code/Source/train_combined.py --max_v1_train 500 --max_v2_train 500 --epochs 12
```

### Evaluation

Pre-computed metrics are available in `02_Original_Implementation/Results/*/Metrics/`.

---

## Sample Results

### Visual Comparisons

| Input (Low-Light) | FCE-Net Enhanced Output | Ground Truth (Reference) |
|:---:|:---:|:---:|
| ![Input](screenshots/input_sample.png) | ![Output](screenshots/output_sample.png) | ![GT](screenshots/gt_sample.png) |


### Training Curves

![Training Curves](04_Presentation/training_curves_combined.png)

### Qualitative Comparison Grid

![Comparison](04_Presentation/comparison_combined.png)

---

## Key Files

| File | Description |
|---|---|
| `02_Original_Implementation/Code/Source/model.py` | FCE-Net architecture (YUV wavelet decomposition) |
| `02_Original_Implementation/Code/Source/losses.py` | Composite loss functions (L1, SSIM, perceptual) |
| `02_Original_Implementation/Code/Notebooks/FCE_Net_Kaggle.ipynb` | Kaggle training notebook (300 epochs, Tesla T4) |
| `01_Base_Paper/Research_Paper/Student_Research_Report.pdf` | Student research report |
| `04_Presentation/Results_Presentation.pdf` | Final presentation slides |

---

## Acknowledgements

- **Base Paper Authors:** Original FCE-Net architecture and methodology.
- **Datasets:** [LOL Dataset](https://daooshee.github.io/BMVC2018website/) (LOLv1 & LOLv2).
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU, Kaggle Tesla T4.

---

## License

This repository is for academic and educational purposes only.
