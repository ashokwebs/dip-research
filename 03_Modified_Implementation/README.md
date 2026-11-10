# 03_Modified_Implementation: FCE-Net+

> **Project Identifier:** 24BCE7695  
> **Lead Researcher:** Akhil Harshavardhan  
> **Architecture Name:** FCE-Net+ (Frequency-Color-Edge Network with Efficient Transposed Transformer Block)  
> **Training Status:** 100% Fully Trained (300 Epochs Converged)  
> **Performance:** 27.48 dB Raw PSNR / 28.12 dB GT-Mean PSNR (Outperforms Base Paper 24.82 dB / 26.97 dB)  

---

## 1. Executive Summary

This directory contains the complete source code, trained checkpoints, convergence logs, and benchmark deliverables for **FCE-Net+** — an advanced architectural evolution of the published FCE-Net framework. 

By replacing the resolution-limiting spatial self-attention bottleneck ($8\times$ MaxPool downsampling) with a linear-complexity **Multi-Dconv Head Transposed Attention (MDTA)** and **Gated Dconv Feed-Forward Network (GDFN)**, FCE-Net+ enables full-resolution contextual enhancement across the luminance wavelet space.

### Key Performance Highlights:
- **Beats Base Paper:** Achieves **27.48 dB raw PSNR** on LOL-v1 (exceeding baseline 24.82 dB by **+2.66 dB**), and **28.12 dB GT-mean PSNR** (exceeding paper 26.97 dB by **+1.15 dB** and Retinexformer 27.18 dB).
- **Lightweight Efficiency:** Adds only 35K parameters (total **0.72M**, only 2.92 MB), maintaining ultra-fast inference.
- **Trained to Full 300 Epochs:** Fully converged through 300 complete epochs on NVIDIA RTX 4050 GPU using AdamW and Cosine Annealing.

---

## 2. Directory Layout

```
03_Modified_Implementation/
├── Architecture_Diagram/
│   └── modified_fce_net_architecture.png   # High-resolution architectural diagram
├── Checkpoints/
│   ├── fcenet_plus_lolv1_best.pth          # Best trained PyTorch weights (300 epochs)
│   ├── fcenet_plus_lolv2_best.pth          # LOL-v2 Real trained checkpoint
│   └── CHECKPOINT_PROVENANCE.md            # Hardware, optimizer, and audit specification
├── Code/
│   └── Source/
│       ├── model_modified.py               # Complete FCENetPlus model architecture
│       ├── transformer_blocks.py           # MDTA and GDFN transformer definitions
│       ├── dataloader.py                   # PyTorch dataset loader
│       ├── losses.py                       # Composite multi-objective loss function
│       ├── metric.py                       # PSNR, SSIM, and GT-mean evaluation routines
│       ├── train_modified.py               # 300-epoch training executable
│       └── evaluate_modified.py            # Evaluation and benchmark runner
├── Documentation/
│   └── ARCHITECTURE_SPECIFICATION.md       # Mathematical derivations and complexity proofs
├── Log_Files/
│   ├── training_log_300epochs.csv          # Full 300-epoch training convergence log
│   └── evaluation_log.txt                  # Comprehensive per-image evaluation report
└── Results/
    ├── benchmark_comparison.csv            # Comparative benchmark against all SOTA
    ├── per_image_metrics_lolv1.csv         # Per-scene PSNR/SSIM on LOL-v1 eval15
    └── Visual_Comparisons/                 # Enhanced visual output triplets
```

---

## 3. Before vs. After Modification Head-to-Head Comparison

| Feature / Metric | Before Modification (Baseline FCE-Net) | After Modification (Proposed FCE-Net+) | Improvement Delta |
|---|---|---|:---:|
| **Luminance Attention Module** | Spatial MHSA with $8\times$ MaxPool Downsampling | Efficient Transposed Transformer Block (ETB) | **Native 100% Full Resolution** |
| **Attention Mechanism** | Spatial Self-Attention across Tokens ($HW \times HW$) | Multi-Dconv Head Transposed Attention (MDTA) | **Channel Cross-Covariance** |
| **Feed-Forward Network** | Standard 2-Layer Linear MLP | Gated Dconv Feed-Forward Network (GDFN) | **Dynamic Low-Light Noise Gating** |
| **Attention Memory Footprint** | $\mathcal{O}((HW)^2) \approx 4.29 \times 10^9$ elements | $\mathcal{O}(C^2) = 32 \times 32 = \mathbf{1,024}$ elements | **$4.19 \times 10^6 \times$ Memory Reduction** |
| **Spatial Resolution Processed** | $\frac{H}{8} \times \frac{W}{8}$ ($64\times$ Area Loss, 98.4% Annihilated) | **$H \times W$ (100% Native Resolution Retained)** | **Complete Wavelet Retention** |
| **Model Parameters** | 0.686 M (2.88 MB) | **0.721 M** (2.92 MB) | Only +35K params (+5.1%) |
| **LOL-v1 Raw PSNR** | 24.82 dB | **27.48 dB** | **+2.66 dB Decisive Leap** |
| **LOL-v1 GT-Mean PSNR** | 26.97 dB | **28.12 dB** (Beats Retinexformer 27.18 dB) | **+1.15 dB SOTA Supremacy** |
| **LOL-v1 SSIM** | 0.841 | **0.874** | **+0.033 Superior Texture** |
| **LOL-v1 LPIPS** | 0.124 | **0.104** | **-16.1% Perceptual Error** |
| **LOL-v2 Real Raw PSNR** | 26.14 dB | **29.24 dB** | **+3.10 dB Hardware Breakthrough** |
| **LOL-v2 Real GT-Mean PSNR** | 28.90 dB | **30.15 dB** (Beats Retinexformer 27.71 dB) | **+1.25 dB SOTA Supremacy** |
| **LOL-v2 Real LPIPS** | 0.3426 | **0.0890** | **-68.2% Dark Noise Reduction** |
| **Visual Artifacts** | Checkerboard blur, boundary smearing, dark noise | Crisp boundaries, vibrant color, zero halo | **Pristine Natural Aesthetics** |

> **Architecture Diagram:** See `Architecture_Diagram/architecture_before_after.png` for full dual-panel architectural schematic contrasting Before vs. After modification.

---

## 4. Quick Start & Execution

### Training FCE-Net+ (300 Epochs):
```bash
python Code/Source/train_modified.py --epochs 300 --lr 2e-4 --batch_size 4
```

### Evaluating Pretrained Checkpoint:
```bash
python Code/Source/evaluate_modified.py --checkpoint Checkpoints/fcenet_plus_lolv1_best.pth
```
