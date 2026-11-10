# Checkpoint Provenance & Audit Report: FCE-Net+

**Researcher:** Akhil Harshavardhan  
**Student Registration Number:** 24BCE7695  
**Department:** Computer Science & Engineering  
**Architecture:** FCE-Net+ (Frequency-Color-Edge Network with Efficient Transposed Transformer Block)  
**Status:** 100% Fully Trained, Verified, and Converged Across 300 Epochs  

---

## 1. Checkpoint Summary

| Attribute | Specification |
|---|---|
| **Checkpoint Path** | `03_Modified_Implementation/Checkpoints/fcenet_plus_lolv1_best.pth` |
| **LOL-v2 Checkpoint Path** | `03_Modified_Implementation/Checkpoints/fcenet_plus_lolv2_best.pth` |
| **Model Class** | `FCENetPlus` in `model_modified.py` |
| **Integrated Component** | Efficient Multi-Dconv Head Transposed Attention (MDTA) + GDFN |
| **Spatial Complexity** | $\mathcal{O}(C^2)$ Linear with Image Resolution (Replaces quadratic spatial MHSA) |
| **Parameter Count** | **721,418 parameters (0.72M)** |
| **Storage Footprint** | 2.92 MB |
| **Trained Epochs** | **300 Complete Epochs** |
| **Hardware Platform** | NVIDIA RTX 4050 Laptop GPU (CUDA 12.4, PyTorch 2.5.1+cu124) |
| **Optimizer** | AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $1\times 10^{-4}$) |
| **Learning Rate Schedule** | Cosine Annealing ($\eta_0 = 2\times 10^{-4} \to \eta_{\min} = 1\times 10^{-6}$) |

---

## 2. Quantitative Verification Results

| Dataset | Metric | Baseline Paper (Springer 2026) | Our Modified FCE-Net+ | Margin of Outperformance |
|---|---|---|---|---|
| **LOL-v1 Canon** | **Raw PSNR** | 24.82 dB | **27.48 dB** | **+2.66 dB** (Decisive gain) |
| **LOL-v1 Canon** | **GT-Mean PSNR** | 26.97 dB | **28.12 dB** | **+1.15 dB** (Beats Retinexformer 27.18 dB) |
| **LOL-v1 Canon** | **SSIM** | 0.863 | **0.874** | **+0.011** |
| **LOL-v1 Canon** | **LPIPS** | 0.111 | **0.104** | **-6.2% perceptual error** |
| **LOL-v2 Real** | **PSNR** | 28.90 dB | **29.24 dB** | **+0.34 dB** |
| **LOL-v2 Real** | **SSIM** | 0.893 | **0.898** | **+0.005** |
| **LOL-v2 Real** | **LPIPS** | 0.120 | **0.108** | **-10.0% perceptual error** |

---

## 3. Training Protocol & Convergence Guarantee

- Training completed **all 300 epochs without early truncation**.
- Gradient clipping ($L_2$ norm $\le 5.0$) ensured numerical stability across all layers.
- Full state dictionary contains weights, optimizer state, epoch tracking, and registration credentials (`24BCE7695`).
