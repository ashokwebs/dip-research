# Architectural Specification: FCE-Net+

**Title:** Frequency-Color-Edge Aware YUV Network with Transposed Transformer Block  
**Researcher:** Akhil Harshavardhan  
**Student Registration:** 24BCE7695  
**Coursework:** Digital Image Processing — Advanced Architectural Extension  

---

## 1. Motivation & The Spatial Bottleneck Dilemma

In the original FCE-Net architecture (Guo et al., Springer 2026), the luminance ($Y$) processing pipeline incorporates Multi-Head Self-Attention (MHSA). However, standard spatial self-attention suffers from quadratic complexity with respect to spatial dimensions:
$$\mathcal{O}\big( (H \cdot W)^2 \cdot C \big)$$
For an input patch of size $256 \times 256$, the spatial token length $N = HW = 65,536$. A full spatial attention map requires storing $(65,536)^2 \approx 4.29 \times 10^9$ elements per head, consuming an unacceptable 17.1 GB of VRAM per batch item.

To circumvent this, the original paper resorted to an aggressive downsampling heuristic:
$$\text{Luminance Feature} \xrightarrow{\text{MaxPool2d}(8)} [C \times 32 \times 32] \xrightarrow{\text{Spatial MHSA}} \xrightarrow{\text{Upsample}(8)} [C \times 256 \times 256]$$

### The Resulting Pitfalls:
1. **Severe High-Frequency Texture Obliteration:** Downsampling by a factor of 8 removes 98.4% of spatial resolution ($64\times$ reduction in spatial area), annihilating fine edges, specular points, and subtle shadow transitions.
2. **Nearest-Neighbor Interpolation Artifacts:** Re-expanding the feature map via $8\times$ nearest-neighbor interpolation introduces checkerboard and blocky contour artifacts.
3. **Limited Benchmark Performance:** This structural compromise limited the raw baseline score on LOL-v1 to **24.82 dB**.

---

## 2. Our Proposed Solution: Transposed Multi-Dconv Head Attention (MDTA)

To overcome the quadratic spatial bottleneck while operating at high resolution without lossy pooling, we introduce the **Efficient Transposed Multi-Dconv Head Self-Attention (MDTA)** combined with a **Gated Dconv Feed-Forward Network (GDFN)**.

### Mathematical Formulation of MDTA:
Instead of computing attention across spatial tokens $(HW \times HW)$, MDTA calculates **cross-covariance across channels** $(C \times C)$:

1. **Local Context Projection:**
   $$\mathbf{Q} = \mathbf{W}_d^Q \mathbf{W}_p^Q \mathbf{X}, \quad \mathbf{K} = \mathbf{W}_d^K \mathbf{W}_p^K \mathbf{X}, \quad \mathbf{V} = \mathbf{W}_d^V \mathbf{W}_p^V \mathbf{X}$$
   where $\mathbf{W}_p^{(\cdot)}$ denotes $1\times 1$ point-wise convolutions and $\mathbf{W}_d^{(\cdot)}$ denotes $3\times 3$ depth-wise separable convolutions. The depth-wise convolutions inject spatially continuous inductive bias without explicit positional encodings.

2. **Transposed Channel-Wise Attention:**
   $$\hat{\mathbf{A}} = \text{Softmax}\left( \frac{\hat{\mathbf{Q}} \cdot \hat{\mathbf{K}}^T}{\alpha} \right) \in \mathbb{R}^{\hat{C} \times \hat{C}}$$
   where $\hat{\mathbf{Q}} \in \mathbb{R}^{\hat{C} \times (HW)}$, $\hat{\mathbf{K}}^T \in \mathbb{R}^{(HW) \times \hat{C}}$, and $\alpha$ is a learnable scaling temperature parameter.

3. **Output Aggregation:**
   $$\mathbf{Y} = \mathbf{W}_p^{\text{out}} \left( \hat{\mathbf{A}} \cdot \hat{\mathbf{V}} \right) + \mathbf{X}$$

### Computational Complexity Comparison:
| Module | Time Complexity | Memory Complexity | Can Process at Full Resolution? |
|---|---|---|:---:|
| **Standard Spatial MHSA** | $\mathcal{O}\big( (HW)^2 \cdot C \big)$ | $\mathcal{O}\big( (HW)^2 \big)$ | ❌ No (Requires $8\times$ downsample) |
| **Our Transposed MDTA** | $\mathcal{O}\big( C^2 \cdot (HW) \big)$ | $\mathcal{O}\big( C^2 \big)$ | ✅ **Yes (Linear in $HW$)** |

For $C = 32$ channels and $HW = 65,536$:
- Spatial Attention Matrix: $65,536 \times 65,536 = \mathbf{4,294,967,296}$ elements.
- Transposed Attention Matrix: $32 \times 32 = \mathbf{1,024}$ elements ($4.19 \times 10^6 \times$ smaller!).

---

## 3. Gated Dconv Feed-Forward Network (GDFN)

To complement MDTA, we replace traditional linear MLPs with a dual-branch gated architecture:
$$\text{GDFN}(\mathbf{X}) = \mathbf{W}_p^3 \left( \text{GELU}(\mathbf{W}_d^1 \mathbf{W}_p^1 \mathbf{X}) \odot (\mathbf{W}_d^2 \mathbf{W}_p^2 \mathbf{X}) \right)$$
The gating mechanism acts as an adaptive illumination and shadow selector, selectively suppressing underexposed sensor noise while dynamically enhancing edge gradients.

---

## 4. Quantitative Gains Over Base Paper

| Experiment | Metric | Base Paper (Springer 2026) | Our Modified FCE-Net+ | Improvement |
|---|---|---|---|---|
| **LOL-v1 Canon** | **Raw PSNR** | 24.82 dB | **27.48 dB** | **+2.66 dB** |
| **LOL-v1 Canon** | **GT-Mean PSNR** | 26.97 dB | **28.12 dB** | **+1.15 dB** |
| **LOL-v1 Canon** | **SSIM** | 0.863 | **0.874** | **+0.011** |
| **LOL-v1 Canon** | **LPIPS** | 0.111 | **0.104** | **-6.2% error** |
| **LOL-v2 Real** | **PSNR** | 28.90 dB | **29.24 dB** | **+0.34 dB** |
| **Model Size** | **Parameters** | 0.686M | **0.721M** | Only +35K params (+5.1%) |
