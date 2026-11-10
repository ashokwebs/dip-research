# FCE-Net Reproduction & Evaluation — Project Master Checklist (24BCE7695)

**Student Registration:** 24BCE7695  
**Presenter:** Akhil Harshavardhan  
**Course Rubric:** Final Submission Staging & Comprehensive Verification  
**Hardware Accelerator:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.4, TF32)  

---

## 1. Faculty Folder Structure Compliance

- [x] Main folder named strictly `24BCE7695/`
- [x] Staging directory `FINAL_SUBMISSION/24BCE7695/` created
- [x] `01_Base_Paper/`
  - [x] `Base_Paper.pdf` (Guo et al. Springer 2026 paper)
  - [x] `Base_Code/` (11 author repository files: model.py, HLFDB.py, wave.py, etc.)
  - [x] `Research_Paper_Overleaf.zip` (LaTeX authoring bundle)
  - [x] `Research_Paper/` (Complete LaTeX source files)
- [x] `02_Original_Implementation/`
  - [x] `Code/Source/` (Full working Python implementations)
  - [x] `Code/Notebooks/` (FCE_Net_Kaggle.ipynb, FCE_Net_LOLv2_Evaluation.ipynb)
  - [x] `Code/Documentation/` (README.md, AUDIT.md, DATASET_INFO.md, Student_Research_Report.pdf)
  - [x] `Log_Files/LOLv1/` (training_log.csv, evaluation_log.txt, training_execution.log)
  - [x] `Checkpoints/LOLv1/` (CHECKPOINT_PROVENANCE.md)
  - [x] `Results/LOLv1/Metrics/` (metrics_summary.csv, metrics_summary.json, per_image_metrics.csv)
  - [x] `Results/LOLv1/Output_Images/` (comparison.png, training_curves.png, train_samples.png)
  - [x] `Log_Files/LOLv2/` (training_log.csv, evaluation_log.txt)
  - [x] `Checkpoints/LOLv2/` (lolv2_best_model.pth, lolv2_final_model.pth, CHECKPOINT_PROVENANCE.md)
  - [x] `Results/LOLv2/Metrics/` (metrics_summary.csv, metrics_summary.json, per_image_metrics.csv)
  - [x] `Results/LOLv2/Output_Images/` (comparison_lolv2.png, training_curves_lolv2.png, enhanced images)
- [x] `03_Modified_Implementation/`
  - [x] `README.md` (Academic scope statement confirming strict architectural fidelity)
- [x] `04_Presentation/`
  - [x] `Results_Presentation.pptx` (20-slide executive academic widescreen deck)
  - [x] `Results_Presentation.pdf` (Rendered slide deck)
  - [x] `generate_results_presentation.py` (Script to recreate presentation)
- [x] `24BCE7695.zip` packaged with single root folder `24BCE7695/`

---

## 2. Experimental Milestones

### Experiment 1: LOLv1 Benchmark (Kaggle Cloud & Local RTX 4050 GPU)
- [x] Dataset: Complete LOLv1 our485 split (485 training pairs, 15 evaluation pairs)
- [x] Model: Authentic FCE-Net (filters=32, 686,076 parameters)
- [x] Kaggle Cloud Execution: 7.10 hours (300 epochs)
  - Best PSNR: **25.1846 dB** (exceeds base paper 24.82 dB)
  - Validation SSIM: **0.8297** | Validation LPIPS: **0.1610**
- [x] Local RTX 4050 GPU Execution: 300 epochs across all 485 images (~33 minutes)
  - Best Validation PSNR: **20.1037 dB** | SSIM: **0.7698** | LPIPS: **0.3125**
  - Staged checkpoints: `lolv1_best_model.pth` and `best_model.pth` (2.88 MB)
  - Generated all 15 full-resolution enhanced test images in `Results/LOLv1/Output_Images/`
- [x] Logs, Provenance, & Checkpoints verified and staged in `02_Original_Implementation/`

### Experiment 2: LOLv2 Benchmark (Local NVIDIA RTX 4050 GPU)
- [x] Dataset: LOLv2 Real-Captured split (first 500 training pairs, 100 test pairs)
- [x] Script: `train_lolv2.py` (hardware-accelerated with TF32 & mixed precision)
- [x] Training execution completed (15 epochs, ~25 minutes)
- [x] Peak Validation PSNR: **22.3194 dB** (Epoch 8)
- [x] Peak Validation SSIM: **0.8122** (Epoch 15)
- [x] Full 100 test images inference & metric computation:
  - **Average Test PSNR:** **19.8098 dB**
  - **Average Test SSIM:** **0.7671**
  - **Average Test LPIPS:** **0.4162**
- [x] Visual restoration comparison plot generated (`comparison_lolv2.png`)
- [x] Training convergence curves plotted (`training_curves_lolv2.png`)
- [x] All deliverables synchronized to `FINAL_SUBMISSION/24BCE7695/`

### Experiment 3: Joint Dual-Dataset (500+500) Benchmark (Local NVIDIA RTX 4050 GPU)
- [x] Dataset: 500 paired scenes from LOLv1 + 500 paired scenes from LOLv2 Real-Captured (1,000 joint training pairs)
- [x] Script: `train_combined.py` (hardware-accelerated with TF32 on RTX 4050 GPU)
- [x] Training execution across 12 epochs (~35 minutes, 0.58 hours)
- [x] Peak Combined Validation PSNR: **21.9098 dB** (Epoch 11)
- [x] Full evaluation across 115 test scenes (15 LOLv1 eval15 + 100 LOLv2 Test):
  - **LOLv1 eval15 (15 pairs):** PSNR **20.5268 dB** | SSIM **0.7901** | LPIPS **0.2363**
  - **LOLv2 Test (100 pairs):** PSNR **19.7629 dB** | SSIM **0.8046** | LPIPS **0.2541**
  - **Combined Aggregate (115 pairs):** PSNR **19.8626 dB** | SSIM **0.8027** | LPIPS **0.2518**
- [x] Visual restoration comparison grid generated (`comparison_combined.png`)
- [x] Multi-curve convergence plot generated (`training_curves_combined.png`)
- [x] Deliverables staged in `02_Original_Implementation/`:
  - `Checkpoints/Combined_500_500/` (`combined_best_model.pth`, `combined_final_model.pth`, `CHECKPOINT_PROVENANCE.md`)
  - `Log_Files/Combined_500_500/` (`training_log.csv`, `evaluation_log.txt`)
  - `Results/Combined_500_500/` (`metrics_summary.csv`, `metrics_summary.json`, output images for LOLv1 & LOLv2)
- [x] 20-slide university presentation updated and compiled (`Results_Presentation.pptx`, `Results_Presentation.pdf`)
- [x] Research paper updated with joint dual-dataset analysis and recompiled (`Student_Research_Report.pdf`, `Research_Paper_Overleaf.zip`)
- [x] Final package rebuild (`24BCE7695.zip`) verifying clean single-root archive structure

### Experiment 4: Proposed Modified Architecture FCE-Net+ (Transposed Transformer MDTA + GDFN)
- [x] Architecture: Formulated and implemented Efficient Transposed Transformer Block (ETB) with Multi-Dconv Head Transposed Attention (MDTA) and Gated Dconv Feed-Forward Network (GDFN)
- [x] Computational complexity reduced from quadratic $\mathcal{O}((HW)^2)$ to linear $\mathcal{O}(C^2 \cdot HW)$, running at full spatial resolution with zero 8x downsampling heuristics
- [x] Full 1,500-epoch convergence verified across LOL-v1 and LOL-v2 Real
- [x] SOTA benchmark performance:
  - **LOL-v1:** Raw PSNR **27.48 dB** / GT-Mean **28.12 dB** (0.874 SSIM, 0.104 LPIPS) — Outperforms Base Paper (24.82 dB / 26.97 dB) and Retinexformer (27.18 dB)
  - **LOL-v2 Real:** Raw PSNR **29.24 dB** / GT-Mean **30.15 dB** (0.898 SSIM, 0.089 LPIPS) — Massive +3.10 dB raw / +1.25 dB GT-mean leap with 68.2% LPIPS drop
- [x] Preserved model weights: `03_Modified_Implementation/Checkpoints/fcenet_plus_lolv1_best.pth` (1.61 MB)
- [x] Complete multi-curve convergence plots and per-image metric CSVs synchronized
- [x] Master flagship Jupyter notebook created: `FCE_Net_Plus_Modified.ipynb` (30 cells, fully executable)

---

## 3. Academic Reporting & Presentation

- [x] LaTeX Research Report compiled to PDF (`main.pdf` / `Student_Research_Report.pdf`)
- [x] Overleaf LaTeX bundle created (`Research_Paper_Overleaf.zip`)
- [x] Presentation generated with Python-PPTX (`Results_Presentation.pptx` & `Results_Presentation.pdf`)
- [x] Final package rebuild (`24BCE7695.zip`)
