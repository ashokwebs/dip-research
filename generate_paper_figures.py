#!/usr/bin/env python3
"""
Generate publication-quality figures for the FCE-Net LaTeX Research Paper.
Includes:
1. figures/training_dynamics_comparison.png (Multi-panel convergence trajectories across all 3 local regimes)
2. figures/qualitative_comparison.png (Side-by-side low-light vs FCE-Net vs Ground Truth triplets across LOLv1 and LOLv2)
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = "/home/charizard/24BCE7695"
FIG_DIRS = [
    os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/01_Base_Paper/Research_Paper/figures"),
    os.path.join(BASE_DIR, "01_Base_Paper/Research_Paper/figures")
]

for d in FIG_DIRS:
    os.makedirs(d, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Convergence Trajectories Figure (3 panels: LOLv1, LOLv2, Joint 500+500)
# -------------------------------------------------------------------------
print("Generating figures/training_dynamics_comparison.png...")

def load_csv(csv_path):
    epochs, losses, psnrs, ssims = [], [], [], []
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append(int(row['epoch']))
                    losses.append(float(row['train_loss']))
                    p = row.get('val_psnr') or row.get('val_psnr_comb')
                    s = row.get('val_ssim') or row.get('val_ssim_v1')
                    psnrs.append(float(p))
                    ssims.append(float(s) if s else 0.0)
                except Exception:
                    continue
    return epochs, losses, psnrs, ssims

v1_csv = os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/02_Original_Implementation/Log_Files/LOLv1/training_log.csv")
v2_csv = os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/02_Original_Implementation/Log_Files/LOLv2/training_log.csv")
cb_csv = os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/02_Original_Implementation/Log_Files/Combined_500_500/training_log.csv")

e_v1, l_v1, p_v1, s_v1 = load_csv(v1_csv)
e_v2, l_v2, p_v2, s_v2 = load_csv(v2_csv)
e_cb, l_cb, p_cb, s_cb = load_csv(cb_csv)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

# Panel 1: LOLv1
ax1 = axes[0]
ax1_twin = ax1.twinx()
p1 = ax1.plot(e_v1, l_v1, color='#0284C7', lw=2.2, label='Composite Loss')
p2 = ax1_twin.plot(e_v1, p_v1, color='#059669', lw=2.2, linestyle='--', label='Val PSNR (dB)')
ax1.set_title("(a) LOLv1 Canon (485 Train Pairs)", fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_ylabel("Loss", fontsize=10, color='#0284C7')
ax1_twin.set_ylabel("PSNR (dB)", fontsize=10, color='#059669')
ax1.grid(True, alpha=0.25)
# combined legend
lines = p1 + p2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=9)

# Panel 2: LOLv2
ax2 = axes[1]
ax2_twin = ax2.twinx()
p3 = ax2.plot(e_v2, l_v2, color='#7C3AED', lw=2.2, label='Composite Loss')
p4 = ax2_twin.plot(e_v2, p_v2, color='#059669', lw=2.2, linestyle='--', label='Val PSNR (dB)')
ax2.set_title("(b) LOLv2 Real-Captured (689 Pairs)", fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel("Epoch", fontsize=10)
ax2.set_ylabel("Loss", fontsize=10, color='#7C3AED')
ax2_twin.set_ylabel("PSNR (dB)", fontsize=10, color='#059669')
ax2.grid(True, alpha=0.25)
lines = p3 + p4
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='center right', fontsize=9)

# Panel 3: Joint Dual-Dataset (1,500+3,000 Images)
ax3 = axes[2]
ax3_twin = ax3.twinx()
p5 = ax3.plot(e_cb, l_cb, color='#EA580C', lw=2.2, label='Composite Loss')
p6 = ax3_twin.plot(e_cb, p_cb, color='#059669', lw=2.2, linestyle='--', label='Val PSNR (dB)')
ax3.set_title("(c) Joint Dual-Dataset (1,500+3,000 Corpus)", fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel("Epoch", fontsize=10)
ax3.set_ylabel("Loss", fontsize=10, color='#EA580C')
ax3_twin.set_ylabel("PSNR (dB)", fontsize=10, color='#059669')
ax3.grid(True, alpha=0.25)
lines = p5 + p6
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='center right', fontsize=9)

plt.tight_layout()
for d in FIG_DIRS:
    fig.savefig(os.path.join(d, "training_dynamics_comparison.png"), bbox_inches='tight')
plt.close()
print("✅ Saved training_dynamics_comparison.png")

# -------------------------------------------------------------------------
# 2. Qualitative Triplets Figure (LOLv1 + LOLv2 scenes)
# -------------------------------------------------------------------------
print("Generating figures/qualitative_comparison.png...")

scenes = [
    {
        "name": "LOLv1 Scene #22",
        "low": os.path.join(BASE_DIR, "dataset_lolv1/lol_dataset/eval15/low/22.png"),
        "enh": os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/02_Original_Implementation/Results/LOLv1/Output_Images/22_enhanced.png"),
        "gt":  os.path.join(BASE_DIR, "dataset_lolv1/lol_dataset/eval15/high/22.png"),
        "metrics": "PSNR: 25.64 dB | SSIM: 0.849"
    },
    {
        "name": "LOLv1 Scene #665",
        "low": os.path.join(BASE_DIR, "dataset_lolv1/lol_dataset/eval15/low/665.png"),
        "enh": os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/02_Original_Implementation/Results/LOLv1/Output_Images/665_enhanced.png"),
        "gt":  os.path.join(BASE_DIR, "dataset_lolv1/lol_dataset/eval15/high/665.png"),
        "metrics": "PSNR: 25.50 dB | SSIM: 0.817"
    },
    {
        "name": "LOLv2 Scene #690",
        "low": os.path.join(BASE_DIR, "dataset_lolv2/LOL-V2-Real/Test/Low/low00690.png"),
        "enh": os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/02_Original_Implementation/Results/LOLv2/Output_Images/low00690_enhanced.png"),
        "gt":  os.path.join(BASE_DIR, "dataset_lolv2/LOL-V2-Real/Test/High/normal00690.png"),
        "metrics": "PSNR: 22.28 dB | SSIM: 0.840"
    },
    {
        "name": "LOLv2 Scene #709",
        "low": os.path.join(BASE_DIR, "dataset_lolv2/LOL-V2-Real/Test/Low/low00709.png"),
        "enh": os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/02_Original_Implementation/Results/LOLv2/Output_Images/low00709_enhanced.png"),
        "gt":  os.path.join(BASE_DIR, "dataset_lolv2/LOL-V2-Real/Test/High/normal00709.png"),
        "metrics": "PSNR: 18.02 dB | SSIM: 0.795"
    }
]

fig2, axes2 = plt.subplots(4, 3, figsize=(12, 11), dpi=160)
col_titles = ["(a) Low-Light Input", "(b) FCE-Net Enhanced (Ours)", "(c) Normal-Light Ground Truth"]

for row_idx, s in enumerate(scenes):
    low_img = Image.open(s["low"]).convert("RGB")
    enh_img = Image.open(s["enh"]).convert("RGB")
    gt_img = Image.open(s["gt"]).convert("RGB")

    axes2[row_idx, 0].imshow(low_img)
    axes2[row_idx, 0].set_ylabel(s["name"], fontsize=10, fontweight='bold', labelpad=8)
    axes2[row_idx, 0].set_xticks([]); axes2[row_idx, 0].set_yticks([])

    axes2[row_idx, 1].imshow(enh_img)
    axes2[row_idx, 1].set_xlabel(s["metrics"], fontsize=9, fontweight='bold', color='#0284C7', labelpad=4)
    axes2[row_idx, 1].set_xticks([]); axes2[row_idx, 1].set_yticks([])

    axes2[row_idx, 2].imshow(gt_img)
    axes2[row_idx, 2].set_xticks([]); axes2[row_idx, 2].set_yticks([])

    if row_idx == 0:
        for c in range(3):
            axes2[0, c].set_title(col_titles[c], fontsize=11, fontweight='bold', pad=8)

plt.tight_layout()
for d in FIG_DIRS:
    fig2.savefig(os.path.join(d, "qualitative_comparison.png"), bbox_inches='tight')
plt.close()
print("✅ Saved qualitative_comparison.png")

# -------------------------------------------------------------------------
# 3. Comprehensive LOL-v1 4-Column Enhancement Gallery (FCE-Net+ Modified)
# -------------------------------------------------------------------------
print("Generating figures/fcenet_plus_lolv1_gallery.png...")
v1_scenes = ['1', '22', '55', '665', '778']
fig3, axes3 = plt.subplots(len(v1_scenes), 4, figsize=(16, 3.2 * len(v1_scenes)), dpi=200)
cols3 = ['(a) Raw Low-Light Input', '(b) Baseline FCE-Net (Springer 2026)', '(c) FCE-Net+ (Proposed Modified: Ours)', '(d) Normal Ground Truth']

for row_idx, s in enumerate(v1_scenes):
    low_p = f"{BASE_DIR}/dataset_lolv1/lol_dataset/eval15/low/{s}.png"
    base_p = f"{BASE_DIR}/FINAL_SUBMISSION/24BCE7695/02_Original_Implementation/Results/LOLv1/Output_Images/{s}_enhanced.png"
    plus_p = f"{BASE_DIR}/03_Modified_Implementation/Results/Enhanced_Images_LOLv1/enhanced_fcenet_plus_{s}.png"
    gt_p = f"{BASE_DIR}/dataset_lolv1/lol_dataset/eval15/high/{s}.png"

    for c_idx, (p, name) in enumerate([(low_p, 'Input'), (base_p, 'Baseline'), (plus_p, 'FCE-Net+'), (gt_p, 'GT')]):
        ax = axes3[row_idx, c_idx]
        if os.path.exists(p):
            im = Image.open(p).convert('RGB')
            ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(cols3[c_idx], fontsize=12, fontweight='bold', pad=8)
    axes3[row_idx, 0].set_ylabel(f'LOL-v1 #{s}', fontsize=11, fontweight='bold', labelpad=8)

plt.tight_layout()
for d in FIG_DIRS:
    fig3.savefig(os.path.join(d, "fcenet_plus_lolv1_gallery.png"), bbox_inches='tight')
plt.close()
print("✅ Saved fcenet_plus_lolv1_gallery.png")

# -------------------------------------------------------------------------
# 4. Comprehensive LOL-v2 Real 4-Column Enhancement Gallery (FCE-Net+ Modified)
# -------------------------------------------------------------------------
print("Generating figures/fcenet_plus_lolv2_gallery.png...")
v2_scenes = ['low00690', 'low00700', 'low00720', 'low00750', 'low00780']
fig4, axes4 = plt.subplots(len(v2_scenes), 4, figsize=(16, 3.2 * len(v2_scenes)), dpi=200)
cols4 = ['(a) Raw Low-Light Input (Real Hardware)', '(b) Baseline FCE-Net (Springer 2026)', '(c) FCE-Net+ (Proposed Modified: Ours)', '(d) Normal Ground Truth Target']

for row_idx, s in enumerate(v2_scenes):
    norm_s = s.replace('low', 'normal')
    low_p = f"{BASE_DIR}/dataset_lolv2/LOL-V2-Real/Test/Low/{s}.png"
    base_p = f"{BASE_DIR}/FINAL_SUBMISSION/24BCE7695/02_Original_Implementation/Results/LOLv2/Output_Images/{s}_enhanced.png"
    plus_p = f"{BASE_DIR}/03_Modified_Implementation/Results/Enhanced_Images_LOLv2/enhanced_fcenet_plus_{s}.png"
    gt_p = f"{BASE_DIR}/dataset_lolv2/LOL-V2-Real/Test/High/{norm_s}.png"

    for c_idx, (p, name) in enumerate([(low_p, 'Input'), (base_p, 'Baseline'), (plus_p, 'FCE-Net+'), (gt_p, 'GT')]):
        ax = axes4[row_idx, c_idx]
        if os.path.exists(p):
            im = Image.open(p).convert('RGB')
            ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(cols4[c_idx], fontsize=12, fontweight='bold', pad=8)
    axes4[row_idx, 0].set_ylabel(f'LOL-v2 #{s[3:]}', fontsize=11, fontweight='bold', labelpad=8)

plt.tight_layout()
for d in FIG_DIRS:
    fig4.savefig(os.path.join(d, "fcenet_plus_lolv2_gallery.png"), bbox_inches='tight')
plt.close()
print("✅ Saved fcenet_plus_lolv2_gallery.png")
