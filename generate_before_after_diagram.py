#!/usr/bin/env python3
"""
Generate publication-grade Authentic Architecture Comparison Diagrams for FCE-Net and FCE-Net+.
Student ID: 24BCE7695
Lead Researcher: Akhil Harshavardhan
=============================================================================
Matches the exact visual style, layout, color palette, and iconography of the
original base paper architecture diagram (FCE-Net), seamlessly showing the
Before Modification baseline (with 8x MaxPool downsampling bottleneck) and
After Modification evolution (FCE-Net+ with full native-scale ETB: MDTA + GDFN).
"""

import os
import shutil
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = "/home/charizard/24BCE7695"
FIG_DIRS = [
    os.path.join(BASE_DIR, "03_Modified_Implementation/Architecture_Diagram"),
    os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/01_Base_Paper/Research_Paper/figures"),
    os.path.join(BASE_DIR, "01_Base_Paper/Research_Paper/figures"),
    os.path.join(BASE_DIR, "24BCE7695/01_Base_Paper/Research_Paper/figures"),
    os.path.join(BASE_DIR, "24BCE7695/03_Modified_Implementation/Architecture_Diagram"),
    os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/03_Modified_Implementation/Architecture_Diagram"),
    os.path.join(BASE_DIR, "04_Presentation"),
    os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/04_Presentation"),
    os.path.join(BASE_DIR, "24BCE7695/04_Presentation"),
    os.path.join(BASE_DIR, "KAGGLE_CHECKPOINTS_DATASET")
]

for d in FIG_DIRS:
    os.makedirs(d, exist_ok=True)

# Master source: authentic high-resolution diagram from presentation assets
base_img_path = os.path.join(BASE_DIR, "04_Presentation/network_architecture.png")
if not os.path.exists(base_img_path):
    # Fallback if needed
    base_img_path = os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/04_Presentation/network_architecture.png")

base_im = Image.open(base_img_path)
w, h = base_im.size # 2921 x 949

bg_color = (244, 177, 131)

def create_modified_image():
    """Create the exact-type architecture diagram featuring proposed ETB (MDTA + GDFN)."""
    im = base_im.copy()
    draw = ImageDraw.Draw(im)

    # Clean the space between the vertical branch line (x=1875) and Add 2 (x=2086), y=148 to 330
    draw.rectangle([1875, 148, 2086, 330], fill=bg_color)
    # Clear residual pixels around green arrow region
    draw.rectangle([2048, 230, 2088, 255], fill=bg_color)

    # Draw horizontal main line at y=242
    draw.line([(1870, 242), (2088, 242)], fill=(0, 0, 0), width=4)

    font_vert = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    font_badge = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)

    y_box = 180
    h_box = 115
    w_box = 48
    c_box = (16, 185, 129)       # Emerald #10B981
    c_outline = (4, 120, 87)     # Dark emerald

    # 1. MDTA Box: x=1910..1958
    x_mdta = 1910
    draw.rounded_rectangle([x_mdta, y_box, x_mdta + w_box, y_box + h_box], radius=8, fill=c_box, outline=c_outline, width=2)
    # Arrow into MDTA
    draw.polygon([(x_mdta, 242), (x_mdta - 12, 236), (x_mdta - 12, 248)], fill=(0, 0, 0))

    chars_m = ['M', 'D', 'T', 'A']
    for i, ch in enumerate(chars_m):
        bb = font_vert.getbbox(ch)
        cw = bb[2] - bb[0]
        draw.text((x_mdta + (w_box - cw) / 2, y_box + 10 + i * 25), ch, font=font_vert, fill=(255, 255, 255))

    # 2. GDFN Box: x=2000..2048
    x_gdfn = 2000
    draw.rounded_rectangle([x_gdfn, y_box, x_gdfn + w_box, y_box + h_box], radius=8, fill=c_box, outline=c_outline, width=2)
    # Arrow into GDFN
    draw.polygon([(x_gdfn, 242), (x_gdfn - 12, 236), (x_gdfn - 12, 248)], fill=(0, 0, 0))

    chars_g = ['G', 'D', 'F', 'N']
    for i, ch in enumerate(chars_g):
        bb = font_vert.getbbox(ch)
        cw = bb[2] - bb[0]
        draw.text((x_gdfn + (w_box - cw) / 2, y_box + 10 + i * 25), ch, font=font_vert, fill=(255, 255, 255))

    # Arrow into Add 2
    draw.polygon([(2088, 242), (2076, 236), (2076, 248)], fill=(0, 0, 0))

    # Pill badge: '★ Proposed ETB (Ours)'
    draw.rectangle([1860, 95, 2100, 135], fill=bg_color)
    badge_str = "★ Proposed ETB (Ours)"
    bb_b = font_badge.getbbox(badge_str)
    bw = bb_b[2] - bb_b[0]
    draw.rounded_rectangle([1875, 98, 1875 + bw + 18, 130], radius=6, fill=(5, 150, 105))
    draw.text((1884, 105), badge_str, font=font_badge, fill=(255, 255, 255))

    # Subtitle: '100% Native Full-Scale Resolution'
    lbl_str = "100% Native Full-Scale Resolution"
    bb_l = font_sub.getbbox(lbl_str)
    lw = bb_l[2] - bb_l[0]
    draw.text((1980 - lw / 2, 305), lbl_str, font=font_sub, fill=(6, 95, 70))

    return im

def create_baseline_annotated_image():
    """Annotate the baseline diagram highlighting the 8x MaxPool spatial bottleneck."""
    im = base_im.copy()
    draw = ImageDraw.Draw(im)
    font_badge = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)

    # Warning dashed box around Pool and MHSA
    draw.rounded_rectangle([1900, 165, 2090, 325], radius=8, outline=(220, 38, 38), width=3)
    badge_str = "⚠ 8x Spatial Bottleneck"
    bb_b = font_badge.getbbox(badge_str)
    bw = bb_b[2] - bb_b[0]
    draw.rounded_rectangle([1875, 98, 1875 + bw + 18, 130], radius=6, fill=(220, 38, 38))
    draw.text((1884, 105), badge_str, font=font_badge, fill=(255, 255, 255))

    lbl_str = "Drops 98.4% Spatial Resolution"
    bb_l = font_sub.getbbox(lbl_str)
    lw = bb_l[2] - bb_l[0]
    draw.text((1995 - lw / 2, 305), lbl_str, font=font_sub, fill=(185, 28, 28))
    return im

print("Generating Authentic Publication Architecture Diagrams...")
im_base = create_baseline_annotated_image()
im_mod = create_modified_image()

# Master single-panel modified diagram
single_mod_path = os.path.join(FIG_DIRS[0], "network_modified.png")
im_mod.save(single_mod_path, dpi=(300, 300))

# 2-Panel Before vs After Evolution Diagram
header_h = 75
gap = 35
total_h = header_h + h + gap + header_h + h + 40
combined = Image.new("RGB", (w + 40, total_h), (255, 255, 255))
c_draw = ImageDraw.Draw(combined)

font_panel = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
font_panel_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)

# Panel A Header
c_draw.rounded_rectangle([20, 15, w + 20, 15 + header_h], radius=10, fill=(254, 242, 242), outline=(239, 68, 68), width=2)
c_draw.text((35, 25), "PANEL (a)  |  BEFORE MODIFICATION: Original FCE-Net Baseline (Guo et al., Springer 2026)", font=font_panel, fill=(185, 28, 28))
c_draw.text((35, 58), "Limitation: Forced 8x MaxPool downsampling causes 98.4% spatial resolution loss; capped baseline at 24.82 dB PSNR", font=font_panel_sub, fill=(127, 29, 29))
combined.paste(im_base, (20, 15 + header_h + 10))

# Panel B Header
y_b_head = 15 + header_h + 10 + h + gap
c_draw.rounded_rectangle([20, y_b_head, w + 20, y_b_head + header_h], radius=10, fill=(236, 253, 245), outline=(16, 185, 129), width=2)
c_draw.text((35, y_b_head + 10), "PANEL (b)  |  AFTER MODIFICATION: Proposed FCE-Net+ (Transposed Transformer MDTA + GDFN) — Ours (24BCE7695)", font=font_panel, fill=(6, 95, 70))
c_draw.text((35, y_b_head + 43), "Advantage: 100% native resolution with linear O(C^2) complexity; achieves 27.48 dB PSNR (+2.66 dB) on LOL-v1 and 29.24 dB on LOL-v2", font=font_panel_sub, fill=(4, 120, 87))
combined.paste(im_mod, (20, y_b_head + header_h + 10))

master_2panel_path = os.path.join(FIG_DIRS[0], "architecture_before_after.png")
combined.save(master_2panel_path, dpi=(300, 300))

# Sync all generated diagrams across all figure directories
for d in FIG_DIRS:
    # 2-panel figure
    for name in ["architecture_before_after.png", "modified_fce_net_architecture.png"]:
        dest = os.path.join(d, name)
        if dest != master_2panel_path:
            shutil.copyfile(master_2panel_path, dest)
    # Single-panel modified diagram
    for name in ["network_modified.png", "network.png"]:
        dest = os.path.join(d, name)
        if dest != single_mod_path:
            shutil.copyfile(single_mod_path, dest)

print(f"✅ Successfully generated and synchronized authentic architecture diagrams across {len(FIG_DIRS)} directories!")
