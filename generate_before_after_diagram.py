#!/usr/bin/env python3
"""
Generate publication-grade Clean Architecture Diagrams for FCE-Net and FCE-Net+.
Student ID: 24BCE7695
Lead Researcher: Akhil Harshavardhan
=============================================================================
Matches the exact authentic visual style, layout, color palette, and iconography
of the original base paper architecture diagram (FCE-Net), stripping all decorative
badges, subtitles, warning boxes, and extraneous annotations.
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

# Master source: authentic high-resolution diagram
base_img_path = os.path.join(BASE_DIR, "04_Presentation/network_architecture.png")
if not os.path.exists(base_img_path):
    base_img_path = os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695/04_Presentation/network_architecture.png")

base_im = Image.open(base_img_path).convert("RGB")
w, h = base_im.size  # 2921 x 949

bg_color = (244, 177, 131)

def create_clean_modified_image():
    """Create the exact-type architecture diagram featuring proposed ETB (MDTA + GDFN) with ZERO useless stuff."""
    im = base_im.copy()
    draw = ImageDraw.Draw(im)

    # 1. Clean the space between vertical branch line (x=1870) and Add 2 (x=2088), y=80 to 340
    draw.rectangle([1865, 80, 2088, 340], fill=bg_color)

    # 2. Draw horizontal connecting line at y=242
    draw.line([(1870, 242), (2088, 242)], fill=(0, 0, 0), width=4)

    font_vert = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)

    y_box = 178
    h_box = 117
    w_box = 46

    # Sage/emerald green matching authentic diagram style
    c_box = (16, 185, 129)
    c_outline = (0, 0, 0)

    # 1. MDTA Box: x=1912..1958
    x_mdta = 1912
    draw.rounded_rectangle([x_mdta, y_box, x_mdta + w_box, y_box + h_box], radius=6, fill=c_box, outline=c_outline, width=2)
    # Arrow into MDTA
    draw.polygon([(x_mdta, 242), (x_mdta - 12, 236), (x_mdta - 12, 248)], fill=(0, 0, 0))

    chars_m = ['M', 'D', 'T', 'A']
    for i, ch in enumerate(chars_m):
        bb = font_vert.getbbox(ch)
        cw = bb[2] - bb[0]
        draw.text((x_mdta + (w_box - cw) / 2, y_box + 10 + i * 25), ch, font=font_vert, fill=(255, 255, 255))

    # 2. GDFN Box: x=2002..2048
    x_gdfn = 2002
    draw.rounded_rectangle([x_gdfn, y_box, x_gdfn + w_box, y_box + h_box], radius=6, fill=c_box, outline=c_outline, width=2)
    # Arrow into GDFN
    draw.polygon([(x_gdfn, 242), (x_gdfn - 12, 236), (x_gdfn - 12, 248)], fill=(0, 0, 0))

    chars_g = ['G', 'D', 'F', 'N']
    for i, ch in enumerate(chars_g):
        bb = font_vert.getbbox(ch)
        cw = bb[2] - bb[0]
        draw.text((x_gdfn + (w_box - cw) / 2, y_box + 10 + i * 25), ch, font=font_vert, fill=(255, 255, 255))

    # Arrow into Add 2
    draw.polygon([(2088, 242), (2076, 236), (2076, 248)], fill=(0, 0, 0))

    # No badges, no subtitles, no extra text. 100% clean authentic diagram.
    return im

def create_clean_baseline_image():
    """Pristine authentic baseline image with ZERO warning boxes or annotations."""
    return base_im.copy()

print("Generating 100% Clean Authentic Architecture Diagrams (No useless stuff)...")
im_clean_base = create_clean_baseline_image()
im_clean_mod = create_clean_modified_image()

# 1. Save single-panel modified diagram
single_mod_path = os.path.join(FIG_DIRS[0], "network_modified.png")
im_clean_mod.save(single_mod_path, dpi=(300, 300))

# 2. Save 2-Panel Clean Evolution Diagram (clean labels only, no warning cards or text paragraphs)
header_h = 45
gap = 25
total_h = header_h + h + gap + header_h + h + 20
combined = Image.new("RGB", (w + 40, total_h), (255, 255, 255))
c_draw = ImageDraw.Draw(combined)

font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)

# Label (a)
c_draw.text((25, 12), "(a) Baseline Architecture — Original FCE-Net (Guo et al., Springer 2026)", font=font_label, fill=(40, 40, 40))
combined.paste(im_clean_base, (20, header_h))

# Label (b)
y_b_head = header_h + h + gap
c_draw.text((25, y_b_head + 12), "(b) Proposed Architecture — Modified FCE-Net+ with Full-Resolution ETB (Ours)", font=font_label, fill=(6, 95, 70))
combined.paste(im_clean_mod, (20, y_b_head + header_h))

master_2panel_path = os.path.join(FIG_DIRS[0], "architecture_before_after.png")
combined.save(master_2panel_path, dpi=(300, 300))

# 3. Synchronize diagrams across all directories
for d in FIG_DIRS:
    for name in ["architecture_before_after.png", "modified_fce_net_architecture.png"]:
        dest = os.path.join(d, name)
        if dest != master_2panel_path:
            shutil.copyfile(master_2panel_path, dest)

    for name in ["network_modified.png", "network.png"]:
        dest = os.path.join(d, name)
        if dest != single_mod_path:
            shutil.copyfile(single_mod_path, dest)

# Also ensure 24BCE7695 subfolder has the identical clean diagrams
bce_dir = os.path.join(BASE_DIR, "24BCE7695")
if os.path.exists(bce_dir):
    shutil.copyfile("/home/charizard/24BCE7695/generate_before_after_diagram.py", os.path.join(bce_dir, "generate_before_after_diagram.py"))

print(f" Successfully generated and synchronized clean architecture diagrams across {len(FIG_DIRS)} directories!")
