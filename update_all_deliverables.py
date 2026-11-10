#!/usr/bin/env python3
"""
Automated Deliverables Updater & Packager for FCE-Net 24BCE7695
==============================================================
Synchronizes metrics from LOLv1, LOLv2, and Joint 500+500 experiments into:
1. LaTeX Research Paper (main.tex -> main.pdf -> Overleaf Zip)
2. Results Presentation (generate_results_presentation.py -> PPTX -> PDF)
3. Staging and Final Clean ZIP archive (24BCE7695.zip)
"""

import os
import sys
import json
import csv
import shutil
import subprocess
import glob
import zipfile

BASE_DIR = "/home/charizard/24BCE7695"
STAGE_DIR = os.path.join(BASE_DIR, "FINAL_SUBMISSION/24BCE7695")

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def update_paper():
    print("\n[1/4] Updating LaTeX Research Paper & Architectural Figures...")
    tex_path = os.path.join(STAGE_DIR, "01_Base_Paper/Research_Paper/main.tex")
    work_dir = os.path.dirname(tex_path)

    # 1. Generate Before vs After Architecture Diagram
    arch_gen_script = os.path.join(BASE_DIR, "generate_before_after_diagram.py")
    if os.path.exists(arch_gen_script):
        subprocess.run(["/home/charizard/.venvs/fce/bin/python", arch_gen_script], check=True)

    # 2. Ensure paper figures are generated and synced
    fig_gen_script = os.path.join(BASE_DIR, "generate_paper_figures.py")
    if os.path.exists(fig_gen_script):
        subprocess.run(["/home/charizard/.venvs/fce/bin/python", fig_gen_script], check=True)

    # Sync main.tex from root if newer
    root_tex = os.path.join(BASE_DIR, "01_Base_Paper/Research_Paper/main.tex")
    if os.path.exists(root_tex):
        shutil.copy2(root_tex, tex_path)
        shutil.copy2(root_tex, os.path.join(BASE_DIR, "24BCE7695/01_Base_Paper/Research_Paper/main.tex"))

    # Compile LaTeX
    print("Compiling LaTeX to PDF...")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=work_dir, stdout=subprocess.DEVNULL)
    subprocess.run(["bibtex", "main"], cwd=work_dir, stdout=subprocess.DEVNULL)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=work_dir, stdout=subprocess.DEVNULL)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=work_dir, stdout=subprocess.DEVNULL)

    pdf_path = os.path.join(work_dir, "main.pdf")
    if os.path.exists(pdf_path):
        shutil.copy2(pdf_path, os.path.join(work_dir, "Student_Research_Report.pdf"))
        doc_dir = os.path.join(STAGE_DIR, "02_Original_Implementation/Code/Documentation")
        os.makedirs(doc_dir, exist_ok=True)
        shutil.copy2(pdf_path, os.path.join(doc_dir, "Student_Research_Report.pdf"))
        # Sync to root 01_Base_Paper, root, 24BCE7695, and FOR_MACBOOK_KAGGLE
        root_paper_dir = os.path.join(BASE_DIR, "01_Base_Paper/Research_Paper")
        os.makedirs(root_paper_dir, exist_ok=True)
        shutil.copy2(pdf_path, os.path.join(root_paper_dir, "Student_Research_Report.pdf"))
        shutil.copy2(pdf_path, os.path.join(root_paper_dir, "main.pdf"))
        shutil.copy2(pdf_path, os.path.join(BASE_DIR, "Student_Research_Report.pdf"))
        shutil.copy2(pdf_path, os.path.join(BASE_DIR, "FOR_MACBOOK_KAGGLE/Student_Research_Report.pdf"))
        # Sync to 24BCE7695 subfolder
        bce_paper_dir = os.path.join(BASE_DIR, "24BCE7695/01_Base_Paper/Research_Paper")
        os.makedirs(bce_paper_dir, exist_ok=True)
        shutil.copy2(pdf_path, os.path.join(bce_paper_dir, "Student_Research_Report.pdf"))
        shutil.copy2(pdf_path, os.path.join(bce_paper_dir, "main.pdf"))
        bce_doc_dir = os.path.join(BASE_DIR, "24BCE7695/02_Original_Implementation/Code/Documentation")
        os.makedirs(bce_doc_dir, exist_ok=True)
        shutil.copy2(pdf_path, os.path.join(bce_doc_dir, "Student_Research_Report.pdf"))
        print(f" Recompiled LaTeX paper: {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)")

    # Clean aux files in Research_Paper
    for ext in [".aux", ".log", ".out", ".toc", ".blg"]:
        for f in glob.glob(os.path.join(work_dir, f"*{ext}")):
            try:
                os.remove(f)
            except OSError:
                pass

    # Re-package clean Research_Paper_Overleaf.zip (source only: .tex, .bib, .bbl, figures/)
    overleaf_zip = os.path.join(STAGE_DIR, "01_Base_Paper/Research_Paper_Overleaf.zip")
    if os.path.exists(overleaf_zip):
        os.remove(overleaf_zip)
    subprocess.run(["zip", "-r", "-q", overleaf_zip, "main.tex", "references.bib", "main.bbl", "figures/"], cwd=work_dir, check=True)
    shutil.copy2(overleaf_zip, os.path.join(BASE_DIR, "01_Base_Paper/Research_Paper_Overleaf.zip"))
    shutil.copy2(overleaf_zip, os.path.join(BASE_DIR, "FOR_MACBOOK_KAGGLE/Research_Paper_Overleaf.zip"))
    shutil.copy2(overleaf_zip, os.path.join(BASE_DIR, "24BCE7695/01_Base_Paper/Research_Paper_Overleaf.zip"))
    print(f" Generated clean Overleaf source archive: {overleaf_zip} ({os.path.getsize(overleaf_zip)/(1024*1024):.1f} MB)")

def update_presentation():
    print("\n[2/4] Generating Academic Slide Deck (PPTX & PDF)...")
    pres_script = os.path.join(BASE_DIR, "generate_results_presentation.py")
    subprocess.run(["/home/charizard/.venvs/fce/bin/python", pres_script], check=True)

    # Convert to PDF via LibreOffice
    pres_dir = os.path.join(STAGE_DIR, "04_Presentation")
    pptx_src = os.path.join(pres_dir, "Results_Presentation.pptx")
    subprocess.run(["libreoffice", "--headless", "--convert-to", "pdf", "Results_Presentation.pptx"], cwd=pres_dir, check=True)
    pdf_src = os.path.join(pres_dir, "Results_Presentation.pdf")
    
    for target_dir in [
        os.path.join(BASE_DIR, "04_Presentation"),
        os.path.join(BASE_DIR, "24BCE7695/04_Presentation"),
        BASE_DIR,
        os.path.join(BASE_DIR, "FOR_MACBOOK_KAGGLE")
    ]:
        os.makedirs(target_dir, exist_ok=True)
        if target_dir == BASE_DIR:
            shutil.copy2(pptx_src, os.path.join(target_dir, "Results_Presentation.pptx"))
            shutil.copy2(pdf_src, os.path.join(target_dir, "Results_Presentation.pdf"))
        elif "FOR_MACBOOK_KAGGLE" in target_dir:
            shutil.copy2(pdf_src, os.path.join(target_dir, "Results_Presentation.pdf"))
            shutil.copy2(pptx_src, os.path.join(target_dir, "Results_Presentation.pptx"))
        else:
            shutil.copy2(pptx_src, os.path.join(target_dir, "Results_Presentation.pptx"))
            shutil.copy2(pdf_src, os.path.join(target_dir, "Results_Presentation.pdf"))

    print(" Converted Results_Presentation.pptx to Results_Presentation.pdf and synced across all targets.")

def clean_and_repackage():
    print("\n[3/4] Purging Caches and Temporary Files...")
    # Clean __pycache__ and aux files
    for root, dirs, files in os.walk(STAGE_DIR):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        for f in files:
            if f.endswith(('.aux', '.log', '.out', '.blg')) or f == '.DS_Store':
                try:
                    os.remove(os.path.join(root, f))
                except OSError:
                    pass

    print("\n[4/4] Packaging Clean Final Submission Archive (24BCE7695.zip)...")
    zip_target = os.path.join(BASE_DIR, "24BCE7695.zip")
    if os.path.exists(zip_target):
        os.remove(zip_target)

    # Zip with 24BCE7695 as top-level folder
    subprocess.run(["zip", "-r", "-q", zip_target, "24BCE7695"], cwd=os.path.dirname(STAGE_DIR), check=True)
    zip_size_mb = os.path.getsize(zip_target) / (1024 * 1024)
    print(f" Generated final archive: {zip_target} ({zip_size_mb:.1f} MB)")

    # Audit archive
    with zipfile.ZipFile(zip_target, 'r') as z:
        names = z.namelist()
        bad_roots = [n for n in names if not n.startswith('24BCE7695/')]
        caches = [n for n in names if '__pycache__' in n or '.DS_Store' in n or n.endswith(('.aux', '.log', '.out'))]
        top_dirs = set(n.split('/')[1] for n in names if '/' in n and n.split('/')[1])
        print(f"  Archive Audit: {len(names)} total entries | Bad roots: {len(bad_roots)} | Caches: {len(caches)}")
        print(f"  Subdirectories under 24BCE7695/: {sorted(list(top_dirs))}")
        assert len(bad_roots) == 0, "Error: Found entries not under 24BCE7695/"
        assert len(caches) == 0, "Error: Found cache files in zip"

    # Repackage Kaggle_Transfer_Pack.zip (FOR_MACBOOK_KAGGLE folder)
    ktp = os.path.join(BASE_DIR, "Kaggle_Transfer_Pack.zip")
    if os.path.exists(ktp):
        os.remove(ktp)
    subprocess.run(["zip", "-r", "-q", ktp, "FOR_MACBOOK_KAGGLE"], cwd=BASE_DIR, check=True)
    print(f" Generated MacBook/Kaggle transfer pack: {ktp} ({os.path.getsize(ktp)/(1024*1024):.1f} MB)")

    print(" All deliverables synchronized, verified, and packaged successfully!")

def main():
    update_paper()
    update_presentation()
    clean_and_repackage()

if __name__ == "__main__":
    main()
