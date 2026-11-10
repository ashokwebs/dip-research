#!/usr/bin/env python3
"""
FCE-Net+ Training Pipeline (300 Epochs Full Convergence)
======================================================
Author: Akhil Harshavardhan (Candidate Reg. No.: 24BCE7695)
Department of Computer Science & Engineering

Model: FCE-Net+ with Efficient Transposed Transformer Block (MDTA + GDFN)
Dataset: LOL-v1 Canon (485 train, 15 test pairs) and LOL-v2 Real (689 train, 100 test pairs)
Schedule: 300 epochs, Cosine Annealing, AdamW optimizer
"""

import os
import sys
import argparse
import time
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_modified import FCENetPlus
from losses import ModifiedCompositeLoss
from dataloader import create_dataloaders
from metric import calculate_psnr, calculate_ssim

def parse_args():
    parser = argparse.ArgumentParser(description="Train FCE-Net+ with Transposed Transformer Block")
    parser.add_argument("--epochs", type=int, default=300, help="Total training epochs (default: 300)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--crop_size", type=int, default=256, help="Random crop size for training")
    parser.add_argument("--train_low", type=str, default="/home/charizard/24BCE7695/dataset_lolv1/lol_dataset/our485/low")
    parser.add_argument("--train_high", type=str, default="/home/charizard/24BCE7695/dataset_lolv1/lol_dataset/our485/high")
    parser.add_argument("--test_low", type=str, default="/home/charizard/24BCE7695/dataset_lolv1/lol_dataset/eval15/low")
    parser.add_argument("--test_high", type=str, default="/home/charizard/24BCE7695/dataset_lolv1/lol_dataset/eval15/high")
    parser.add_argument("--save_dir", type=str, default="/home/charizard/24BCE7695/03_Modified_Implementation/Checkpoints")
    parser.add_argument("--log_dir", type=str, default="/home/charizard/24BCE7695/03_Modified_Implementation/Log_Files")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting FCE-Net+ 300-Epoch Training Pipeline ===")
    print(f"Device: {device} | Target Epochs: {args.epochs}")
    print(f"Researcher: Akhil Harshavardhan (24BCE7695)")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Initialize model
    model = FCENetPlus(filters=32, num_heads=4).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,} ({params/1e6:.3f}M)")

    # Dataloaders
    train_loader, test_loader = create_dataloaders(
        args.train_low, args.train_high, args.test_low, args.test_high,
        crop_size=args.crop_size, batch_size=args.batch_size
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = ModifiedCompositeLoss(device)

    best_psnr = 0.0
    best_ssim = 0.0
    log_csv_path = os.path.join(args.log_dir, "training_log_300epochs.csv")

    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_psnr", "val_ssim", "lr", "best_psnr"])

    print(f"Initialized training log: {log_csv_path}")

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.time()

        for step, (low, high) in enumerate(train_loader):
            low, high = low.to(device), high.to(device)
            optimizer.zero_grad()
            pred = model(low)
            loss = criterion(high, pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)

        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            val_psnr, val_ssim = 0.0, 0.0
            with torch.no_grad():
                for low, high in test_loader:
                    low, high = low.to(device), high.to(device)
                    pred = model(low)
                    val_psnr += calculate_psnr(pred, high, gt_mean=False)
                    val_ssim += calculate_ssim(pred, high, gt_mean=False)
            val_psnr /= max(len(test_loader), 1)
            val_ssim /= max(len(test_loader), 1)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_ssim = val_ssim
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psnr': best_psnr,
                    'ssim': best_ssim,
                    'author': 'Akhil Harshavardhan (24BCE7695)'
                }, os.path.join(args.save_dir, "fcenet_plus_lolv1_best.pth"))

            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch:3d}/{args.epochs}] | Loss: {avg_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | Best PSNR: {best_psnr:.2f} dB | LR: {current_lr:.2e}")

            with open(log_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{avg_loss:.6f}", f"{val_psnr:.4f}", f"{val_ssim:.4f}", f"{current_lr:.6e}", f"{best_psnr:.4f}"])

    print("=== Training Successfully Completed 300 Epochs ===")

if __name__ == '__main__':
    main()
