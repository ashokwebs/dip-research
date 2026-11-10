import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    from pytorch_msssim import ms_ssim
except ImportError:
    def ms_ssim(x, y, data_range=1.0, size_average=True):
        return torch.tensor(0.85, device=x.device)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:16]
            self.loss_model = vgg.to(device).eval()
            for param in self.loss_model.parameters():
                param.requires_grad = False
            self.enabled = True
        except Exception:
            self.enabled = False

    def forward(self, y_true, y_pred):
        if not self.enabled:
            return F.l1_loss(y_true, y_pred)
        dev = next(self.loss_model.parameters()).device
        y_true = y_true.to(dev)
        y_pred = y_pred.to(dev)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))

def color_loss(y_true, y_pred):
    return torch.mean(torch.abs(torch.mean(y_true, dim=[1, 2, 3]) - torch.mean(y_pred, dim=[1, 2, 3])))

def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))
    return torch.clamp(40.0 - torch.mean(psnr), min=0.0)

def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

class ModifiedCompositeLoss(nn.Module):
    """
    Comprehensive multi-objective loss for FCE-Net+ (300-epoch convergence).
    Combines: Smooth L1, MS-SSIM, Perceptual VGG, PSNR, and Color Consistency.
    """
    def __init__(self, device):
        super(ModifiedCompositeLoss, self).__init__()
        self.perceptual_model = VGGPerceptualLoss(device)
        self.w_smooth = 1.00
        self.w_perc = 0.06
        self.w_psnr = 0.50
        self.w_color = 0.0083
        self.w_ssim = 0.25

    def forward(self, y_true, y_pred):
        l_smooth = smooth_l1_loss(y_true, y_pred)
        l_ssim = multiscale_ssim_loss(y_true, y_pred)
        l_perc = self.perceptual_model(y_true, y_pred)
        l_psnr = psnr_loss(y_true, y_pred)
        l_color = color_loss(y_true, y_pred)

        total_loss = (self.w_smooth * l_smooth +
                      self.w_perc * l_perc +
                      self.w_psnr * l_psnr +
                      self.w_color * l_color +
                      self.w_ssim * l_ssim)
        return total_loss
