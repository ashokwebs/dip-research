import os
import sys
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Ensure fallback search paths for authentic FCE model
base_search_paths = [
    current_dir,
    os.path.abspath(os.path.join(current_dir, "../../02_Original_Implementation/Code/Source")),
    os.path.abspath(os.path.join(current_dir, "../../../FINAL_SUBMISSION/24BCE7695/02_Original_Implementation/Code/Source")),
    os.path.abspath(os.path.join(current_dir, "../../../24BCE7695/02_Original_Implementation/Code/Source"))
]
for p in base_search_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from model import FCE
from transformer_blocks import EfficientTransformerBlock

class FCENetPlus(FCE):
    """
    FCE-Net+ (Proposed Modified Architecture with Transposed Transformer Block).
    Lead Researcher: Akhil Harshavardhan (24BCE7695).
    Replaces 8x downsampled spatial attention (MaxPool2d(8) + MHSA + Upsample(8))
    with native-resolution Efficient Transposed Transformer Block (MDTA + GDFN).
    Preserves 100% spatial resolution and fine boundary edge recovery.
    """
    def __init__(self, filters=32, num_heads=4):
        super().__init__(filters=filters)
        self.lum_transformer = EfficientTransformerBlock(dim=filters, num_heads=num_heads)
        self.gamma_trans = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs):
        ycbcr = self._rgb_to_ycbcr(inputs)
        y, cb, cr = torch.split(ycbcr, 1, dim=1)
        cb = self.denoiser_cb(cb) + cb
        cr = self.denoiser_cr(cr) + cr
        y_feat = self.process_y(y)
        y_processed = self.hfdm(y_feat)
        cb_processed = self.process_cb(cb)
        cr_processed = self.process_cr(cr)
        color = torch.cat([cb_processed, cr_processed], dim=1)
        ref = self.cagm(color)
        y_processed = self.edge_enhance(y_processed)
        lum = y_processed

        # Proposed ETB (MDTA + GDFN) directly at 100% native spatial resolution
        lum_trans = self.lum_transformer(lum)
        lum = lum + self.gamma_trans * lum_trans

        ref = self.ref_conv(ref)
        shortcut = ref
        ref = ref + 0.2 * self.lum_conv(lum)
        ref = self.msef(ref)
        ref = ref + shortcut
        recombined = self.recombine(torch.cat([ref, lum], dim=1))
        output = self.final_adjustments(recombined)

        return torch.sigmoid(output)
