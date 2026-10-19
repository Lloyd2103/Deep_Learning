import torch
import torch.nn as nn
import torch.nn.functional as F

class HighFreqBranch(nn.Module):
    """
    Lightweight CNN branch to extract and reconstruct high-frequency text and edge details.
    Uses high-pass filtering (x - low_pass(x)) to isolate edge/stroke features
    from large-area shadow variations.
    """
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(HighFreqBranch, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Depthwise + Pointwise conv for efficiency and low parameter count
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        # High-pass filter: x_high = x - low_pass(x)
        x_low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_high = x - x_low
        return self.net(x_high)


class AdaptiveGatedFusion(nn.Module):
    """
    Adaptive Gated Spatial-Frequency Fusion (GSFF).
    Dynamically estimates a pixel-wise spatial gate map in [0, 1] conditioned on
    both global restored features (f_main) and high-frequency edge details (f_hf).

    Formula:
        Gate = Sigmoid(Conv1x1(LeakyReLU(Conv3x3([f_main, f_hf]))))
        Output = f_main + Gate * f_hf

    - Where text strokes/sharp edges are blurred, Gate -> 1 to inject crisp details.
    - Where background/clean illumination is already smooth, Gate -> 0 to avoid noise.
    """
    def __init__(self, in_channels=3, mid_channels=16):
        super(AdaptiveGatedFusion, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, f_main, f_hf):
        combined = torch.cat([f_main, f_hf], dim=1)
        gate = self.gate_net(combined)
        return f_main + gate * f_hf
