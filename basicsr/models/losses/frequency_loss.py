import torch
import torch.nn as nn
from torch.nn import functional as F

class FrequencyLoss(nn.Module):
    """
    Frequency Domain Loss using 2D Real Fast Fourier Transform (rfft2).
    Computes loss between the spectral magnitudes of predicted and target images.
    Preserves high-frequency details (text strokes, sharp boundaries, and textures)
    which spatial L1/L2 pixel losses often over-smooth.

    Args:
        loss_weight (float): Loss weight for frequency loss. Default: 0.1.
        criterion (str): 'l1' (MAE) or 'l2' (MSE). Default: 'l1'.
    """
    def __init__(self, loss_weight=0.1, criterion='l1'):
        super(FrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = criterion
        if criterion == 'l1':
            self.loss_fn = F.l1_loss
        elif criterion == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise NotImplementedError(f"Criterion {criterion} is not supported in FrequencyLoss.")

    def forward(self, pred, target, **kwargs):
        # torch.fft.rfft2 computes the real-input 2D FFT, returning Hermitian-symmetric
        # half-spectrum which saves ~50% VRAM and computation time.
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        loss = self.loss_fn(pred_mag, target_mag, reduction='mean')
        return self.loss_weight * loss
