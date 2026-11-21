"""Loss functions and metrics.

SSIM + L1 mix follows Zhao et al., "Loss Functions for Image Restoration with
Neural Networks" (NVIDIA, 2017), which found 0.84 SSIM + 0.16 L1 works well.
"""

import math

import torch
import torch.nn.functional as F


def calculate_ssim(x_original, x_reconstructed, data_range=2.0, window_size=11, sigma=1.5):
    """Mean SSIM over the batch (higher is better, 1.0 = identical).

    data_range is max - min of the pixel values: 2.0 for images in [-1, 1].
    """
    # Gaussian window, built as the outer product of a 1D Gaussian with itself
    coords = torch.arange(window_size, dtype=x_original.dtype, device=x_original.device)
    coords = coords - (window_size - 1) / 2.0
    gaussian_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    gaussian_2d = gaussian_1d.unsqueeze(1) * gaussian_1d.unsqueeze(0)
    gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)

    num_channels = x_original.shape[1]
    gaussian_kernel = gaussian_2d.repeat(num_channels, 1, 1, 1)
    padding = window_size // 2

    # Stabilizing constants from the SSIM paper
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # Local means, variances, and covariance, computed per channel (groups=num_channels)
    mean_x = F.conv2d(x_original, gaussian_kernel, groups=num_channels, padding=padding)
    mean_y = F.conv2d(x_reconstructed, gaussian_kernel, groups=num_channels, padding=padding)
    mean_x_sq = F.conv2d(x_original * x_original, gaussian_kernel, groups=num_channels, padding=padding)
    mean_y_sq = F.conv2d(x_reconstructed * x_reconstructed, gaussian_kernel, groups=num_channels, padding=padding)
    mean_xy = F.conv2d(x_original * x_reconstructed, gaussian_kernel, groups=num_channels, padding=padding)

    variance_x = mean_x_sq - mean_x ** 2
    variance_y = mean_y_sq - mean_y ** 2
    covariance_xy = mean_xy - mean_x * mean_y

    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance_xy + c2)
    denominator = (mean_x ** 2 + mean_y ** 2 + c1) * (variance_x + variance_y + c2)
    ssim_map = numerator / denominator

    return ssim_map.mean()


def compute_mixed_loss(x_original, x_reconstructed, vq_loss, ssim_weight=0.84, l1_weight=0.16):
    """Stage 1 loss: 0.84 * (1 - SSIM) + 0.16 * L1 + VQ commitment loss."""
    ssim_score = calculate_ssim(x_original, x_reconstructed, data_range=2.0)
    ssim_loss = 1.0 - ssim_score

    l1_loss = F.l1_loss(x_reconstructed, x_original)

    reconstruction_loss = ssim_weight * ssim_loss + l1_weight * l1_loss
    total_loss = reconstruction_loss + vq_loss

    return {
        'total': total_loss,
        'reconstruction': reconstruction_loss,
        'ssim_score': ssim_score,
        'l1_loss': l1_loss,
    }


def images_to_pixel_bins(images):
    # [-1, 1] floats -> integer bins 0..255, the classes the PixelCNN decoder predicts
    pixels_0_to_1 = (images + 1.0) / 2.0
    pixel_bins = (pixels_0_to_1 * 255.0).round().long().clamp(0, 255)
    return pixel_bins


def compute_pixel_nll_loss(pixel_logits, target_images):
    """Cross-entropy over 256 intensity bins per subpixel.

    pixel_logits: (batch, channels, 256, height, width)
    target_images: (batch, channels, height, width) in [-1, 1]
    Returns the loss in nats and the same value in bits per dimension.
    """
    target_bins = images_to_pixel_bins(target_images)

    # F.cross_entropy expects the class dimension at position 1
    loss = F.cross_entropy(pixel_logits.permute(0, 2, 1, 3, 4), target_bins)

    bits_per_dim = loss / math.log(2)
    return loss, bits_per_dim
