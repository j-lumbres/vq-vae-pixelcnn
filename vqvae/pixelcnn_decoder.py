"""Main model: a VQ-VAE whose decoder is a conditional Gated PixelCNN.

Follows Section 3.4 of van den Oord et al., "Conditional Image Generation with
PixelCNN Decoders" (2016): the encoder's latent conditions a PixelCNN that
models p(x | z) one subpixel at a time, in raster order and R -> G -> B
within a pixel.

Known failure mode: a strong autoregressive decoder can predict pixels from
their neighbours alone and stop using z (posterior collapse). The training
script counters this by warm-starting the encoder from Stage 1 and tracks it
with latent_probe().
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vqvae.vqvae import Encoder, VectorQuantizerEMA
from vqvae.losses import compute_pixel_nll_loss


class VerticalStackConv(nn.Module):
    """Sees only the rows strictly above the current row."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.horizontal_padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x):
        # Pad kernel_size rows on top: output row i then covers input rows i-k .. i-1,
        # plus one extra row at the bottom that we crop off
        x_padded = F.pad(x, (self.horizontal_padding, self.horizontal_padding, self.kernel_size, 0))
        vertical_features = self.conv(x_padded)
        return vertical_features[:, :, :-1, :]


def build_color_mask(mask, in_channels, out_channels, mask_type, center_slice):
    """Channel mask at the current pixel so R -> G -> B stays causal.

    Channels are split into 3 equal groups (R, G, B).
    Type 'A' (first layer, raw pixels): R sees nothing, G sees R, B sees R and G.
    Type 'B' (later layers): each group may also see its own group's features.
    """
    in_group = in_channels // 3
    out_group = out_channels // 3

    for out_color in range(3):
        for in_color in range(3):
            out_slice = slice(out_color * out_group, (out_color + 1) * out_group)
            in_slice = slice(in_color * in_group, (in_color + 1) * in_group)

            if mask_type == 'A' and in_color < out_color:
                mask[out_slice, in_slice, :, center_slice] = 1
            if mask_type == 'B' and in_color <= out_color:
                mask[out_slice, in_slice, :, center_slice] = 1
    return mask


class HorizontalStackConv(nn.Module):
    """1 x k convolution over the current row: pixels to the left, plus the
    current pixel's earlier colors (and own color for mask 'B')."""

    def __init__(self, in_channels, out_channels, kernel_size, mask_type):
        super().__init__()
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        mask = torch.zeros(out_channels, in_channels, 1, kernel_size)
        center = kernel_size // 2

        # Every pixel strictly to the left is visible in full
        mask[:, :, :, :center] = 1
        mask = build_color_mask(mask, in_channels, out_channels, mask_type, center_slice=center)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=(0, self.padding))


class MaskedConv1x1(nn.Conv2d):
    """1x1 conv that mixes channels at the current pixel without breaking R -> G -> B order."""

    def __init__(self, in_channels, out_channels, mask_type):
        super().__init__(in_channels, out_channels, kernel_size=1)
        mask = torch.zeros_like(self.weight)
        mask = build_color_mask(mask, in_channels, out_channels, mask_type, center_slice=slice(None))
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias)


def split_filter_and_gate_by_color_group(features, num_hiddens):
    # The horizontal conv output is laid out as [R: filter, gate | G: filter, gate | B: filter, gate],
    # so filter and gate must be split inside each color group, not down the middle
    batch_size, channels, height, width = features.shape
    group_size = num_hiddens // 3
    features_by_group = features.view(batch_size, 3, 2, group_size, height, width)
    filter_part = features_by_group[:, :, 0].reshape(batch_size, num_hiddens, height, width)
    gate_part = features_by_group[:, :, 1].reshape(batch_size, num_hiddens, height, width)
    return filter_part, gate_part


class GatedPixelCNNLayer(nn.Module):
    """One gated layer: tanh(filter) * sigmoid(gate), with a vertical and a horizontal stack.

    The vertical stack feeds the horizontal stack, which removes the blind spot
    of the original PixelCNN (Section 2.2 of the paper).
    """

    def __init__(self, in_channels, num_hiddens, kernel_size, mask_type, conditioning_channels):
        super().__init__()
        self.mask_type = mask_type
        self.num_hiddens = num_hiddens

        self.vertical_conv = VerticalStackConv(in_channels, 2 * num_hiddens, kernel_size)
        self.horizontal_conv = HorizontalStackConv(in_channels, 2 * num_hiddens, kernel_size, mask_type)
        self.vertical_to_horizontal = nn.Conv2d(2 * num_hiddens, 2 * num_hiddens, kernel_size=1)

        # The latent condition enters both stacks (Equation 5 in the paper)
        self.vertical_condition_proj = nn.Conv2d(conditioning_channels, 2 * num_hiddens, kernel_size=1)
        self.horizontal_condition_proj = nn.Conv2d(conditioning_channels, 2 * num_hiddens, kernel_size=1)

        self.horizontal_output_conv = MaskedConv1x1(num_hiddens, num_hiddens, mask_type='B')

    def forward(self, vertical_input, horizontal_input, spatial_condition):
        vertical_features = self.vertical_conv(vertical_input)
        vertical_features = vertical_features + self.vertical_condition_proj(spatial_condition)
        vertical_filter, vertical_gate = vertical_features.chunk(2, dim=1)
        vertical_output = torch.tanh(vertical_filter) * torch.sigmoid(vertical_gate)

        horizontal_features = self.horizontal_conv(horizontal_input)
        horizontal_features = horizontal_features + self.vertical_to_horizontal(vertical_features)
        horizontal_features = horizontal_features + self.horizontal_condition_proj(spatial_condition)
        horizontal_filter, horizontal_gate = split_filter_and_gate_by_color_group(horizontal_features, self.num_hiddens)
        horizontal_output = torch.tanh(horizontal_filter) * torch.sigmoid(horizontal_gate)
        horizontal_output = self.horizontal_output_conv(horizontal_output)

        # Residual connection only after the first layer (shapes differ before that)
        if self.mask_type == 'B':
            horizontal_output = horizontal_output + horizontal_input

        return vertical_output, horizontal_output


class ConditionalPixelCNNDecoder(nn.Module):
    def __init__(self, latent_channels, image_channels=3, num_hiddens=120, num_layers=6, pixel_levels=256):
        super().__init__()
        self.image_channels = image_channels
        self.pixel_levels = pixel_levels

        # Upsample the 8x8 latent to 32x32 so every pixel gets its own conditioning vector
        self.condition_upsampler = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.first_layer = GatedPixelCNNLayer(
            in_channels=image_channels, num_hiddens=num_hiddens, kernel_size=7,
            mask_type='A', conditioning_channels=num_hiddens,
        )

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = GatedPixelCNNLayer(
                in_channels=num_hiddens, num_hiddens=num_hiddens, kernel_size=3,
                mask_type='B', conditioning_channels=num_hiddens,
            )
            self.hidden_layers.append(layer)

        self.output_conv1 = MaskedConv1x1(num_hiddens, num_hiddens, mask_type='B')
        self.output_conv2 = MaskedConv1x1(num_hiddens, image_channels * pixel_levels, mask_type='B')

    def forward(self, target_image, z_quantized):
        spatial_condition = self.condition_upsampler(z_quantized)

        # Teacher forcing: during training the real image is the autoregressive input
        vertical, horizontal = self.first_layer(target_image, target_image, spatial_condition)
        for layer in self.hidden_layers:
            vertical, horizontal = layer(vertical, horizontal, spatial_condition)

        features = F.relu(horizontal)
        features = F.relu(self.output_conv1(features))
        logits = self.output_conv2(features)

        # Output channels are grouped by color, so this view gives (batch, color, 256, H, W)
        batch_size, _, height, width = logits.shape
        logits = logits.view(batch_size, self.image_channels, self.pixel_levels, height, width)
        return logits

    @torch.no_grad()
    def sample(self, z_quantized, temperature=1.0):
        """Generates images one subpixel at a time: height * width * 3 forward passes."""
        batch_size = z_quantized.shape[0]

        # The upsampler doubles the latent twice, so an 8x8 latent always means a 32x32 image
        height = z_quantized.shape[2] * 4
        width = z_quantized.shape[3] * 4
        generated_image = torch.zeros(batch_size, self.image_channels, height, width, device=z_quantized.device)

        for row in range(height):
            for col in range(width):
                for channel in range(self.image_channels):
                    # The forward pass sits inside the channel loop because G must see the sampled R
                    pixel_logits = self.forward(generated_image, z_quantized)
                    channel_logits = pixel_logits[:, channel, :, row, col] / temperature
                    pixel_distribution = F.softmax(channel_logits, dim=1)

                    sampled_bin = torch.multinomial(pixel_distribution, num_samples=1).squeeze(1)
                    generated_image[:, channel, row, col] = (sampled_bin.float() / 255.0) * 2.0 - 1.0

        return generated_image


class VQ_VAE_PixelCNN(nn.Module):
    """Encoder and quantizer from Stage 1, PixelCNN in place of the deconv decoder."""

    def __init__(self, num_hiddens, num_embeddings, embedding_dim, commitment_cost, decay,
                 pixelcnn_hiddens=120, pixelcnn_layers=6):
        super().__init__()
        self.encoder = Encoder(num_hiddens)
        self.pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1)
        self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        self.pixelcnn_decoder = ConditionalPixelCNNDecoder(
            latent_channels=embedding_dim,
            num_hiddens=pixelcnn_hiddens,
            num_layers=pixelcnn_layers,
        )

    def encode(self, x):
        z_encoded = self.encoder(x)
        z_encoded = self.pre_vq_conv(z_encoded)
        vq_loss, z_quantized, perplexity = self.quantizer(z_encoded)
        return vq_loss, z_quantized, perplexity

    def forward(self, x):
        vq_loss, z_quantized, perplexity = self.encode(x)
        pixel_logits = self.pixelcnn_decoder(x, z_quantized)
        return pixel_logits, vq_loss, perplexity

    def set_encoder_frozen(self, frozen):
        # Freezing covers the encoder, the projection, and the codebook's EMA updates.
        # The quantizer only updates its codebook in train mode, so eval() stops that.
        for module in [self.encoder, self.pre_vq_conv]:
            for parameter in module.parameters():
                parameter.requires_grad = not frozen
        if frozen:
            self.encoder.eval()
            self.quantizer.eval()
        else:
            self.encoder.train()
            self.quantizer.train()


@torch.no_grad()
def latent_probe(model, images):
    """Bits/dim with the real latent vs. a zeroed latent, plus codebook perplexity.

    gap = blind - real. A gap near 0 means the decoder ignores z (posterior collapse).
    """
    _, z_quantized, perplexity = model.encode(images)

    logits_real = model.pixelcnn_decoder(images, z_quantized)
    _, bits_real = compute_pixel_nll_loss(logits_real, images)

    logits_blind = model.pixelcnn_decoder(images, torch.zeros_like(z_quantized))
    _, bits_blind = compute_pixel_nll_loss(logits_blind, images)

    return {
        'bits_per_dim_real_z': bits_real.item(),
        'bits_per_dim_zeroed_z': bits_blind.item(),
        'latent_gap_bits': (bits_blind - bits_real).item(),
        'perplexity': perplexity.item(),
    }
