"""Optional Stage 2: a Gated PixelCNN prior over the VQ-VAE's 8x8 code grid.

This is how the VQ-VAE paper generates new images: learn p(codes) with an
autoregressive model, sample a code grid, then run the Stage 1 decoder on it.
Each position holds one of num_embeddings discrete codes, so there is no
R -> G -> B ordering, and sampling takes only 8 * 8 = 64 forward passes.

The prior can also be class-conditional (CIFAR-10 label -> learned bias),
as in Section 2.3 of the Gated PixelCNN paper.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorVerticalConv(nn.Module):
    """k x k conv that sees rows strictly above the current row."""

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size, padding=0)

    def forward(self, x):
        # Same trick as the RGB decoder: pad k rows on top, crop the extra bottom row
        half = self.kernel_size // 2
        x_padded = F.pad(x, (half, half, self.kernel_size, 0))
        features = self.conv(x_padded)
        return features[:, :, :-1, :]


class PriorHorizontalConv(nn.Module):
    """1 x k conv over the current row: pixels to the left, plus the current
    position itself when mask_type is 'B'."""

    def __init__(self, channels, kernel_size, mask_type):
        super().__init__()
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.empty(2 * channels, channels, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(2 * channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        center = kernel_size // 2
        mask = torch.zeros(2 * channels, channels, 1, kernel_size)
        mask[:, :, :, :center] = 1

        # The first layer reads the raw codes, so it must not see the code it is predicting.
        # Later layers only hold information about earlier positions, so the centre is safe.
        if mask_type == 'B':
            mask[:, :, :, center] = 1
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=(0, self.padding))


class GatedPriorLayer(nn.Module):
    def __init__(self, channels, kernel_size, mask_type, num_classes):
        super().__init__()
        self.mask_type = mask_type
        self.vertical_conv = PriorVerticalConv(channels, kernel_size)
        self.horizontal_conv = PriorHorizontalConv(channels, kernel_size, mask_type)
        self.vertical_to_horizontal = nn.Conv2d(2 * channels, 2 * channels, kernel_size=1)
        self.horizontal_output_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # One learned bias per class, added to both stacks (the label is the same everywhere)
        self.class_bias_vertical = nn.Embedding(num_classes, 2 * channels)
        self.class_bias_horizontal = nn.Embedding(num_classes, 2 * channels)

    def forward(self, vertical_input, horizontal_input, labels):
        class_vertical = self.class_bias_vertical(labels).unsqueeze(-1).unsqueeze(-1)
        class_horizontal = self.class_bias_horizontal(labels).unsqueeze(-1).unsqueeze(-1)

        vertical_features = self.vertical_conv(vertical_input) + class_vertical
        vertical_filter, vertical_gate = vertical_features.chunk(2, dim=1)
        vertical_output = torch.tanh(vertical_filter) * torch.sigmoid(vertical_gate)

        horizontal_features = self.horizontal_conv(horizontal_input)
        horizontal_features = horizontal_features + self.vertical_to_horizontal(vertical_features)
        horizontal_features = horizontal_features + class_horizontal
        horizontal_filter, horizontal_gate = horizontal_features.chunk(2, dim=1)
        horizontal_output = torch.tanh(horizontal_filter) * torch.sigmoid(horizontal_gate)
        horizontal_output = self.horizontal_output_conv(horizontal_output)

        if self.mask_type == 'B':
            horizontal_output = horizontal_output + horizontal_input

        return vertical_output, horizontal_output


class PixelCNNPrior(nn.Module):
    def __init__(self, num_embeddings, num_hiddens=128, num_layers=12, num_classes=10):
        super().__init__()
        self.num_embeddings = num_embeddings

        # Each discrete code becomes a learned vector the convolutions can read
        self.code_embedding = nn.Embedding(num_embeddings, num_hiddens)

        self.first_layer = GatedPriorLayer(num_hiddens, kernel_size=7, mask_type='A', num_classes=num_classes)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = GatedPriorLayer(num_hiddens, kernel_size=3, mask_type='B', num_classes=num_classes)
            self.hidden_layers.append(layer)

        self.output_conv1 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=1)
        self.output_conv2 = nn.Conv2d(num_hiddens, num_embeddings, kernel_size=1)

    def forward(self, code_indices, labels):
        # (batch, H, W) long -> (batch, num_hiddens, H, W)
        x = self.code_embedding(code_indices)
        x = x.permute(0, 3, 1, 2).contiguous()

        vertical, horizontal = self.first_layer(x, x, labels)
        for layer in self.hidden_layers:
            vertical, horizontal = layer(vertical, horizontal, labels)

        features = F.relu(horizontal)
        features = F.relu(self.output_conv1(features))
        logits = self.output_conv2(features)

        # (batch, num_embeddings, H, W): ready for F.cross_entropy against the code grid
        return logits

    @torch.no_grad()
    def sample(self, labels, grid_size, temperature=1.0):
        batch_size = labels.shape[0]
        height, width = grid_size
        code_indices = torch.zeros(batch_size, height, width, dtype=torch.long, device=labels.device)

        for row in range(height):
            for col in range(width):
                logits = self.forward(code_indices, labels)
                position_logits = logits[:, :, row, col] / temperature
                code_distribution = F.softmax(position_logits, dim=1)
                sampled_code = torch.multinomial(code_distribution, num_samples=1).squeeze(1)
                code_indices[:, row, col] = sampled_code

        return code_indices
