"""Stage 1: the VQ-VAE (encoder, EMA vector quantizer, deterministic decoder).

Follows van den Oord et al., "Neural Discrete Representation Learning" (2017).
A 32x32 CIFAR-10 image becomes an 8x8 grid of codebook indices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """input + (ReLU -> 3x3 conv -> ReLU -> 1x1 conv)."""

    def __init__(self, num_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens, num_hiddens, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """Maps a 3x32x32 image to a num_hiddens x 8x8 feature map (two stride-2 convs)."""

    def __init__(self, num_hiddens):
        super().__init__()
        self.conv1 = nn.Conv2d(3, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual1 = ResidualBlock(num_hiddens)
        self.residual2 = ResidualBlock(num_hiddens)

    def forward(self, x):
        # 32x32 -> 16x16
        x = self.conv1(x)
        x = F.relu(x)

        # 16x16 -> 8x8
        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.residual1(x)
        x = self.residual2(x)
        return x


class Decoder(nn.Module):
    """Maps a quantized 8x8 latent back to a 3x32x32 image in [-1, 1]."""

    def __init__(self, num_hiddens):
        super().__init__()
        self.conv = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual1 = ResidualBlock(num_hiddens)
        self.residual2 = ResidualBlock(num_hiddens)
        self.deconv1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(num_hiddens // 2, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, z_quantized):
        x = self.conv(z_quantized)
        x = self.residual1(x)
        x = self.residual2(x)

        # 8x8 -> 16x16
        x = self.deconv1(x)
        x = F.relu(x)

        # 16x16 -> 32x32
        x = self.deconv2(x)

        # tanh keeps the output in [-1, 1], the same range as the normalized input images
        x = torch.tanh(x)
        return x


class VectorQuantizerEMA(nn.Module):
    """Codebook updated with exponential moving averages instead of a codebook loss.

    Two changes from the original notebook version:
      1. _ema_w is a buffer, not an nn.Parameter. It never receives gradients, so it
         should not sit inside the optimizer. (Old checkpoints still load: buffers and
         parameters share the same state_dict key.)
      2. The EMA state starts equal to the codebook (cluster size 1, _ema_w = codebook).
         Before, it started at cluster size 0 with random _ema_w, so every code that
         missed the first batch was divided by ~1e-5 and blew up to a huge vector.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay,
                 epsilon=1e-5, dead_code_threshold=0.03):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # A code's usage tracker is an EMA (weight 0.01) of how many vectors picked it per batch.
        # It starts at 1.0, so an unused code falls below 0.03 after about
        # log(0.03) / log(0.99) = 349 steps, roughly one CIFAR-10 epoch at batch size 128.
        # The old threshold of 1.0 reset any code used less than once per batch, every step.
        self.dead_code_threshold = dead_code_threshold

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        # EMA state: only the update step inside forward() should touch these
        self.register_buffer('_ema_cluster_size', torch.ones(num_embeddings))
        self.register_buffer('_ema_w', self.embedding.weight.data.clone())
        self.register_buffer('_usage_tracker', torch.ones(num_embeddings))

    def find_nearest_codes(self, flat_z_encoded):
        # L2 distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        distances = (torch.sum(flat_z_encoded ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_z_encoded, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices

    def forward(self, z_encoded):
        # z_encoded: (batch, embedding_dim, height, width)
        batch_size, channels, height, width = z_encoded.shape

        # Move channels last so each spatial position is one embedding_dim vector
        z_encoded = z_encoded.permute(0, 2, 3, 1).contiguous()
        flat_z_encoded = z_encoded.view(-1, self.embedding_dim)

        encoding_indices = self.find_nearest_codes(flat_z_encoded)
        encoding_indices = encoding_indices.unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z_encoded.device)
        encodings.scatter_(1, encoding_indices, 1)

        z_quantized = torch.matmul(encodings, self.embedding.weight)
        z_quantized = z_quantized.view(batch_size, height, width, channels)

        if self.training:
            self.update_codebook(encodings, flat_z_encoded)

        # Straight-through estimator: the forward pass uses z_quantized,
        # but gradients flow back into z_encoded as if quantization were the identity
        z_quantized_ste = z_encoded + (z_quantized - z_encoded).detach()

        # With EMA updates the codebook needs no loss term, only the commitment term
        # that keeps the encoder output close to its chosen code
        commitment_loss = self.commitment_cost * F.mse_loss(z_encoded, z_quantized.detach())

        # Perplexity = effective number of codes in use (max = num_embeddings)
        average_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(average_probs * torch.log(average_probs + 1e-10)))

        z_quantized_ste = z_quantized_ste.permute(0, 3, 1, 2).contiguous()
        return commitment_loss, z_quantized_ste, perplexity

    @torch.no_grad()
    def update_codebook(self, encodings, flat_z_encoded):
        codes_per_batch = torch.sum(encodings, dim=0)

        self._ema_cluster_size.mul_(self.decay).add_(codes_per_batch * (1 - self.decay))

        # Laplace smoothing so a cluster size can never reach exactly zero
        total_count = torch.sum(self._ema_cluster_size)
        smoothed_cluster_size = ((self._ema_cluster_size + self.epsilon)
                                 / (total_count + self.num_embeddings * self.epsilon)
                                 * total_count)
        self._ema_cluster_size.copy_(smoothed_cluster_size)

        # Sum of the encoder vectors assigned to each code
        summed_vectors = torch.matmul(encodings.t(), flat_z_encoded)
        self._ema_w.mul_(self.decay).add_(summed_vectors * (1 - self.decay))

        # Each code moves to the running mean of the vectors assigned to it
        self.embedding.weight.data.copy_(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Restart dead codes on random encoder outputs from this batch
        self._usage_tracker.mul_(0.99).add_(codes_per_batch * 0.01)
        dead_mask = self._usage_tracker < self.dead_code_threshold
        num_dead = int(dead_mask.sum().item())

        if num_dead > 0:
            num_dead = min(num_dead, flat_z_encoded.shape[0])
            dead_indices = dead_mask.nonzero(as_tuple=True)[0][:num_dead]
            random_rows = torch.randperm(flat_z_encoded.shape[0], device=flat_z_encoded.device)[:num_dead]
            sampled_vectors = flat_z_encoded[random_rows]

            self.embedding.weight.data[dead_indices] = sampled_vectors
            self._ema_w[dead_indices] = sampled_vectors
            self._ema_cluster_size[dead_indices] = 1.0
            self._usage_tracker[dead_indices] = 1.0

    @torch.no_grad()
    def get_code_indices(self, z_encoded):
        # Returns the (batch, height, width) grid of code indices, used to train the prior
        batch_size, channels, height, width = z_encoded.shape
        z_encoded = z_encoded.permute(0, 2, 3, 1).contiguous()
        flat_z_encoded = z_encoded.view(-1, self.embedding_dim)

        encoding_indices = self.find_nearest_codes(flat_z_encoded)
        return encoding_indices.view(batch_size, height, width)

    def lookup_codes(self, code_indices):
        # (batch, height, width) indices -> (batch, embedding_dim, height, width) vectors
        z_quantized = self.embedding(code_indices)
        z_quantized = z_quantized.permute(0, 3, 1, 2).contiguous()
        return z_quantized


class VQ_VAE(nn.Module):
    # The decoder reads z_quantized directly, so embedding_dim must equal num_hiddens (256 in all runs)
    def __init__(self, num_hiddens, num_embeddings, embedding_dim, commitment_cost, decay):
        super().__init__()
        self.encoder = Encoder(num_hiddens)

        # 1x1 conv projects encoder features to the codebook dimension
        self.pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1)
        self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        self.decoder = Decoder(num_hiddens)

    def forward(self, x):
        z_encoded = self.encoder(x)
        z_encoded = self.pre_vq_conv(z_encoded)

        vq_loss, z_quantized, perplexity = self.quantizer(z_encoded)

        reconstruction = self.decoder(z_quantized)
        return reconstruction, vq_loss, perplexity

    @torch.no_grad()
    def encode_to_codes(self, x):
        z_encoded = self.encoder(x)
        z_encoded = self.pre_vq_conv(z_encoded)
        return self.quantizer.get_code_indices(z_encoded)

    @torch.no_grad()
    def decode_codes(self, code_indices):
        z_quantized = self.quantizer.lookup_codes(code_indices)
        return self.decoder(z_quantized)


# The original notebooks named submodules with a leading underscore
# (_encoder, _decoder._deconv_1, ...). This table maps those names to the new ones.
OLD_TO_NEW_NAMES = {
    '_encoder': 'encoder',
    '_pre_vq_conv': 'pre_vq_conv',
    '_quantizer': 'quantizer',
    '_embedding': 'embedding',
    '_conv_1': 'conv1',
    '_conv_2': 'conv2',
    '_conv_3': 'conv3',
    '_block': 'block',
    '_decoder': 'decoder',
    '_conv': 'conv',
    '_residual_1': 'residual1',
    '_residual_2': 'residual2',
    '_deconv_1': 'deconv1',
    '_deconv_2': 'deconv2',
}


def load_stage1_checkpoint(model, checkpoint_path, device):
    """Loads a VQ-VAE state_dict saved by either the old notebooks or this repo."""
    old_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    new_state_dict = {}
    for old_key, value in old_state_dict.items():
        name_parts = old_key.split('.')
        new_parts = []
        for part in name_parts:
            new_parts.append(OLD_TO_NEW_NAMES.get(part, part))
        new_key = '.'.join(new_parts)
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    return model
