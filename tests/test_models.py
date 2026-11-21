"""CPU smoke tests. Tiny models only, so they run in seconds on a laptop.

Run with:  python -m pytest tests   (or)   python tests/test_models.py

The causality tests change ONE input position and check that no output
belonging to an earlier position changed. If any did, the PixelCNN would be
peeking at the value it is supposed to predict.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqvae.vqvae import VQ_VAE, load_stage1_checkpoint
from vqvae.pixelcnn_decoder import ConditionalPixelCNNDecoder, VQ_VAE_PixelCNN, latent_probe
from vqvae.pixelcnn_prior import PixelCNNPrior
from vqvae.losses import compute_pixel_nll_loss, images_to_pixel_bins

STAGE1_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'checkpoints', 'vqvae_stage1.pth')


def test_rgb_decoder_is_causal():
    torch.manual_seed(0)
    decoder = ConditionalPixelCNNDecoder(latent_channels=4, num_hiddens=12, num_layers=2, pixel_levels=8)
    decoder.eval()

    # The upsampler turns a (H/4) latent into H, so pick a size divisible by 4
    image = torch.rand(1, 3, 8, 8) * 2 - 1
    z_quantized = torch.randn(1, 4, 2, 2)
    base_logits = decoder(image, z_quantized)

    height, width = 8, 8
    for changed_row in range(height):
        for changed_col in range(width):
            for changed_channel in range(3):
                perturbed = image.clone()
                perturbed[0, changed_channel, changed_row, changed_col] += 1.0
                new_logits = decoder(perturbed, z_quantized)
                logit_change = (new_logits - base_logits).abs().amax(dim=2)[0]  # (3, H, W)

                for row in range(height):
                    for col in range(width):
                        for channel in range(3):
                            earlier_pixel = (row, col) < (changed_row, changed_col)
                            same_pixel_not_later_color = ((row, col) == (changed_row, changed_col)
                                                          and channel <= changed_channel)
                            if earlier_pixel or same_pixel_not_later_color:
                                assert logit_change[channel, row, col] < 1e-5, (
                                    f'output (c={channel}, {row},{col}) saw input '
                                    f'(c={changed_channel}, {changed_row},{changed_col})')


def test_rgb_decoder_actually_uses_context():
    # Guards against a mask so strict that the model sees nothing at all
    torch.manual_seed(0)
    decoder = ConditionalPixelCNNDecoder(latent_channels=4, num_hiddens=12, num_layers=2, pixel_levels=8)
    image = torch.rand(1, 3, 8, 8) * 2 - 1
    z_quantized = torch.randn(1, 4, 2, 2)
    base_logits = decoder(image, z_quantized)

    perturbed = image.clone()
    perturbed[0, 0, 3, 3] += 1.0  # red at (3,3)
    new_logits = decoder(perturbed, z_quantized)

    green_same_pixel = (new_logits - base_logits)[0, 1, :, 3, 3].abs().max()
    next_pixel = (new_logits - base_logits)[0, :, :, 3, 4].abs().max()
    pixel_below = (new_logits - base_logits)[0, :, :, 4, 3].abs().max()
    assert green_same_pixel > 1e-6
    assert next_pixel > 1e-6
    assert pixel_below > 1e-6


def test_prior_is_causal():
    torch.manual_seed(0)
    num_codes = 16
    prior = PixelCNNPrior(num_embeddings=num_codes, num_hiddens=16, num_layers=3, num_classes=10)
    prior.eval()

    codes = torch.randint(0, num_codes, (1, 8, 8))
    labels = torch.tensor([3])
    base_logits = prior(codes, labels)

    for changed_row in range(8):
        for changed_col in range(8):
            perturbed = codes.clone()
            perturbed[0, changed_row, changed_col] = (perturbed[0, changed_row, changed_col] + 1) % num_codes
            new_logits = prior(perturbed, labels)
            logit_change = (new_logits - base_logits).abs().amax(dim=1)[0]  # (H, W)

            for row in range(8):
                for col in range(8):
                    if (row, col) <= (changed_row, changed_col):
                        assert logit_change[row, col] < 1e-5, (
                            f'prior output ({row},{col}) saw code ({changed_row},{changed_col})')

    # The prior must also see something: the next position and the row below should react
    perturbed = codes.clone()
    perturbed[0, 3, 3] = (perturbed[0, 3, 3] + 1) % num_codes
    new_logits = prior(perturbed, labels)
    assert (new_logits - base_logits)[0, :, 3, 4].abs().max() > 1e-6
    assert (new_logits - base_logits)[0, :, 4, 3].abs().max() > 1e-6


def test_prior_sampling_shapes():
    prior = PixelCNNPrior(num_embeddings=16, num_hiddens=16, num_layers=2, num_classes=10)
    labels = torch.tensor([0, 9])
    codes = prior.sample(labels, grid_size=(8, 8))
    assert codes.shape == (2, 8, 8)
    assert codes.min() >= 0 and codes.max() < 16


def test_vqvae_shapes_and_ema_buffer():
    model = VQ_VAE(num_hiddens=32, num_embeddings=16, embedding_dim=32, commitment_cost=0.25, decay=0.9)
    images = torch.rand(4, 3, 32, 32) * 2 - 1
    reconstruction, vq_loss, perplexity = model(images)
    assert reconstruction.shape == images.shape

    # _ema_w must not be trainable
    parameter_names = [name for name, _ in model.named_parameters()]
    assert not any('_ema_w' in name for name in parameter_names)

    codes = model.encode_to_codes(images)
    assert codes.shape == (4, 8, 8)
    decoded = model.decode_codes(codes)
    assert decoded.shape == images.shape


def test_pixel_bins_round_trip():
    images = torch.linspace(-1, 1, 256).view(1, 1, 16, 16)
    bins = images_to_pixel_bins(images)
    assert bins.min() == 0 and bins.max() == 255


def test_pixelcnn_vqvae_forward_and_probe():
    model = VQ_VAE_PixelCNN(num_hiddens=32, num_embeddings=16, embedding_dim=32, commitment_cost=0.25,
                            decay=0.9, pixelcnn_hiddens=12, pixelcnn_layers=2)
    images = torch.rand(2, 3, 32, 32) * 2 - 1
    pixel_logits, vq_loss, perplexity = model(images)
    loss, bits_per_dim = compute_pixel_nll_loss(pixel_logits, images)
    loss.backward()
    assert pixel_logits.shape == (2, 3, 256, 32, 32)

    model.eval()
    probe = latent_probe(model, images)
    assert set(probe) == {'bits_per_dim_real_z', 'bits_per_dim_zeroed_z', 'latent_gap_bits', 'perplexity'}

    model.train()
    model.set_encoder_frozen(True)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert not model.quantizer.training
    model.set_encoder_frozen(False)
    assert all(p.requires_grad for p in model.encoder.parameters())


def test_old_stage1_checkpoint_loads():
    if not os.path.exists(STAGE1_CHECKPOINT):
        print('  (skipped: checkpoints/vqvae_stage1.pth not found)')
        return
    model = VQ_VAE(num_hiddens=256, num_embeddings=512, embedding_dim=256, commitment_cost=0.25, decay=0.90)
    load_stage1_checkpoint(model, STAGE1_CHECKPOINT, device='cpu')
    model.eval()
    images = torch.rand(2, 3, 32, 32) * 2 - 1
    reconstruction, _, _ = model(images)
    assert reconstruction.shape == images.shape


if __name__ == '__main__':
    tests = [value for name, value in list(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
        print(f'PASS  {test.__name__}')
