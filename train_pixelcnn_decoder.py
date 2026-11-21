"""Main model: VQ-VAE with a conditional Gated PixelCNN decoder.

What changed from the collapsed notebook run (vqvae_ssim_l1_PixelCNN.ipynb,
stopped at epoch 3 with codebook perplexity ~5 and a latent gap of ~0.17 bits/dim):

  1. Warm start. Encoder, projection, and codebook load from the Stage 1
     checkpoint, so z already carries image content on step 1. Trained from
     scratch, the PixelCNN learns to predict from neighbouring pixels long
     before the random encoder says anything useful, and then never needs z.
  2. Frozen start. Those Stage 1 parts stay frozen for FREEZE_EPOCHS so the
     decoder has to learn to read z before the encoder can drift.
  3. Collapse monitoring. Every epoch logs codebook perplexity and the latent
     gap (test bits/dim with z zeroed minus with the real z). The gap is the
     number to watch, not SSIM: a sample from p(x | z) is a plausible image
     with the same content, not a pixel-exact copy, so SSIM stays low even
     when the model works.

Needs a GPU. Each epoch of 6 gated layers on 32x32x3 takes a few minutes on a T4.
An interrupted run continues from RESUME_PATH. Delete it to start over.
"""

import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from vqvae.data import make_cifar10_loaders, pick_device
from vqvae.losses import compute_pixel_nll_loss
from vqvae.pixelcnn_decoder import VQ_VAE_PixelCNN, latent_probe
from vqvae.vqvae import VQ_VAE, load_stage1_checkpoint

# Hyperparameters
DATA_DIR = os.path.expanduser('~/data')
STAGE1_CHECKPOINT = 'checkpoints/vqvae_stage1.pth'
SAVE_PATH = 'checkpoints/vqvae_pixelcnn_decoder.pth'
HISTORY_PATH = 'results/pixelcnn_decoder_history.json'
SAMPLES_PATH = 'results/pixelcnn_decoder_samples.png'
RESUME_PATH = 'checkpoints/vqvae_pixelcnn_decoder_resume.pth'

BATCH_SIZE = 128
NUM_EPOCHS = 40
FREEZE_EPOCHS = 10
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
NUM_HIDDENS = 256
NUM_EMBEDDINGS = 512
EMBEDDING_DIM = 256
COMMITMENT_COST = 0.25
DECAY = 0.90
PIXELCNN_HIDDENS = 120  # must be divisible by 3 (R, G, B channel groups)
PIXELCNN_LAYERS = 6
NUM_SAMPLE_IMAGES = 8
SEED = 67

torch.manual_seed(SEED)
device = pick_device()
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)

train_loader, test_loader = make_cifar10_loaders(DATA_DIR, BATCH_SIZE)

model = VQ_VAE_PixelCNN(NUM_HIDDENS, NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST, DECAY,
                        pixelcnn_hiddens=PIXELCNN_HIDDENS, pixelcnn_layers=PIXELCNN_LAYERS).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LEARNING_RATE)

history = []
start_epoch = 0

if os.path.exists(RESUME_PATH):
    resume_state = torch.load(RESUME_PATH, map_location=device, weights_only=True)
    model.load_state_dict(resume_state['model'])
    optimizer.load_state_dict(resume_state['optimizer'])
    scheduler.load_state_dict(resume_state['scheduler'])
    history = resume_state['history']
    start_epoch = resume_state['epoch'] + 1
    print(f'Resuming from {RESUME_PATH} at epoch {start_epoch}')

elif os.path.exists(SAVE_PATH) and os.path.exists(HISTORY_PATH):
    # Older runs saved weights only. Adam restarts with empty moment estimates,
    # but the learning rate goes back to where it was on the cosine schedule.
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
    with open(HISTORY_PATH) as history_file:
        history = json.load(history_file)
    start_epoch = history[-1]['epoch'] + 1
    for _ in range(start_epoch):
        scheduler.step()
    print(f'Resuming from weights-only {SAVE_PATH} at epoch {start_epoch}, fresh Adam state')

else:
    # Warm start: copy encoder, projection, and codebook from the trained Stage 1 VQ-VAE
    stage1_model = VQ_VAE(NUM_HIDDENS, NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST, DECAY)
    load_stage1_checkpoint(stage1_model, STAGE1_CHECKPOINT, device='cpu')
    model.encoder.load_state_dict(stage1_model.encoder.state_dict())
    model.pre_vq_conv.load_state_dict(stage1_model.pre_vq_conv.state_dict())
    model.quantizer.load_state_dict(stage1_model.quantizer.state_dict())
    del stage1_model

# One fixed test batch for the collapse probe, so epochs are comparable
X_probe, _ = next(iter(test_loader))
X_probe = X_probe.to(device)

start_time = time.time()

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    encoder_frozen = epoch < FREEZE_EPOCHS
    model.set_encoder_frozen(encoder_frozen)

    epoch_bits_per_dim = 0.0
    epoch_vq_loss = 0.0

    for X_train, _ in train_loader:
        X_train = X_train.to(device)

        # Teacher forcing: the decoder reads the real image as its autoregressive input
        pixel_logits, vq_loss, perplexity = model(X_train)
        pixel_nll_loss, bits_per_dim = compute_pixel_nll_loss(pixel_logits, X_train)
        total_loss = pixel_nll_loss + vq_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_bits_per_dim += bits_per_dim.item()
        epoch_vq_loss += vq_loss.item()

    scheduler.step()

    model.eval()
    probe = latent_probe(model, X_probe)

    epoch_record = {
        'epoch': epoch,
        'encoder_frozen': encoder_frozen,
        'train_bits_per_dim': epoch_bits_per_dim / len(train_loader),
        'vq_loss': epoch_vq_loss / len(train_loader),
    }
    epoch_record.update(probe)
    history.append(epoch_record)

    print(f"Epoch {epoch} {'[frozen]' if encoder_frozen else '        '} "
          f"train bits/dim {epoch_record['train_bits_per_dim']:.3f}  "
          f"test bits/dim {probe['bits_per_dim_real_z']:.3f}  "
          f"latent gap {probe['latent_gap_bits']:.3f}  "
          f"perplexity {probe['perplexity']:.1f}")

    # A shrinking gap after unfreezing is the early warning sign of posterior collapse
    if not encoder_frozen and probe['latent_gap_bits'] < 0.05:
        print('  WARNING: latent gap < 0.05 bits/dim, the decoder is close to ignoring z')

    torch.save(model.state_dict(), SAVE_PATH)
    with open(HISTORY_PATH, 'w') as history_file:
        json.dump(history, history_file, indent=1)

    resume_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'history': history,
    }
    # Write to a temp file first so a crash mid-save cannot corrupt the last good state
    torch.save(resume_state, RESUME_PATH + '.tmp')
    os.replace(RESUME_PATH + '.tmp', RESUME_PATH)

print(f'Training done in {(time.time() - start_time) / 60:.1f} min')

# Samples: top row originals, then two independent samples from p(x | z) per image.
# The two sample rows differ in detail but should share the layout and colours of the original.
model.eval()
X_show = X_probe[:NUM_SAMPLE_IMAGES]
with torch.no_grad():
    _, z_quantized, _ = model.encode(X_show)
    first_samples = model.pixelcnn_decoder.sample(z_quantized)
    second_samples = model.pixelcnn_decoder.sample(z_quantized)

grid = make_grid(torch.cat([X_show, first_samples, second_samples]).cpu(),
                 nrow=NUM_SAMPLE_IMAGES, normalize=True, value_range=(-1, 1))
plt.figure(figsize=(NUM_SAMPLE_IMAGES * 1.5, 5))
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title('Row 1: test images. Rows 2-3: two samples from the PixelCNN decoder given each latent')
plt.savefig(SAMPLES_PATH, bbox_inches='tight', dpi=150)
print(f'Saved {SAMPLES_PATH}')
