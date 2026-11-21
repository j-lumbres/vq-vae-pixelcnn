"""Stage 1: train the VQ-VAE with the 0.84 SSIM + 0.16 L1 loss.

Optional: checkpoints/vqvae_stage1.pth already holds a 100-epoch run of this
exact setup. Run this only to reproduce it. Needs a GPU (Colab T4 or better).
"""

import json
import os
import time

import torch

from vqvae.data import make_cifar10_loaders, pick_device
from vqvae.losses import calculate_ssim, compute_mixed_loss
from vqvae.vqvae import VQ_VAE

# Hyperparameters
DATA_DIR = os.path.expanduser('~/data')
SAVE_PATH = 'checkpoints/vqvae_ssim_l1_retrained.pth'
HISTORY_PATH = 'results/vqvae_history.json'

BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
NUM_HIDDENS = 256
NUM_EMBEDDINGS = 512
EMBEDDING_DIM = 256
COMMITMENT_COST = 0.25
DECAY = 0.90
SEED = 67

torch.manual_seed(SEED)
device = pick_device()
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)

train_loader, test_loader = make_cifar10_loaders(DATA_DIR, BATCH_SIZE)

model = VQ_VAE(NUM_HIDDENS, NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST, DECAY).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LEARNING_RATE)

history = []
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_ssim = 0.0
    epoch_vq_loss = 0.0
    epoch_perplexity = 0.0

    for X_train, _ in train_loader:
        X_train = X_train.to(device)

        reconstruction, vq_loss, perplexity = model(X_train)
        losses = compute_mixed_loss(X_train, reconstruction, vq_loss)

        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        epoch_loss += losses['total'].item()
        epoch_ssim += losses['ssim_score'].item()
        epoch_vq_loss += vq_loss.item()
        epoch_perplexity += perplexity.item()

    scheduler.step()

    # Test-set SSIM, so the curve reflects images the model never trained on
    model.eval()
    test_ssim = 0.0
    with torch.no_grad():
        for X_test, _ in test_loader:
            X_test = X_test.to(device)
            reconstruction, _, _ = model(X_test)
            test_ssim += calculate_ssim(X_test, reconstruction).item()

    num_batches = len(train_loader)
    epoch_record = {
        'epoch': epoch,
        'train_loss': epoch_loss / num_batches,
        'train_ssim': epoch_ssim / num_batches,
        'vq_loss': epoch_vq_loss / num_batches,
        'perplexity': epoch_perplexity / num_batches,
        'test_ssim': test_ssim / len(test_loader),
    }
    history.append(epoch_record)
    print(f"Epoch {epoch}: loss {epoch_record['train_loss']:.4f}  "
          f"train SSIM {epoch_record['train_ssim']:.4f}  test SSIM {epoch_record['test_ssim']:.4f}  "
          f"VQ {epoch_record['vq_loss']:.4f}  perplexity {epoch_record['perplexity']:.1f}")

    torch.save(model.state_dict(), SAVE_PATH)
    with open(HISTORY_PATH, 'w') as history_file:
        json.dump(history, history_file, indent=1)

print(f'Done in {(time.time() - start_time) / 60:.1f} min. Saved {SAVE_PATH}')
