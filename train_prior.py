"""Optional Stage 2: class-conditional PixelCNN prior over the Stage 1 code grid.

Steps:
  1. Encode all CIFAR-10 images to 8x8 code grids with the frozen Stage 1 VQ-VAE.
  2. Train the prior to predict each code from the codes before it (cross-entropy).
  3. Sample new code grids per class and decode them with the Stage 1 decoder.

Small enough to train on a Colab T4 in about 15-20 minutes. Training on an
8 GB laptop is possible on MPS but slow; the tests in tests/ are the local check.
"""

import json
import math
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid

from vqvae.data import make_cifar10_loaders, pick_device
from vqvae.pixelcnn_prior import PixelCNNPrior
from vqvae.vqvae import VQ_VAE, load_stage1_checkpoint

# Hyperparameters
DATA_DIR = os.path.expanduser('~/data')
STAGE1_CHECKPOINT = 'checkpoints/vqvae_stage1.pth'
SAVE_PATH = 'checkpoints/pixelcnn_prior.pth'
HISTORY_PATH = 'results/prior_history.json'
SAMPLES_PATH = 'results/prior_samples.png'

BATCH_SIZE = 128
NUM_EPOCHS = 60
LEARNING_RATE = 3e-4
NUM_HIDDENS = 256
NUM_EMBEDDINGS = 512
EMBEDDING_DIM = 256
COMMITMENT_COST = 0.25
DECAY = 0.90
PRIOR_HIDDENS = 128
PRIOR_LAYERS = 12
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 8
SAMPLE_TEMPERATURE = 1.0
SEED = 67

CIFAR10_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

torch.manual_seed(SEED)
device = pick_device()
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 1. Encode the dataset once. The VQ-VAE is frozen, so the codes never change.
vqvae = VQ_VAE(NUM_HIDDENS, NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST, DECAY)
load_stage1_checkpoint(vqvae, STAGE1_CHECKPOINT, device='cpu')
vqvae = vqvae.to(device)
vqvae.eval()

image_train_loader, image_test_loader = make_cifar10_loaders(DATA_DIR, batch_size=500)


def encode_whole_split(image_loader):
    all_codes = []
    all_labels = []
    for images, labels in image_loader:
        codes = vqvae.encode_to_codes(images.to(device))
        all_codes.append(codes.cpu())
        all_labels.append(labels)
    return torch.cat(all_codes), torch.cat(all_labels)


train_codes, train_labels = encode_whole_split(image_train_loader)
test_codes, test_labels = encode_whole_split(image_test_loader)
codes_in_use = len(torch.unique(train_codes))
print(f'Encoded {len(train_codes)} train / {len(test_codes)} test images to {tuple(train_codes.shape[1:])} grids. '
      f'{codes_in_use} of {NUM_EMBEDDINGS} codes appear in the train set.')

train_loader = DataLoader(TensorDataset(train_codes, train_labels), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(test_codes, test_labels), batch_size=BATCH_SIZE, shuffle=False)

# 2. Train the prior
prior = PixelCNNPrior(NUM_EMBEDDINGS, num_hiddens=PRIOR_HIDDENS, num_layers=PRIOR_LAYERS,
                      num_classes=NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(prior.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

history = []
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    prior.train()
    epoch_loss = 0.0
    for codes, labels in train_loader:
        codes = codes.to(device)
        labels = labels.to(device)

        logits = prior(codes, labels)
        loss = F.cross_entropy(logits, codes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()

    prior.eval()
    test_loss = 0.0
    with torch.no_grad():
        for codes, labels in test_loader:
            codes = codes.to(device)
            labels = labels.to(device)
            logits = prior(codes, labels)
            test_loss += F.cross_entropy(logits, codes).item()

    # Bits per code: log2(512) = 9 bits would mean the prior learned nothing
    train_bits = epoch_loss / len(train_loader) / math.log(2)
    test_bits = test_loss / len(test_loader) / math.log(2)
    history.append({'epoch': epoch, 'train_bits_per_code': train_bits, 'test_bits_per_code': test_bits})
    print(f'Epoch {epoch}: train {train_bits:.3f} bits/code   test {test_bits:.3f} bits/code')

    torch.save(prior.state_dict(), SAVE_PATH)
    with open(HISTORY_PATH, 'w') as history_file:
        json.dump(history, history_file, indent=1)

print(f'Training done in {(time.time() - start_time) / 60:.1f} min')

# 3. Sample new code grids per class, then decode them with the Stage 1 decoder
prior.eval()
sample_labels = torch.arange(NUM_CLASSES).repeat_interleave(SAMPLES_PER_CLASS).to(device)
sampled_codes = prior.sample(sample_labels, grid_size=(8, 8), temperature=SAMPLE_TEMPERATURE)
generated_images = vqvae.decode_codes(sampled_codes)

grid = make_grid(generated_images.cpu(), nrow=SAMPLES_PER_CLASS, normalize=True, value_range=(-1, 1))
plt.figure(figsize=(SAMPLES_PER_CLASS * 1.2, NUM_CLASSES * 1.2))
plt.imshow(grid.permute(1, 2, 0))
plt.yticks([16 + 34 * row for row in range(NUM_CLASSES)], CIFAR10_CLASSES)
plt.xticks([])
plt.title('New images: PixelCNN prior samples decoded by the Stage 1 decoder')
plt.savefig(SAMPLES_PATH, bbox_inches='tight', dpi=150)
print(f'Saved {SAMPLES_PATH}')
