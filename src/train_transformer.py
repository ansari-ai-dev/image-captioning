from pathlib import Path
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    TRAIN_CSV, VAL_CSV, BATCH_SIZE, CHECKPOINT_DIR, SEED,
    D_MODEL, NHEAD, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, TRANSFORMER_DROPOUT,
    ENCODER_DIM, TRANSFORMER_LR, NUM_EPOCHS, GRAD_CLIP, EARLY_STOP_PATIENCE
)
from vocabulary import Vocabulary
from transforms import train_transform, eval_transform
from dataset import FlickrDataset, collate_fn
from models.encoder import EncoderCNN
from models.transformer_decoder import DecoderTransformer

SANITY_CHECK = True  # flip to False only for the real Colab run

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

vocab = Vocabulary()
vocab.load()
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

train_dataset = FlickrDataset(TRAIN_CSV, vocab, train_transform)
val_dataset = FlickrDataset(VAL_CSV, vocab, eval_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

encoder = EncoderCNN(fine_tune=False).to(device)  # SAME frozen encoder as the LSTM run
decoder = DecoderTransformer(
    vocab_size, D_MODEL, NHEAD, NUM_DECODER_LAYERS,
    DIM_FEEDFORWARD, TRANSFORMER_DROPOUT, ENCODER_DIM
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(decoder.parameters(), lr=TRANSFORMER_LR)


def run_epoch(loader, training, max_batches=None):
    encoder.eval()
    decoder.train(training)
    total_loss, n_batches = 0.0, 0

    for i, (images, captions, lengths) in enumerate(tqdm(loader, desc="train" if training else "val")):
        if max_batches and i >= max_batches:
            break
        images, captions = images.to(device), captions.to(device)

        with torch.set_grad_enabled(training):
            encoder_out = encoder(images)
            outputs, _ = decoder(encoder_out, captions)
            targets = captions[:, 1:]  # same alignment as the LSTM: predict token t+1

            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


if __name__ == "__main__":
    if SANITY_CHECK:
        print("\n=== SANITY CHECK (3 batches, 2 epochs) ===")
        for epoch in range(2):
            tr_loss = run_epoch(train_loader, training=True, max_batches=3)
            val_loss = run_epoch(val_loader, training=False, max_batches=3)
            print(f"Epoch {epoch+1}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")
        print("\nIf these losses are finite (not NaN/inf) and roughly stable or "
              "decreasing, the pipeline is wired correctly.")
    else:
        print(f"\n=== FULL TRAINING ({NUM_EPOCHS} epochs, early stop patience={EARLY_STOP_PATIENCE}) ===")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        Path("results/figures").mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        train_losses, val_losses = [], []

        for epoch in range(NUM_EPOCHS):
            start = time.time()
            tr_loss = run_epoch(train_loader, training=True)
            val_loss = run_epoch(val_loader, training=False)
            elapsed = time.time() - start
            train_losses.append(tr_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train={tr_loss:.4f} | "
                  f"val={val_loss:.4f} | {elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save({
                    "encoder_state": encoder.state_dict(),
                    "decoder_state": decoder.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, CHECKPOINT_DIR / "transformer_best.pth")
                print(f"  -> best so far, checkpoint saved (val_loss={val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"  -> no improvement ({epochs_without_improvement}/{EARLY_STOP_PATIENCE})")
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                    break

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", marker='o')
        plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Transformer: Training vs Validation Loss")
        plt.legend()
        plt.savefig("results/figures/transformer_loss_curve.png", dpi=150, bbox_inches='tight')
        print("\nLoss curve saved to results/figures/transformer_loss_curve.png")