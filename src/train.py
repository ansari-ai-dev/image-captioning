from pathlib import Path
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    TRAIN_CSV, VAL_CSV, BATCH_SIZE, CHECKPOINT_DIR, SEED,
    EMBED_DIM, DECODER_DIM, ATTENTION_DIM, ENCODER_DIM,
    LEARNING_RATE, NUM_EPOCHS, GRAD_CLIP, EARLY_STOP_PATIENCE
)
from vocabulary import Vocabulary
from transforms import train_transform, eval_transform
from dataset import FlickrDataset, collate_fn
from models.encoder import EncoderCNN
from models.lstm_decoder import DecoderLSTM

SANITY_CHECK = False

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

encoder = EncoderCNN(fine_tune=False).to(device)
decoder = DecoderLSTM(vocab_size, EMBED_DIM, DECODER_DIM, ATTENTION_DIM, ENCODER_DIM).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # index 0 = <PAD>, excluded from loss
optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)  # encoder is frozen


def run_epoch(loader, training, max_batches=None):
    encoder.eval()          # frozen encoder: always eval mode (fixes BatchNorm stats)
    decoder.train(training)
    total_loss, n_batches = 0.0, 0

    for i, (images, captions, lengths) in enumerate(tqdm(loader, desc="train" if training else "val")):
        if max_batches and i >= max_batches:
            break
        images, captions = images.to(device), captions.to(device)

        with torch.set_grad_enabled(training):
            encoder_out = encoder(images)
            outputs, alphas = decoder(encoder_out, captions)
            targets = captions[:, 1:]  # skip <START>, predict the rest

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
        Path("results/figures").mkdir(parents=True, exist_ok=True)  # FIX: ensure folder exists before saving

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
                }, CHECKPOINT_DIR / "lstm_best.pth")
                print(f"  -> best so far, checkpoint saved (val_loss={val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"  -> no improvement ({epochs_without_improvement}/{EARLY_STOP_PATIENCE})")
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f"\nEarly stopping triggered at epoch {epoch+1} "
                          f"(no val improvement for {EARLY_STOP_PATIENCE} epochs).")
                    break

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", marker='o')
        plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("LSTM + Attention: Training vs Validation Loss")
        plt.legend()
        plt.savefig("results/figures/lstm_loss_curve.png", dpi=150, bbox_inches='tight')
        print("\nLoss curve saved to results/figures/lstm_loss_curve.png")