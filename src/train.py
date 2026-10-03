import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    TRAIN_CSV, VAL_CSV, BATCH_SIZE, CHECKPOINT_DIR, SEED,
    EMBED_DIM, DECODER_DIM, ATTENTION_DIM, ENCODER_DIM,
    LEARNING_RATE, NUM_EPOCHS, GRAD_CLIP
)
from vocabulary import Vocabulary
from transforms import train_transform, eval_transform
from dataset import FlickrDataset, collate_fn
from models.encoder import EncoderCNN
from models.lstm_decoder import DecoderLSTM

SANITY_CHECK = True  # flip to False only when running the real training on Colab

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

    for i, (images, captions, lengths) in enumerate(loader):
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
        print(f"\n=== FULL TRAINING ({NUM_EPOCHS} epochs) ===")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")
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
                torch.save({
                    "encoder_state": encoder.state_dict(),
                    "decoder_state": decoder.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, CHECKPOINT_DIR / "lstm_best.pth")
                print(f"  -> best so far, checkpoint saved (val_loss={val_loss:.4f})")

        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("LSTM + Attention: Training vs Validation Loss")
        plt.legend()
        plt.savefig("results/figures/lstm_loss_curve.png")
        print("\nLoss curve saved to results/figures/lstm_loss_curve.png")