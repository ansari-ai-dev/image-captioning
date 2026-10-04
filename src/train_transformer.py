from pathlib import Path
from tqdm import tqdm
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    TRAIN_CSV, VAL_CSV, BATCH_SIZE, CHECKPOINT_DIR, SEED,
    D_MODEL, NHEAD, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, TRANSFORMER_DROPOUT,
    ENCODER_DIM, TRANSFORMER_LR, NUM_EPOCHS, GRAD_CLIP, EARLY_STOP_PATIENCE,
    FIGURES_DIR, METRICS_DIR, EXPERIMENTS_CSV, WEIGHT_DECAY
)
from vocabulary import Vocabulary
from transforms import train_transform, eval_transform
from dataset import FlickrDataset, collate_fn
from models.encoder import EncoderCNN
from models.transformer_decoder import DecoderTransformer
from experiment_utils import count_parameters, log_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
parser.add_argument("--experiment_name", type=str, default="transformer_baseline")
parser.add_argument("--sanity_check", action="store_true")
args = parser.parse_args()

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(f"Experiment: {args.experiment_name} | weight_decay={args.weight_decay}")

vocab = Vocabulary()
vocab.load()
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

train_dataset = FlickrDataset(TRAIN_CSV, vocab, train_transform)
val_dataset = FlickrDataset(VAL_CSV, vocab, eval_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

encoder = EncoderCNN(fine_tune=False).to(device)
decoder = DecoderTransformer(
    vocab_size, D_MODEL, NHEAD, NUM_DECODER_LAYERS,
    DIM_FEEDFORWARD, TRANSFORMER_DROPOUT, ENCODER_DIM
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(decoder.parameters(), lr=TRANSFORMER_LR, weight_decay=args.weight_decay)


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
            targets = captions[:, 1:]
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
    if args.sanity_check:
        print("\n=== SANITY CHECK (3 batches, 2 epochs) ===")
        for epoch in range(2):
            tr_loss = run_epoch(train_loader, training=True, max_batches=3)
            val_loss = run_epoch(val_loader, training=False, max_batches=3)
            print(f"Epoch {epoch+1}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")
        print("\nSanity check complete.")
    else:
        print(f"\n=== FULL TRAINING: {args.experiment_name} "
              f"({NUM_EPOCHS} epochs, early stop patience={EARLY_STOP_PATIENCE}) ===")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        train_losses, val_losses = [], []
        total_train_time = 0.0

        for epoch in range(NUM_EPOCHS):
            start = time.time()
            tr_loss = run_epoch(train_loader, training=True)
            val_loss = run_epoch(val_loader, training=False)
            elapsed = time.time() - start
            total_train_time += elapsed
            train_losses.append(tr_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train={tr_loss:.4f} | "
                  f"val={val_loss:.4f} | {elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save({
                    "encoder_state": encoder.state_dict(),
                    "decoder_state": decoder.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, CHECKPOINT_DIR / f"{args.experiment_name}_best.pth")
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
        plt.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5, label=f"Best checkpoint (epoch {best_epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"Transformer [{args.experiment_name}]: Training vs Validation Loss")
        plt.legend()
        fig_path = FIGURES_DIR / f"{args.experiment_name}_loss_curve.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nLoss curve saved to {fig_path}")

        enc_total, enc_trainable = count_parameters(encoder)
        dec_total, dec_trainable = count_parameters(decoder)
        log_experiment(EXPERIMENTS_CSV, {
            "experiment_id": args.experiment_name,
            "model": "Transformer",
            "seed": SEED, "hardware": "Colab T4" if torch.cuda.is_available() else "CPU",
            "epochs_run": len(train_losses), "best_epoch": best_epoch,
            "batch_size": BATCH_SIZE, "learning_rate": TRANSFORMER_LR, "optimizer": "Adam",
            "weight_decay": args.weight_decay, "dropout": TRANSFORMER_DROPOUT, "encoder_frozen": True,
            "total_params": enc_total + dec_total, "trainable_params": enc_trainable + dec_trainable,
            "best_val_loss": round(best_val_loss, 4),
            "training_time_min": round(total_train_time / 60, 1),
            "notes": f"weight_decay={args.weight_decay} regularization experiment",
        })