import matplotlib.pyplot as plt
from pathlib import Path

# From the training run (7 epochs, early stopping triggered)
train_losses = [3.3325, 2.6391, 2.3810, 2.2084, 2.0686, 1.9468, 1.8421]
val_losses   = [2.8727, 2.7110, 2.6705, 2.6668, 2.6716, 2.7030, 2.7455]

Path("results/figures").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 8), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, 8), val_losses, label="Val Loss", marker='o')
plt.axvline(x=4, color='gray', linestyle='--', alpha=0.5, label="Best checkpoint (epoch 4)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Transformer: Training vs Validation Loss")
plt.legend()
plt.savefig("results/figures/transformer_loss_curve.png", dpi=150, bbox_inches='tight')
print("Saved to results/figures/transformer_loss_curve.png")