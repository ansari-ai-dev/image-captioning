import matplotlib.pyplot as plt
from pathlib import Path

train_losses = [3.8618, 3.0634, 2.7777, 2.5846, 2.4420, 2.3190, 2.2145, 2.1229,
                2.0386, 1.9654, 1.8963, 1.8317, 1.7742, 1.7164, 1.6659]
val_losses   = [3.2095, 2.9181, 2.7869, 2.7265, 2.6879, 2.6678, 2.6535, 2.6499,
                2.6630, 2.6676, 2.6687, 2.6941, 2.7019, 2.7298, 2.7363]

Path("results/figures").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 16), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, 16), val_losses, label="Val Loss", marker='o')
plt.axvline(x=8, color='gray', linestyle='--', alpha=0.5, label="Best checkpoint (epoch 8)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("LSTM + Attention: Training vs Validation Loss")
plt.legend()
plt.savefig("results/figures/lstm_loss_curve.png", dpi=150, bbox_inches='tight')
print("Saved to results/figures/lstm_loss_curve.png")