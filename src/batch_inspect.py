from torch.utils.data import DataLoader

from config import TRAIN_CSV, VAL_CSV, BATCH_SIZE
from vocabulary import Vocabulary
from transforms import train_transform, eval_transform
from dataset import FlickrDataset, collate_fn

vocab = Vocabulary()
vocab.load()
print(f"Loaded vocabulary: {len(vocab)} tokens")

train_dataset = FlickrDataset(TRAIN_CSV, vocab, train_transform)
val_dataset = FlickrDataset(VAL_CSV, vocab, eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

images, captions, lengths = next(iter(train_loader))
print("\n--- Sample batch ---")
print("Images shape   :", images.shape)   # expect [BATCH_SIZE, 3, 224, 224]
print("Captions shape :", captions.shape) # expect [BATCH_SIZE, max_len_in_batch]
print("Lengths        :", lengths[:8].tolist())

print("\n--- Decoded sample captions from this batch ---")
for i in range(3):
    print(f"  {vocab.decode(captions[i].tolist())}")

print("\nPixel value range check (should be roughly -2 to 2 after normalization):")
print("  min:", images.min().item(), "max:", images.max().item())