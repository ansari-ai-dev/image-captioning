import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

from config import IMAGES_DIR
from vocabulary import Vocabulary


class FlickrDataset(Dataset):
    def __init__(self, csv_path, vocab: Vocabulary, transform):
        self.transform = transform
        self.vocab = vocab
        self.samples = []  # list of (image_filename, caption_str)

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image"], row["caption"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        image = Image.open(IMAGES_DIR / img_name).convert("RGB")
        image = self.transform(image)

        token_ids = self.vocab.encode(caption)
        caption_tensor = torch.tensor(token_ids, dtype=torch.long)

        return image, caption_tensor


def collate_fn(batch):
    """Pads variable-length caption sequences within a batch to the same length."""
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)

    lengths = [len(c) for c in captions]
    max_len = max(lengths)
    pad_id = 0  # <PAD> is always index 0 by construction in Vocabulary.build()

    padded = torch.full((len(captions), max_len), pad_id, dtype=torch.long)
    for i, c in enumerate(captions):
        padded[i, :len(c)] = c

    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, padded, lengths