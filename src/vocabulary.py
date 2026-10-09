import json
import re
import csv
from collections import Counter

from config import (
    TRAIN_CSV, VOCAB_PATH, MIN_WORD_FREQ,
    PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN
)


def normalize_caption(caption: str) -> str:
    """Lowercase, strip most punctuation, collapse whitespace."""
    caption = caption.lower().strip()
    caption = re.sub(r"[^a-z0-9' ]+", " ", caption)  # keep letters, digits, apostrophes
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


def tokenize(caption: str) -> list[str]:
    return normalize_caption(caption).split(" ")


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def build(self, csv_path):
        counter = Counter()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                counter.update(tokenize(row["caption"]))

        # Special tokens always get fixed, low indices
        specials = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]
        words = [w for w, c in counter.items() if c >= MIN_WORD_FREQ]
        words = sorted(words)  # deterministic ordering — important for reproducibility

        vocab_list = specials + words
        self.word2idx = {w: i for i, w in enumerate(vocab_list)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        print(f"Vocabulary built: {len(vocab_list)} tokens "
              f"({len(specials)} special + {len(words)} words, "
              f"min_freq={MIN_WORD_FREQ})")
        print(f"Raw unique words before threshold: {len(counter)}")

    def encode(self, caption: str) -> list[int]:
        tokens = tokenize(caption)
        ids = [self.word2idx[START_TOKEN]]
        for t in tokens:
            ids.append(self.word2idx.get(t, self.word2idx[UNK_TOKEN]))
        ids.append(self.word2idx[END_TOKEN])
        return ids

    def decode(self, ids: list[int], skip_special=True) -> str:
        words = []
        for i in ids:
            w = self.idx2word.get(i, UNK_TOKEN)
            if skip_special and w == END_TOKEN:
                break
            if skip_special and w in (PAD_TOKEN, START_TOKEN, END_TOKEN):
                continue
            words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

    def save(self, path=VOCAB_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, indent=2)
        print(f"Vocabulary saved to {path}")

    def load(self, path=VOCAB_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        # fix: json keys load as str, idx2word needs int keys from values
        self.idx2word = {v: k for k, v in self.word2idx.items()}


if __name__ == "__main__":
    vocab = Vocabulary()
    vocab.build(TRAIN_CSV)
    vocab.save()

    # Quick sanity check
    sample = "A child in a pink dress is climbing up a set of stairs."
    encoded = vocab.encode(sample)
    decoded = vocab.decode(encoded)
    print("\nSample encode/decode check:")
    print("Original :", sample)
    print("Encoded  :", encoded)
    print("Decoded  :", decoded)