# --- Reproducibility ---
SEED = 42

# --- Dataset split ---
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
# Split performed at the image level (not caption level) to prevent
# any image's captions from appearing in more than one split.