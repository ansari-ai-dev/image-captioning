from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
VOCAB_PATH = PROJECT_ROOT / "data" / "vocab.json"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# --- Reproducibility ---
SEED = 42

# --- Dataset split (already applied in Phase 2 — recorded here for the report) ---
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
# Split performed at the image level (not caption level) to prevent
# any image's captions from appearing in more than one split.

# --- Image preprocessing ---
IMAGE_SIZE = 224  # required input size for pretrained ResNet50
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # ResNet50 was trained on ImageNet;
IMAGENET_STD = [0.229, 0.224, 0.225]   # inputs must be normalized to match

# --- Vocabulary ---
MIN_WORD_FREQ = 5  # words appearing fewer times than this become <UNK>
PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN = "<PAD>", "<START>", "<END>", "<UNK>"

# --- Training (used starting Phase 4) ---
BATCH_SIZE = 32
DEVICE = "cuda"  # set at runtime via torch.cuda.is_available() check in train.py

# --- Model hyperparameters (LSTM baseline) ---
EMBED_DIM = 256
DECODER_DIM = 512
ATTENTION_DIM = 256
ENCODER_DIM = 2048  # ResNet50's output channel count

# --- Training hyperparameters ---
LEARNING_RATE = 4e-4
NUM_EPOCHS = 15
GRAD_CLIP = 5.0
EARLY_STOP_PATIENCE = 3

# --- Model hyperparameters (Transformer) ---
D_MODEL = 512           # Transformer's internal dimension (analogous to DECODER_DIM)
NHEAD = 8                # number of attention heads
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 2048
TRANSFORMER_DROPOUT = 0.1

# --- Transformer training ---
TRANSFORMER_LR = 3e-4