from config import (
    METRICS_DIR, EXPERIMENTS_CSV, ENCODER_DIM, EMBED_DIM, DECODER_DIM,
    ATTENTION_DIM, D_MODEL, NHEAD, NUM_DECODER_LAYERS, DIM_FEEDFORWARD
)
from experiment_utils import count_parameters, log_experiment
from models.encoder import EncoderCNN
from models.lstm_decoder import DecoderLSTM
from models.transformer_decoder import DecoderTransformer
from vocabulary import Vocabulary

vocab = Vocabulary()
vocab.load()
vocab_size = len(vocab)

encoder = EncoderCNN(fine_tune=False)
enc_total, enc_trainable = count_parameters(encoder)

lstm_decoder = DecoderLSTM(vocab_size, EMBED_DIM, DECODER_DIM, ATTENTION_DIM, ENCODER_DIM)
lstm_total, lstm_trainable = count_parameters(lstm_decoder)

transformer_decoder = DecoderTransformer(vocab_size, D_MODEL, NHEAD, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, 0.1, ENCODER_DIM)
trans_total, trans_trainable = count_parameters(transformer_decoder)

log_experiment(EXPERIMENTS_CSV, {
    "experiment_id": "lstm_baseline",
    "model": "LSTM+Attention",
    "seed": 42, "hardware": "Colab T4", "epochs_run": 15, "best_epoch": 8,
    "batch_size": 32, "learning_rate": 4e-4, "optimizer": "Adam",
    "weight_decay": 0.0, "dropout": 0.5, "encoder_frozen": True,
    "total_params": enc_total + lstm_total,
    "trainable_params": enc_trainable + lstm_trainable,
    "best_val_loss": 2.6499,
    "training_time_min": round(411.0 * 15 / 60, 1),
    "notes": "Overfits after epoch 8; checkpoint-selected, no early stopping active yet",
})

log_experiment(EXPERIMENTS_CSV, {
    "experiment_id": "transformer_baseline",
    "model": "Transformer",
    "seed": 42, "hardware": "Colab T4", "epochs_run": 7, "best_epoch": 4,
    "batch_size": 32, "learning_rate": 3e-4, "optimizer": "Adam",
    "weight_decay": 0.0, "dropout": 0.1, "encoder_frozen": True,
    "total_params": enc_total + trans_total,
    "trainable_params": enc_trainable + trans_trainable,
    "best_val_loss": 2.6668,
    "training_time_min": round(394.0 * 7 / 60, 1),
    "notes": "Early-stopped at epoch 7; converged ~2x faster than LSTM",
})

print("\nBaseline experiments logged.")