import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Injects information about token position, since Transformers have no
    inherent sense of sequence order (unlike LSTMs, which process step-by-step)."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 dim_feedforward, dropout, encoder_dim, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        # ResNet50 gives 2048-dim features; project to d_model so dimensions match
        self.encoder_proj = nn.Linear(encoder_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_causal_mask(self, size, device):
        """Upper-triangular mask: position i can only attend to positions <= i.
        This is what stops the decoder from 'cheating' by seeing future words."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def forward(self, encoder_out, captions):
        # encoder_out: [B, 49, 2048] from ResNet50
        # captions: [B, full_len] including <START> and <END>
        device = captions.device
        input_captions = captions[:, :-1]  # teacher forcing: feed all but the last token
        B, L = input_captions.shape

        memory = self.encoder_proj(encoder_out)  # [B, 49, d_model] -- cross-attention target

        tgt_emb = self.embedding(input_captions) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        tgt_mask = self.generate_causal_mask(L, device)
        tgt_key_padding_mask = (input_captions == 0)  # True where <PAD> -- also masked out

        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.fc(out)  # [B, L, vocab_size]
        return logits, None  # None keeps the same (outputs, aux) shape as the LSTM decoder