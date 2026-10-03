import torch
import torch.nn as nn


class SoftAttention(nn.Module):
    """Bahdanau-style additive attention over the encoder's spatial features."""

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: [B, num_pixels, encoder_dim], decoder_hidden: [B, decoder_dim]
        att1 = self.encoder_att(encoder_out)                 # [B, num_pixels, attn_dim]
        att2 = self.decoder_att(decoder_hidden)               # [B, attn_dim]
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # [B, num_pixels]
        alpha = self.softmax(att)                              # attention weights, sum to 1
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [B, encoder_dim]
        return context, alpha


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, decoder_dim, attention_dim, encoder_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = SoftAttention(encoder_dim, decoder_dim, attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def init_hidden_state(self, encoder_out):
        mean_features = encoder_out.mean(dim=1)
        return self.init_h(mean_features), self.init_c(mean_features)

    def forward(self, encoder_out, captions):
        """Teacher forcing: at each step, feed the GROUND-TRUTH previous
        token (captions[:, t]), not the model's own last prediction."""
        batch_size = encoder_out.size(0)
        max_len = captions.size(1) - 1  # predict tokens 1..end, given 0..end-1

        embeddings = self.embedding(captions)               # [B, full_len, embed_dim]
        h, c = self.init_hidden_state(encoder_out)

        outputs = torch.zeros(batch_size, max_len, self.vocab_size, device=encoder_out.device)
        alphas = torch.zeros(batch_size, max_len, encoder_out.size(1), device=encoder_out.device)

        for t in range(max_len):
            context, alpha = self.attention(encoder_out, h)
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            outputs[:, t, :] = self.fc(self.dropout(h))
            alphas[:, t, :] = alpha

        return outputs, alphas