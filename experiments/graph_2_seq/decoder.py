"""
decoder.py
──────────
Improved Transformer-based sequence decoder.

Changes over the original
─────────────────────────
1. Embedding scale (sqrt(hidden_dim)) restored — standard in Transformers,
   prevents embeddings from being dwarfed by positional information.
2. Sinusoidal positional encoding added — gives the decoder an explicit sense
   of position in the generated sequence, which the original lacked entirely.
3. Deeper decoder: 6 layers instead of 4.
4. Wider feedforward dim: 512 → 1024.
5. Dropout 0.1 for regularisation (helps with 53k samples).
6. Causal mask is generated inside forward() so the caller does not need to
   manage it — reduces chance of the mask being forgotten or wrong.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al. 2017).
    Adds position information to token embeddings so the decoder knows
    where in the sequence it currently is.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1,
                 max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, hidden_dim, 2).float()
            * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Register as buffer so it moves with .to(device) but is not a param
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, H]

    def forward(self, x):
        # x: [B, T, H]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class G2SDecoder(nn.Module):
    """
    Parameters
    ──────────
    vocab_size : int - number of tokens
    hidden_dim : int - model width; must match encoder hidden_dim
    num_layers : int - number of TransformerDecoder layers (default 6)
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_enc   = PositionalEncoding(hidden_dim, dropout=0.1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=1024,   # wider FFN
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, target_seq, encoder_context,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        Parameters
        ──────────
        target_seq             : LongTensor  [B, T]
        encoder_context        : FloatTensor [B, N, hidden_dim]
        tgt_key_padding_mask   : BoolTensor  [B, T]   True = ignore (PAD)
        memory_key_padding_mask: BoolTensor  [B, N]   True = ignore (PAD)

        Returns
        ───────
        logits : FloatTensor [B, T, vocab_size]
        """
        T = target_seq.size(1)

        # Causal mask — generated here so callers never forget it
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=target_seq.device
        )  # [T, T]

        # Embed + scale + positional encoding
        tgt_emb = self.embedding(target_seq) * math.sqrt(self.hidden_dim)
        tgt_emb = self.pos_enc(tgt_emb)       # [B, T, H]

        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=encoder_context,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, T, H]

        return self.out(output)  # [B, T, vocab_size]