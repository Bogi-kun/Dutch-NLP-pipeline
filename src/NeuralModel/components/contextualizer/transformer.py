import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sin/cos positional encoding - expandable to any length"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to embeddings"""
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0)  # [1, T, d_model]
        return x + pe


class TransformerContextualizer(nn.Module):
    """
    Transformer-based contextualizer using PyTorch's built-in TransformerEncoder.

    Key features:
    - Expandable positional encoding (works for any sequence length)
    - Attention masks to ignore padding
    - Fully parallelizable
    """

    def __init__(
            self,
            input_dim: int,
            d_model: int = 512,
            num_layers: int = 4,
            num_heads: int = 4,
            d_ff: int = 1024,
            dropout: float = 0.1,
            output_dim: int = None
    ):
        super().__init__()

        self.d_model = d_model
        self.output_dim = output_dim or d_model

        # Project input to transformer dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # PyTorch's built-in TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True  # [B, T, d_model] instead of [T, B, d_model]
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False  # disable prototype nested tensor fastpath
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def _create_attention_mask(self, seq_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create attention mask for padding.

        PyTorch convention:
        - True = MASK OUT (don't attend)
        - False = ATTEND

        Args:
            seq_lengths: [B] - actual length of each sequence
            max_len: Maximum length in batch

        Returns:
            src_key_padding_mask: [B, T] where True = padding
        """
        batch_size = seq_lengths.shape[0]
        device = seq_lengths.device

        # Create position indices
        pos = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # True where padding (pos >= seq_length)
        mask = pos >= seq_lengths.unsqueeze(1)  # [B, T]

        return mask

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim] - input embeddings
            seq_lengths: [B] - actual length of each sequence

        Returns:
            output: [B, T, output_dim] - contextualized representations
        """
        B, T, _ = x.shape

        # 1. Project to transformer dimension
        x = self.input_projection(x)  # [B, T, d_model]

        # 2. Add positional encoding (works for any length!)
        x = self.pos_encoding(x)  # [B, T, d_model]
        x = self.dropout(x)

        # 3. Create padding mask
        # True = mask out, False = attend
        src_key_padding_mask = self._create_attention_mask(seq_lengths, T)  # [B, T]

        # 4. Pass through transformer
        x = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # [B, T, d_model]

        # 5. Project to output dimension
        output = self.output_projection(x)  # [B, T, output_dim]

        return output