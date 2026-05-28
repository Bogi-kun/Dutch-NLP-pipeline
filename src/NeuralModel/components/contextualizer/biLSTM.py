import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMContextualizer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 2,
            feature_dropout: float = 0.3,
            lstm_dropout: float = 0.5
    ):
        super().__init__()

        # Spatial Dropout: Drops entire feature channels
        self.feature_dropout = nn.Dropout1d(p=feature_dropout)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0
        )

        # Output projection to reach your target dimension
        self.context_projection = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, seq_lengths):
        """
        Args:
            x: [Batch, Seq_Len, Input_Dim]
            seq_lengths: [Batch] - The actual length of each sentence
        """
        # Store original device
        device = x.device

        # 1. Feature Dropout (Permute for Dropout1d logic)
        x = x.transpose(1, 2)
        x = self.feature_dropout(x)
        x = x.transpose(1, 2)

        # 2. Pack the sequence
        # We move lengths to CPU because pack_padded_sequence expects it there.
        # enforce_sorted=False handles batches where sentences aren't in descending order.
        seq_lengths_cpu = seq_lengths.cpu() if seq_lengths.device.type != 'cpu' else seq_lengths

        packed_x = pack_padded_sequence(
            x,
            seq_lengths_cpu,
            batch_first=True,
            enforce_sorted=False
        )

        # 3. Process with LSTM
        packed_out, _ = self.lstm(packed_x)

        # 4. Unpack back to a standard tensor
        # lstm_out shape: [Batch, Seq_Len, hidden_dim * 2]
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Ensure output is on correct device
        lstm_out = lstm_out.to(device)

        # 5. Final Projection
        return self.context_projection(lstm_out)