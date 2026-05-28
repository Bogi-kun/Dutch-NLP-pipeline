from torch import nn


class HybridContextualizer(nn.Module):

    def __init__(self, biLSTM, transformer):
        super().__init__()
        self.biLSTM = biLSTM
        self.transformer = transformer

    def forward(self, x, seq_lengths):
        lstm_out = self.biLSTM(x, seq_lengths)
        transformer_out = lstm_out + self.transformer(lstm_out, seq_lengths)
        return transformer_out