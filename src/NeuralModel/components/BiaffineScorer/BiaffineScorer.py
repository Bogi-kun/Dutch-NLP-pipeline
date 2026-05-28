import torch
from torch import nn


class MLP(nn.Module):
    """Simple multi-layer perceptron for representation splitting."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(x)))


class BiaffineAttention(nn.Module):
    """
    Standard Biaffine Attention layer.
    Calculates: x^T U y + b
    """

    def __init__(self, input_dim: int, num_classes: int = 1, bias_x: bool = True, bias_y: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.bias_x = bias_x
        self.bias_y = bias_y

        # Matrix U shape: [in_dim + bias, num_classes, in_dim + bias]
        weight_shape = (input_dim + int(bias_x), num_classes, input_dim + int(bias_y))
        self.U = nn.Parameter(torch.empty(weight_shape))

        # Initialize orthogonally (best practice for biaffine layers)
        nn.init.orthogonal_(self.U)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Dependent representations [B, N, D]
            y: Head representations [B, M, D]
        Returns:
            scores: [B, N, M] if num_classes=1 else [B, N, M, num_classes]
        """
        # Append bias ones to inputs if requested
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

        # x is [B, N, Dx], U is [Dx, C, Dy], y is [B, M, Dy]
        # We want output [B, N, C, M]
        # Using PyTorch's einsum for highly optimized tensor contraction
        # b = batch, n = dep length, d = in_dim_x
        # c = num_classes, o = in_dim_y, m = head length
        scores = torch.einsum('bnd, dco, bmo -> bncm', x, self.U, y)

        if self.num_classes == 1:
            # Reshape [B, N, 1, M] -> [B, N, M]
            return scores.squeeze(2)
        else:
            # Reshape [B, N, C, M] -> [B, N, M, C] which is standard for labels
            return scores.permute(0, 1, 3, 2)


class BiaffineScorer(nn.Module):
    """
    Computes Arc (head) and Rel (label) scores from the contextualizer output.
    Includes masking for:
    - Padding (beyond seq_length)
    - Diagonal (self-loops: word cannot be its own head)
    """

    def __init__(self,
                 context_dim: int = 512,
                 arc_hidden_dim: int = 500,
                 rel_hidden_dim: int = 100,
                 num_dep_labels: int = 42,
                 dropout: float = 0.3):
        """
        Args:
            context_dim: Output dimension of your biLSTM
            arc_hidden_dim: Dimension for arc MLPs (usually larger, e.g., 500)
            rel_hidden_dim: Dimension for rel MLPs (usually smaller, e.g., 100)
            num_dep_labels: Number of dependency relations (e.g., 42)
        """
        super().__init__()

        # MLPs for Arc (Head) Prediction
        self.mlp_arc_dep = MLP(context_dim, arc_hidden_dim, dropout)
        self.mlp_arc_head = MLP(context_dim, arc_hidden_dim, dropout)

        # MLPs for Rel (Label) Prediction
        self.mlp_rel_dep = MLP(context_dim, rel_hidden_dim, dropout)
        self.mlp_rel_head = MLP(context_dim, rel_hidden_dim, dropout)

        # Biaffine Layers
        self.arc_biaffine = BiaffineAttention(arc_hidden_dim, num_classes=1)
        self.rel_biaffine = BiaffineAttention(rel_hidden_dim, num_classes=num_dep_labels)

    def forward(self, lstm_output: torch.Tensor, seq_lengths: torch.Tensor):
        """
        Args:
            lstm_output: Contextualized representations [B, T+1, context_dim]
                         (Sequence includes the <ROOT> token)
            seq_lengths: [B] - Actual length of each sentence (including ROOT)
        Returns:
            arc_scores: [B, T+1, T+1]
                        Scores representing likelihood of j being head of i.
                        Padded and diagonal positions are masked with -inf
            rel_scores: [B, T+1, T+1, num_dep_labels]
                        Scores representing likelihood of relation type IF j is head of i.
                        Padded and diagonal positions are masked with -inf
        """
        # 1. Split representations for Arc
        arc_dep = self.mlp_arc_dep(lstm_output)
        arc_head = self.mlp_arc_head(lstm_output)

        # 2. Split representations for Rel
        rel_dep = self.mlp_rel_dep(lstm_output)
        rel_head = self.mlp_rel_head(lstm_output)

        # 3. Compute Arc scores [B, SeqLen, SeqLen]
        # arc_scores[b, i, j] = score of word j being the head of word i
        arc_scores = self.arc_biaffine(arc_dep, arc_head)

        # 4. Compute Rel scores [B, SeqLen, SeqLen, num_classes]
        # rel_scores[b, i, j, c] = score of label c assuming word j is head of word i
        rel_scores = self.rel_biaffine(rel_dep, rel_head)

        # 5. Create mask for valid positions based on seq_lengths
        # seq_lengths: [B] -> we need to mask positions beyond the actual sequence length
        batch_size, max_len = lstm_output.shape[:2]
        device = lstm_output.device

        # ===== PADDING MASK =====
        # Create a position mask: [B, T+1]
        # True for valid positions (within seq_length), False for padding
        pos_ids = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)  # [1, T+1]
        seq_mask = pos_ids < seq_lengths.unsqueeze(1)  # [B, T+1]

        # Create 2D mask for valid positions: [B, T+1, T+1]
        # Both the dependent (i) and head (j) positions must be valid
        padding_mask = seq_mask.unsqueeze(2) & seq_mask.unsqueeze(1)  # [B, T+1, T+1]

        # ===== DIAGONAL MASK =====
        # Create diagonal mask: [T+1, T+1]
        # True everywhere EXCEPT on the diagonal (no self-loops)
        diagonal_mask = ~torch.eye(max_len, device=device, dtype=torch.bool).unsqueeze(0)  # [1, T+1, T+1]

        # ===== COMBINE MASKS =====
        # Final mask: both padding AND no self-loops
        combined_mask = padding_mask & diagonal_mask  # [B, T+1, T+1]

        # Apply combined mask to arc_scores
        arc_scores = arc_scores.masked_fill(~combined_mask, float('-inf'))

        # Apply combined mask to rel_scores
        rel_mask = combined_mask.unsqueeze(3)  # [B, T+1, T+1, 1]
        rel_scores = rel_scores.masked_fill(~rel_mask, float('-inf'))

        return arc_scores, rel_scores
