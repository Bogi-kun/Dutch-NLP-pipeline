import torch
from torch import nn
from typing import Dict


class TaskHead(nn.Module):
    """Explicit output head for a single classification task."""

    def __init__(self, input_dim: int, num_classes: int, task_name: str, dropout: float = 0.3):
        super().__init__()
        self.task_name = task_name
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, SeqLen, input_dim]
        Returns:
            logits: [Batch, SeqLen, num_classes]
        """
        x = self.dropout(x)
        return self.linear(x)


class MorphOutput(nn.Module):
    """Container for calculating multiple morphological heads (UPOS + FEATS)."""

    def __init__(self, input_dim: int, output_configs: Dict[str, int], dropout: float = 0.3):
        """
        Args:
            input_dim: Dimension from the contextualizer (biLSTM output dim)
            output_configs: Dictionary mapping task names to their number of classes.
        """
        super().__init__()
        self.task_names = list(output_configs.keys())

        # Dynamically create and register a task head for each provided configuration
        for task_name, num_classes in output_configs.items():
            head = TaskHead(input_dim, num_classes, task_name, dropout=dropout)
            # Register using setattr so PyTorch tracks the parameters correctly
            setattr(self, f'head_{task_name}', head)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [Batch, SeqLen, input_dim] where SeqLen includes ROOT token at position 0

        Returns:
            outputs: Dict mapping task_name -> logits tensor of shape [Batch, SeqLen, num_classes]
                     Position 0 will have logits corresponding to ROOT (ignored in loss via -100)
        """
        outputs = {}
        for task_name in self.task_names:
            head_module = getattr(self, f'head_{task_name}')
            outputs[task_name] = head_module(x)

        return outputs
