from pathlib import Path
from typing import Self
import torch
from torch import nn


class ConcatInput(nn.Module):

    def __init__(self,
                 fast_text_dim=0,
                 pos_dim=0,
                 form_tag_dim=0,
                 lemma_tag_dim=0,
                 output_dim=256,
                 dropout=0.3  # <-- Added dropout parameter
                 ):

        super(ConcatInput, self).__init__()

        self.fastText_dim = fast_text_dim
        self.pos_dim = pos_dim
        self.form_tag_dim = form_tag_dim
        self.lemma_tag_dim = lemma_tag_dim
        self.output_dim = output_dim

        # Keep embeddings on CPU to avoid CUDA memory issues with large embedding tensors
        # Embeddings are loaded on-demand and not registered as a buffer.
        # This prevents PyTorch from trying to move the large embedding tensor to GPU,
        # which can cause CUDA illegal memory access errors during model.to(device) calls.
        # Indexing operations happen on CPU and results are moved to the correct device.
        fast_text_path = Path(__file__).parents[3] / "Dataset" / "training_data_binaries" / "embedding_cache_tensor.pt"
        self.embeddings_path = fast_text_path
        self._embeddings_cache = None

        self.total_input_dim = self.fastText_dim + self.pos_dim + self.form_tag_dim + self.lemma_tag_dim

        if self.total_input_dim == 0:
            raise ValueError("Total input dimension must be greater than 0.")

        # Define the learnable ROOT embedding
        self.root_embedding = nn.Parameter(torch.randn(1, 1, self.total_input_dim))

        # <-- Added Dropouts -->
        self.input_dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(in_features=self.total_input_dim, out_features=self.output_dim)

        self.output_dropout = nn.Dropout(dropout)

    @property
    def embeddings(self):
        """Lazy-load embeddings on CPU (not a model buffer, stays on CPU for indexing)."""
        if self._embeddings_cache is None:
            self._embeddings_cache = torch.load(self.embeddings_path, map_location='cpu')
        return self._embeddings_cache

    def _create_multihot(self, indices, vocab_size):
        mask = (indices >= 0)
        safe_indices = indices.clamp(min=0)
        B, T, C = indices.shape
        multihot = torch.zeros(B, T, vocab_size, device=indices.device)
        multihot.scatter_(2, safe_indices, mask.float())
        return multihot

    def forward(self, fast_text_tokens, pos_tokens, form_tokens, lemma_tokens):

        # 1. Process your word features normally
        # Move tokens to CPU for embedding indexing (embeddings always stay on CPU)
        device = fast_text_tokens.device

        if fast_text_tokens.dtype == torch.long:
            # Training: token IDs → lookup embeddings
            fast_text_tokens_cpu = fast_text_tokens.cpu()
            fast_text_embeddings = self.embeddings[fast_text_tokens_cpu]
            fast_text_embeddings = fast_text_embeddings.to(device)
        else:
            # Inference: already vectors → use directly
            fast_text_embeddings = fast_text_tokens

        pos_multihot = self._create_multihot(pos_tokens, self.pos_dim) if self.pos_dim > 0 else None
        form_multihot = self._create_multihot(form_tokens, self.form_tag_dim) if self.form_tag_dim > 0 else None
        lemma_multihot = self._create_multihot(lemma_tokens, self.lemma_tag_dim) if self.lemma_tag_dim > 0 else None

        inputs = [fast_text_embeddings]
        if pos_multihot is not None: inputs.append(pos_multihot)
        if form_multihot is not None: inputs.append(form_multihot)
        if lemma_multihot is not None: inputs.append(lemma_multihot)

        # word_vectors: [B, T, total_input_dim]
        word_vectors = torch.cat(inputs, dim=-1)

        # 2. INJECT THE ROOT
        batch_size = word_vectors.size(0)
        roots = self.root_embedding.expand(batch_size, -1, -1).to(word_vectors.device)
        concatenated = torch.cat([roots, word_vectors], dim=1)

        # 3. Apply Dropout -> Project -> Apply Dropout again
        concatenated = self.input_dropout(concatenated)  # Regularize the raw concatenated features
        output = self.linear(concatenated)  # [B, T+1, output_dim]
        output = self.output_dropout(output)  # Regularize the projected features

        return output
