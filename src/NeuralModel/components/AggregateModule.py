from torch import nn


class DutchAggregateModule(nn.Module):
    def __init__(self,
                 input_layer,
                 context_layer,
                 biaffine_layer,
                 morph_output,
                 ):
        super(DutchAggregateModule, self).__init__()
        self.input_layer = input_layer
        self.context_layer = context_layer
        self.biaffine_layer = biaffine_layer
        self.morph_output = morph_output

    def forward(self, batch):
        """
        Args:
            batch: The batch dictionary directly from DutchDataset.collate_fn
        """
        # 1. Unpack from the batch dictionary
        inputs = batch['input']
        fast_text_tokens = inputs['tokens']
        pos_tokens = inputs['Lexicon_POS']
        lemma_tokens = inputs['Lexicon_Lemma_Tags']
        form_tokens = inputs['Lexicon_Form_Tags']
        seq_lengths = batch['seq_lengths']

        # 2. Input layer: combines features and embeddings and prepends learnable <ROOT> token
        concat_output = self.input_layer(fast_text_tokens, pos_tokens, form_tokens, lemma_tokens)
        # [B, T+1, input_dim]

        # 2.5 adjust sequenceLenghts for injected <ROOT>
        seq_lengths = seq_lengths + 1

        # 3. Contextualize with biLSTM
        lstm_output = self.context_layer(concat_output, seq_lengths)
        # [B, T+1, contextualizer_output_dim]

        # 4. Syntactic Parsing Computations (HEAD and DEP scores)
        # Uses the full sequence including <ROOT>
        arc_scores, rel_scores = self.biaffine_layer(lstm_output, seq_lengths)
        # arc_scores: [B, T+1, T+1]
        # rel_scores: [B, T+1, T+1, num_dep_labels]

        # 6. Morphological + Tag outputs (UPOS, FEATS)
        predictions = self.morph_output(lstm_output)
        # Returns a dict: {'UPOS': [B, T+1, 18], 'Gender': [B, T+1, 5], ...}

        # 7. Add Syntax scores to the predictions dictionary
        predictions['arc_scores'] = arc_scores
        predictions['rel_scores'] = rel_scores

        return predictions

    def remove_root(self, predictions):
        """
        Removes the <ROOT> token from all output tensors in the predictions dictionary.
        This is useful for evaluation, as the <ROOT> token is not part of the original input sentences.
        """
        cleaned_predictions = {}
        for key, tensor in predictions.items():
            if tensor.dim() == 3:  # For arc_scores and rel_scores
                cleaned_predictions[key] = tensor[:, 1:, :]
            elif tensor.dim() == 4:  # For rel_scores
                cleaned_predictions[key] = tensor[:, 1:, :, :]
            else:
                cleaned_predictions[key] = tensor[:, 1:]
        return cleaned_predictions
