from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class DutchDataset(Dataset):


    def __init__(self,
                 data
                 ):

        self.data = data
        self.sentences = torch.load(data)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    @staticmethod
    def collate_fn(batch, input_padding_value=-1, target_padding_value=-100):

        def pad_nested_lists(nested_batch, pad_value=0):
            batch_size = len(nested_batch)
            max_seq_len = max(len(sent) for sent in nested_batch)
            max_cands = max((len(cands) for sent in nested_batch for cands in sent), default=0)

            out = torch.full((batch_size, max_seq_len, max_cands), pad_value, dtype=torch.long)

            for i, sent in enumerate(nested_batch):
                for j, cands in enumerate(sent):
                    if len(cands) > 0:
                        out[i, j, :len(cands)] = torch.tensor(cands, dtype=torch.long)

            return out
    
        seq_lengths = torch.tensor([len(s['input']['tokens']) for s in batch], dtype=torch.long)

        tokens = pad_sequence([s['input']['tokens'].long() for s in batch],
                              batch_first=True, padding_value=input_padding_value)

        upos = pad_sequence([s['output']['UPOS'].long() for s in batch],
                            batch_first=True, padding_value=target_padding_value)

        dep = pad_sequence([s['output']['DEP'].long() for s in batch],
                           batch_first=True, padding_value=target_padding_value)

        head = pad_sequence([s['output']['Head'].long() for s in batch],
                            batch_first=True, padding_value=target_padding_value)

        feat_keys = sorted(set().union(*(s['output']['FEATS'].keys() for s in batch)))
        feats_dict = {
            k: pad_sequence([s['output']['FEATS'][k].long() for s in batch],
                            batch_first=True, padding_value=target_padding_value)
            for k in feat_keys
        }

        lex_pos = pad_nested_lists([s['input']['Lexicon_POS'] for s in batch], input_padding_value)
        lex_lemma = pad_nested_lists([s['input']['Lexicon_Lemma_Tags'] for s in batch], input_padding_value)
        lex_form = pad_nested_lists([s['input']['Lexicon_Form_Tags'] for s in batch], input_padding_value)

        return {
            'sent_ids': [s['sent_id'] for s in batch],
            'input': {
                'tokens': tokens,
                'Lexicon_POS': lex_pos,
                'Lexicon_Lemma_Tags': lex_lemma,
                'Lexicon_Form_Tags': lex_form,
            },
            'output': {
                'UPOS': upos,
                'FEATS': feats_dict,
                'DEP': dep,
                'Head': head,
            },
            'seq_lengths': seq_lengths,
        }


