from pathlib import Path
import json
import sys

import torch
import numpy as np
import compress_fasttext
from spacy.lang.nl import Dutch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.Lexicon.BuildDictionary import GiGaNT_Lexicon
from src.Lexicon.feature_vectors import create_pos_list, create_form_tag_list, create_lemma_tag_list
from src.Lexicon.tokenizer import tokenize, assign_lemma_candidates, tokenize_list
from src.NeuralModel.model_utils.create_model import create_model
from src.NeuralModel.components.ChuLiuEdmonds import ChuLiuEdmonds


class NLP_Pipeline:
    """
    Complete Dutch NLP pipeline for morphosyntactic parsing.
    Takes sentences and returns token-level predictions (POS, DEP, REL).
    """

    def __init__(self, model_type: str = None):
        """
        Initialize the NLP pipeline.

        Args:
            model_path: Path to production model.pt file (default: NeuralModel/prod_assets/model.pt)
        """
        # Default path - load from prod_assets
        if model_type is None:
            model_path = Path(__file__).parent.parent.parent / "Dataset" / "prod_assets" / "medium_model.pt"
        elif model_type == "nl_lg":
            model_path = Path(__file__).parent.parent.parent / "Dataset" / "prod_assets" / "big_model.pt"
        elif model_type == "nl_sm":
            model_path = Path(__file__).parent.parent.parent / "Dataset" / "prod_assets" / "small_model.pt"
        elif model_type == "nl_md":
            model_path = Path(__file__).parent.parent.parent / "Dataset" / "prod_assets" / "medium_model.pt"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model_path = Path(model_path)

        # initialize mappings
        self.UD_LEXICON_MAPPING = {
            "NOUN": "NOU-C",  # Common Noun
            "PROPN": "NOU-P",  # Proper Noun
            "PUNCT": "PUNCT",
            "NUM": "NUM",
            "SYM": "RES",  # Symbols usually fall under Residual
            "ADP": "ADP",
            "DET": "PD",  # Determiners/Pronoun-Determiners
            "VERB": "VRB",
            "ADJ": "AA",  # Adjectival
            "ADV": "ADV",
            "AUX": "VRB",  # Auxiliaries are verbs
            "CCONJ": "CONJ",
            "SCONJ": "CONJ",
            "PRON": "PD",  # Mapping Pronouns to PD (common in Dutch tagging)
            "INTJ": "INT",
            "X": "RES",  # Unknown/Foreign
            "_": "RES"
        }

        # Initialize components
        self.tokenizer = Dutch()
        self.lexicon = GiGaNT_Lexicon()

        # Load mappings
        self._load_mappings()

        # Load FastText
        self._load_fasttext()

        # Load model
        self.model = self._load_model()

        # Initialize Chu-Liu-Edmonds algorithm for dependency parsing
        self.cle_decoder = ChuLiuEdmonds(use_cuda=torch.cuda.is_available())

    def _load_model(self):
        """Load neural model from production package (config + weights)."""
        print(f"Loading production model from: {self.model_path}")

        # Load the unified production package
        production_package = torch.load(str(self.model_path), map_location='cpu')

        # Extract config and state_dict from package
        model_config = production_package['config']
        state_dict = production_package['state_dict']

        # Create model from config
        model = create_model(model_config)

        # Load state dict
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model


    def _load_mappings(self):
        """Load UD and lexicon mappings."""
        mappings_path = Path(__file__).parent.parent.parent / "Dataset" / "mappings"

        self.mappings = {
            'upos_map': json.load(open(mappings_path / "upos_mapping.json")),
            'dep_map': json.load(open(mappings_path / "dep_mapping.json")),
        }

        # Create reverse mappings
        self.reverse_upos = {v: k for k, v in self.mappings['upos_map'].items()}
        self.reverse_dep = {v: k for k, v in self.mappings['dep_map'].items()}

    def _load_fasttext(self):
        """Load FastText embeddings."""
        fasttext_path = Path(__file__).parent.parent.parent / "Dataset" / "dutch_compact.bin"
        self.ft_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(str(fasttext_path))


    def _get_fasttext_vectors(self, tokens):
        """Get FastText vectors for tokens."""
        vectors = []
        for token in tokens:
            try:
                vec = self.ft_model.get_vector(token)
            except:
                vec = np.zeros(self.ft_model.vector_size, dtype=np.float32)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def _get_lexicon_features(self, tokens):
        """Get POS, LEMMA, FORM candidates from lexicon."""
        tokenized = tokenize_list(tokens)
        tokenized = assign_lemma_candidates(tokenized, self.lexicon)

        pos_list = [create_pos_list(t) for t in tokenized]
        lemma_list = [create_lemma_tag_list(t) for t in tokenized]
        form_list = [create_form_tag_list(t) for t in tokenized]

        return pos_list, lemma_list, form_list

    def _pad_candidates(self, candidates, pad_value=-1):
        """Pad candidate lists to max length."""
        max_cands = max(len(c) for c in candidates) if candidates else 1
        padded = []
        for c in candidates:
            padded_c = c + [pad_value] * (max_cands - len(c))
            padded.append(padded_c)
        return torch.tensor(padded, dtype=torch.long)

    def _run_inference(self, sentence_tokens):
        """Run inference on a sentence."""
        # Get FastText vectors
        fasttext_vectors = self._get_fasttext_vectors(sentence_tokens)
        fasttext_tensor = torch.from_numpy(fasttext_vectors).unsqueeze(0).float()

        # Get lexicon features
        pos_cands, lemma_cands, form_cands = self._get_lexicon_features(sentence_tokens)
        pos_tensor = self._pad_candidates(pos_cands).unsqueeze(0)
        lemma_tensor = self._pad_candidates(lemma_cands).unsqueeze(0)
        form_tensor = self._pad_candidates(form_cands).unsqueeze(0)

        # Create batch
        batch = {
            'sent_ids': ['inference'],
            'input': {
                'tokens': fasttext_tensor,
                'Lexicon_POS': pos_tensor,
                'Lexicon_Lemma_Tags': lemma_tensor,
                'Lexicon_Form_Tags': form_tensor,
            },
            'seq_lengths': torch.tensor([len(sentence_tokens)]),
            'output': {
                'UPOS': torch.zeros(1, len(sentence_tokens), dtype=torch.long),
                'DEP': torch.zeros(1, len(sentence_tokens), dtype=torch.long),
                'Head': torch.zeros(1, len(sentence_tokens), dtype=torch.long),
                'FEATS': {},
            }
        }

        # Run inference
        with torch.no_grad():
            predictions = self.model(batch)

        return predictions

    def _decode_predictions(self, predictions, sentence_tokens, whitespaces):
        """
        Decode model predictions to token dictionaries.

        Uses Chu-Liu-Edmonds algorithm to find optimal non-projective parse tree.

        Returns:
            List of dicts with keys: text, whitespace, POS, DEP, REL
        """
        upos_logits = predictions['UPOS'][0]
        upos_ids = torch.argmax(upos_logits, dim=1).cpu().numpy()

        arc_scores = predictions['arc_scores']  # [batch=1, seq_len, seq_len]
        rel_scores = predictions['rel_scores']  # [batch=1, seq_len, seq_len, num_rels]

        # Get UPOS predictions
        upos_tags = [self.reverse_upos.get(int(id), '<UNK>') for id in upos_ids]

        # ===== Use Chu-Liu-Edmonds to find optimal parse tree =====
        heads_optimal = self.cle_decoder(arc_scores).squeeze(0).cpu().numpy()  # [seq_len]

        # Decode dependency relations using predicted heads
        rels = []
        for i, head_idx in enumerate(heads_optimal):
            head_idx = int(head_idx)
            rel_logits_for_head = rel_scores[0, i, head_idx, :]  # [num_rels]
            rel_id = torch.argmax(rel_logits_for_head).cpu().item()
            rel_tag = self.reverse_dep.get(int(rel_id), '<UNK>')
            rels.append(rel_tag)

        # Format output
        result = []
        for i, token_text in enumerate(sentence_tokens):
            result.append({
                'text': token_text,
                'whitespace': whitespaces[i] if i < len(whitespaces) else '',
                'POS': upos_tags[i+1],
                'DEP': int(heads_optimal[i+1]),
                'REL': rels[i+1]
            })


        # ADD LEMMAS for words (first naive pass - just take first candidate from lexicon)
        for token in result:
            lemma_candidates = self.lexicon.get_candidates_by_POS(
                                        token['text'],
                                        self.UD_LEXICON_MAPPING[token["POS"]]
            )
            if lemma_candidates:
                token['true_lemma'] = lemma_candidates[0]['lemma_text']  # Take the first candidate as the lemma
                token['lemma'] = lemma_candidates[0]['lemma_text']
            else:
                token['true_lemma'] = token['text'].lower()  # Fallback to lowercase token as lemma
                token['lemma'] = token['text'].lower()

        # second pass to handle separable verbs
        for token in result:
            if token["REL"] == "compound:prt":  # This token is a separable particle
                head_idx = token["DEP"]
                if head_idx > 0 and head_idx < len(result):
                    head_token = result[head_idx - 1]  # Get the head token (adjusting for 1-based indexing)
                    if head_token["POS"] == "VERB":  # Check if the head is a verb
                        # Combine lemma of verb with particle
                        lookup_text = head_token['text'] + " " + token['text'].lower()
                        lemma_candidates = self.lexicon.get_candidates_by_POS(
                            lookup_text,
                            self.UD_LEXICON_MAPPING["VERB"]
                        )
                        if lemma_candidates:
                            head_token['lemma'] = lemma_candidates[0]['lemma_text']  # Update verb lemma with combined form
                        else:
                            head_token['lemma'] = head_token['lemma']  # Keep original lemma if no combined form found
        return result


    def process(self, sentence: str | list[str]):
        """
        Process a sentence through the NLP pipeline.

        Args:
            sentence: Input sentence string

        Returns:
            List of token dictionaries with format:
            {
                'text': str,
                'whitespace': str,
                'POS': str,
                'DEP': int,
                'REL': int
            }
        """
        if not sentence or sentence == '' or type(sentence) not in [str, list[str]]:
            raise ValueError("Input sentence cannot be empty.")
        try:
            # Tokenize
            if type(sentence) == str:
                tokenized = tokenize(sentence, self.tokenizer)
            elif type(sentence) == list:
                tokenized = tokenize_list(sentence)
            sentence_tokens = [token['text'] for token in tokenized]
            whitespaces = [token.get('whitespace', '') for token in tokenized]

            # Run inference
            predictions = self._run_inference(sentence_tokens)

            # Decode and format
            results = self._decode_predictions(predictions, sentence_tokens, whitespaces)

            return results

        except Exception as e:
            raise RuntimeError(f"Error processing sentence: {e}")