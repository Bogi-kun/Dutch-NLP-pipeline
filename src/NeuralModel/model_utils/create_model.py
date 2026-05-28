from src.NeuralModel.components.AggregateModule import DutchAggregateModule
from src.NeuralModel.components.BiaffineScorer.BiaffineScorer import BiaffineScorer
from src.NeuralModel.components.contextualizer.biLSTM import BiLSTMContextualizer
from src.NeuralModel.components.contextualizer.hybridContextualizer import HybridContextualizer
from src.NeuralModel.components.contextualizer.transformer import TransformerContextualizer
from src.NeuralModel.components.inputLayer.ConcatInput import ConcatInput
from src.NeuralModel.components.outputLayer.morphOutput import MorphOutput

def create_morph_config(output_mappings, include_feats=False):
    morph_config = {
        "UPOS": len(output_mappings["UPOS"])
    }

    if include_feats:
        for k,v in output_mappings["FEATS"].items():
            morph_config[k] = len(v)

    return morph_config

def create_model(model_config):
    input_config = model_config["input_config"]
    context_config = model_config["context_config"]
    morph_output_config = model_config["morph_output_config"]
    biaffine_config = model_config["biaffine_config"]


    if input_config["type"] == "concat":
        input_layer = ConcatInput(
            fast_text_dim=input_config["fast_text_dim"],
            pos_dim=input_config["pos_dim"],
            form_tag_dim=input_config["form_tag_dim"],
            lemma_tag_dim=input_config["lemma_tag_dim"],
            output_dim=input_config["output_dim"],
            dropout=input_config["dropout"]
        )
    else:
        raise ValueError(f"Unsupported input layer type: {input_config['type']}")

    if context_config["type"] == "lstm":
        context_layer = BiLSTMContextualizer(
            input_dim=context_config["input_dim"],
            hidden_dim=context_config["hidden_dim"],
            output_dim=context_config["output_dim"],
            num_layers=context_config["num_layers"],
            feature_dropout=context_config["feature_dropout"],
            lstm_dropout=context_config["lstm_dropout"]
        )
    elif context_config["type"] == "transformer":
        context_layer = TransformerContextualizer(
            input_dim=context_config["input_dim"],
            d_model=context_config["hidden_dim"],
            output_dim=context_config["output_dim"],
            num_layers=context_config["num_layers"],
            dropout=context_config["feature_dropout"],
        )
    elif context_config["type"] == "hybrid":
        lstm_config = context_config["lstm_config"]
        transformer_config = context_config["transformer_config"]
        lstm = BiLSTMContextualizer(
            input_dim=lstm_config["input_dim"],
            hidden_dim=lstm_config["hidden_dim"],
            output_dim=lstm_config["output_dim"],
            num_layers=lstm_config["num_layers"],
            feature_dropout=lstm_config["feature_dropout"],
            lstm_dropout=lstm_config["lstm_dropout"]
        )
        transformer = TransformerContextualizer(
            input_dim=transformer_config["input_dim"],
            d_model=transformer_config["hidden_dim"],
            output_dim=transformer_config["output_dim"],
            num_layers=transformer_config["num_layers"],
            dropout=transformer_config["feature_dropout"],
        )
        context_layer = HybridContextualizer(lstm, transformer)
    else:
        raise ValueError(f"Unsupported context layer type: {context_config['type']}")


    morph_layer = MorphOutput(
        input_dim=morph_output_config["input_dim"],
        output_configs=morph_output_config["output_configs"],
        dropout=morph_output_config["dropout"]
    )

    biaffine_layer = BiaffineScorer(
        context_dim=biaffine_config["context_dim"],
        arc_hidden_dim=biaffine_config["arc_hidden_dim"],
        rel_hidden_dim=biaffine_config["rel_hidden_dim"],
        num_dep_labels=biaffine_config["num_dep_labels"],
        dropout=biaffine_config["dropout"]
    )

    aggregate_model = DutchAggregateModule(
        input_layer=input_layer,
        context_layer=context_layer,
        morph_output=morph_layer,
        biaffine_layer=biaffine_layer
    )

    return aggregate_model