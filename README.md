# Dutch NLP Pipeline

A morphosyntactic pipeline optimized for grammatical analysis of Dutch sentences, built from scratch. The pipeline features a multitask neural model trained on the LassySmall corpus (~65,000 sentences, 1 million tokens) annotated with POS tags, lemmas, and syntactic dependencies.

## Setup Instructions

1. Clone the repository:
```bash
git clone [https://github.com/Bogi-kun/Dutch-NLP-pipeline.git](https://github.com/Bogi-kun/Dutch-NLP-pipeline.git)
cd Dutch-NLP-pipeline
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

A complete execution example can be found in the root `main.py` file.

## Interface

Initialize the pipeline by instantiating the `NLP_Pipeline` class. You can optionally specify the model size (`"nl_sm"`, `"nl_md"`, or `"nl_lg"`).

```python
from src.Inference.NLP_pipeline import NLP_Pipeline

# Loads the 'nl_md' model by default
pipeline = NLP_Pipeline()

# Or explicitly define the model size
# pipeline = NLP_Pipeline("nl_lg")

# Accepts a string or a list of pre-tokenized strings
doc = pipeline.process("Ik ruim mijn kamer op.")
```

### Output Format

The `.process()` method returns a list of dictionaries structured as follows:

```python
{
  "text": str,          # Raw token text
  "whitespace": str,    # Trailing whitespace attached to the token
  "POS": str,           # Part-of-Speech tag (Universal Dependencies)
  "DEP": int,           # Index of the parent token (0 for root)
  "REL": str,           # Dependency relationship type
  "lemma": str,         # Base form including separable verb particles (e.g., "opruimen")
  "true_lemma": str,    # Standard dictionary base form (e.g., "ruimen")
}
```

### Visualizing Results

Printing the processed document displays a structured dependency table highlighting the distinction between standard lemmas and particle-resolved lemmas:

```
====================================================================================================
index           Token           Lemma           True_Lemma      POS        DEP      REL            
----------------------------------------------------------------------------------------------------
1               Ik              ik              ik              PRON       2        nsubj          
2               ruim            opruimen        ruimen          VERB       0        root           
3               mijn            mijn            mijn            PRON       4        nmod:poss      
4               kamer           kamer           kamer           NOUN       2        obj            
5               op              op              op              ADP        2        compound:prt   
6               .               .               .               PUNCT      2        punct          
====================================================================================================
```
