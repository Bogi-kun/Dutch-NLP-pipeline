from src.Inference.NLP_pipeline import NLP_Pipeline# Dutch NLP Pipeline
---
A morphosyntactic pipeline optimized for grammatical analysis of Dutch sentences built from scratch. The pipeline includes a multitask neural model trained on LassySmall corpus containing around 65000 sentences (1 million tokens) annotated with POS tags, lemmas, and syntactic dependencies.
## Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/Bogi-kun/Dutch-NLP-pipeline.git
cd Dutch-NLP-pipeline
```
2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Congratulations! You have successfully set up the Dutch NLP pipeline.
```main.py``` file contains an example of how to use the module.


## Interface
To use the NLP_Pipeline, we need to create an instance of the 'NLP_Pipeline' class:
```python
from src.Inference.NLP_pipeline import NLP_Pipeline as nlp

# defaultly loads the 'nl_md' model
pipeline = nlp()

# or define the model to be used ["nl_sm", "nl_md", "nl_lg"]
pipeline = nlp("nl_lg")

doc = pipeline.process("Dit is een Nederlandse zin.")
```




