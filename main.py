import spacy

from src.Inference.NLP_pipeline import NLP_Pipeline

if __name__ == "__main__":
    # Initialize pipeline (loads from default paths)
    pipeline = NLP_Pipeline("nl_lg")

    while True:
        sentence = input("Enter sentence: ")
        print("=" * 100)

        results = pipeline.process(sentence)


        # Print as table
        print(f"{'index':<15} {'Token':<15} {'Lemma':<15} {'True_Lemma':<15} {'POS':<10} {'DEP':<8} {'REL':<15}")
        print("-" * 100)
        for i, token_dict in enumerate(results):
            print(
                f"{i+1:<15}"
                f"{token_dict['text']:<15} "
                f"{token_dict['lemma']:<15} "
                f"{token_dict['true_lemma']:<15} "
                f"{token_dict['POS']:<10} "
                f"{token_dict['DEP']:<8} "
                f"{token_dict['REL']:<15}"
            )