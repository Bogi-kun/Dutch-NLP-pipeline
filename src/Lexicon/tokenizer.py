from src.Lexicon.separableVerbs import get_separable_parts

SEPARABLE_PARTICLES = get_separable_parts()


def tokenize(sentence, tokenizer):
    doc = tokenizer(sentence)
    sentence_json = []

    for token in doc:
        sentence_json.append({
            "id": token.i,
            "text": token.text,
            "whitespace": token.whitespace_
        })
    return sentence_json


def tokenize_list(sentence_list):
    sentence_json = []

    for i, token in enumerate(sentence_list):
        sentence_json.append({
            "id": i,
            "text": token,
        })
    return sentence_json


def assign_lemma_candidates(tokens, lexicon):
    tokens = tokens.copy() # we dont want to modify the original tokens
    isVerb = [False for _ in range(len(tokens))]
    for token in tokens:

        lemma_candidates = lexicon.get_candidates(token["text"])  # Try lowercase first, then original case

        for lemma in lemma_candidates:
            if lemma['lemma_pos'] == "VRB":
                isVerb[token["id"]] = True
                break

        token["lemma_candidates"] = lemma_candidates

    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if i != j:
                combined_text = tokens[i]["text"] + " " + tokens[j]["text"]
                if isVerb[i] and tokens[j]["text"].lower() in SEPARABLE_PARTICLES:
                    splitCandidates = lexicon.get_candidates(combined_text)
                    tokens[i]["lemma_candidates"] += splitCandidates
                    tokens[j]["separable_particle"] = True
    return tokens

