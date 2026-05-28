import json
from pathlib import Path


def generate_mapping():
    POS_TO_DIMENSION = {}
    LEMMA_TAG_TO_DIMENSION = {}
    FORM_TAG_TO_DIMENSION = {}

    # Read the mapping from the JSON file
    mapping_path = Path(__file__).resolve().parent / "LexiconData" / "lexicon_stats.json"
    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        for POS in data["pos_values"]:
            POS_TO_DIMENSION[POS] = len(POS_TO_DIMENSION)
        for lemma_tag in data["lemma_tag_components"]:
            LEMMA_TAG_TO_DIMENSION[lemma_tag] = len(LEMMA_TAG_TO_DIMENSION)
        for form_tag in data["form_tag_components"]:
            FORM_TAG_TO_DIMENSION[form_tag] = len(FORM_TAG_TO_DIMENSION)

        FORM_TAG_TO_DIMENSION["separable_particle"] = len(FORM_TAG_TO_DIMENSION)

    # Save the mappings to JSON files
    output_file = Path(__file__).parent / "LexiconData" / "mappings.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({
            "POS_TO_DIMENSION": POS_TO_DIMENSION,
            "LEMMA_TAG_TO_DIMENSION": LEMMA_TAG_TO_DIMENSION,
            "FORM_TAG_TO_DIMENSION": FORM_TAG_TO_DIMENSION
        }, f, indent=2, ensure_ascii=False)