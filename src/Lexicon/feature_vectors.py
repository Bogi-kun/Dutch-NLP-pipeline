import json
from pathlib import Path

import numpy as np

MAPPING_PATH = Path(__file__).parent / "LexiconData" / "mappings.json"
with MAPPING_PATH.open("r", encoding="utf-8") as f:
    MAPPINGS = json.load(f)

POS_MAPPING = MAPPINGS["POS_TO_DIMENSION"]
LEMMA_TAG_MAPPING = MAPPINGS["LEMMA_TAG_TO_DIMENSION"]
FORM_TAG_MAPPING = MAPPINGS["FORM_TAG_TO_DIMENSION"]


def create_pos_list(token):
    pos_list = set()
    for lemma_candidate in token["lemma_candidates"]:
        pos = lemma_candidate["lemma_pos"]
        if pos in POS_MAPPING:
            pos_list.add(POS_MAPPING[pos])
    return list(pos_list)


def create_lemma_tag_list(token):
    lemma_tag_list = set()
    for lemma_candidate in token["lemma_candidates"]:
        if lemma_candidate["lemma_tags"]:
            tags = lemma_candidate["lemma_tags"].split(",")
        else:
            tags = []
        for tag in tags:
            if tag in LEMMA_TAG_MAPPING:
                lemma_tag_list.add(LEMMA_TAG_MAPPING[tag])
    return list(lemma_tag_list)


def create_form_tag_list(token):
    form_tag_list = set()
    for lemma_candidate in token["lemma_candidates"]:
        if lemma_candidate["form_tags"]:
            tags = lemma_candidate["form_tags"].split(",")
        else:
            tags = []
        for tag in tags:
            if tag in FORM_TAG_MAPPING:
                form_tag_list.add(FORM_TAG_MAPPING[tag])
    if token.get("separable_particle"):
        form_tag_list.add(FORM_TAG_MAPPING["separable_particle"])
    return list(form_tag_list)


