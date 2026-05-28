import math
from pathlib import Path


def weighed_log_sum(freq1, freq2, alpha):
    return math.log(freq1 + 1) + alpha * math.log(freq1+1 / freq2 + 1)


def createFrequencyList():
    CNG_TO_GIGANT = {
        "ADJ": "AA",
        "BW": "ADV",
        "LID": "PD",
        "N": "NOU-C",
        "TSW": "INT",
        "TW": "NUM",
        "VG": "CONJ",
        "VNW": "PD",
        "VZ": "ADP",
        "WW": "VRB",
    }

    file = Path(__file__).parent / "LexiconData" / "SUBTLEX-NL.tsv"

    POS = set()

    FORM_FREQUENCY_LIST = dict()  # (FORM, POS): F

    with open(file, "r", encoding="utf-8") as f:
        firstLine = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            word = parts[0]  # FORM
            pos = parts[-3].strip(".").split(".")  # all pos
            freq = list(map(int, parts[-2].strip(".").split(".")))  # word freq
            lemma_freq = list(map(int, parts[-1].strip(".").split(".")))  # lemma freq
            for p, f, l in zip(pos, freq, lemma_freq):
                if p in CNG_TO_GIGANT:
                    gig_pos = CNG_TO_GIGANT[p]
                    POS.add(gig_pos)
                    FORM_FREQUENCY_LIST[(word, gig_pos)] = weighed_log_sum(f, l, alpha=0.25)

        min_value = min(FORM_FREQUENCY_LIST.values())
        max_value = max(FORM_FREQUENCY_LIST.values())

    return FORM_FREQUENCY_LIST
