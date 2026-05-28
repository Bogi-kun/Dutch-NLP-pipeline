def getPunctuation(path):
    file = open(path, "r", encoding="utf-8")
    punctuation = []
    file.readline() # Skip header
    for line in file:
        punctEntry = line.strip().split(",")
         # String from unicode
        punctEntry.append(chr(int(punctEntry[2].strip("U+"), 16)))
        punctuation.append(punctEntry)
    return punctuation