# MOLEX Database Script - Implementation Summary

## Overview
A complete Python script has been created to parse the `molex_22_02_2022.tsv` file and initialize a SQLite database with proper schema and relationships.

## Features Implemented

### 1. **File Parsing**
- Uses `csv.reader` with tab delimiter to parse the TSV file
- Handles UTF-8 encoding for Dutch characters
- Processes 751,448 lines with 223,152 unique lemmas

### 2. **Tag Splitting Logic**
The script intelligently splits POS and linguistic tags using regex:
```python
# Example transformations:
"NOU-C(gender=n,number=sg)" → POS: "NOU-C", Tags: "gender=n,number=sg"
"CONJ" → POS: "CONJ", Tags: None
"VRB(finiteness=fin,mood=ind,tense=past)" → POS: "VRB", Tags: "finiteness=fin,mood=ind,tense=past"
```

### 3. **Database Schema**

#### Lemmas Table
```sql
CREATE TABLE Lemmas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    molex_id TEXT NOT NULL UNIQUE,      -- Original MOLEX identifier
    text TEXT NOT NULL,                  -- The lemma text
    pos TEXT NOT NULL,                   -- Part of Speech (e.g., "NOU-C")
    tags TEXT                            -- Linguistic annotations (nullable)
)
```

#### Forms Table
```sql
CREATE TABLE Forms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lemma_id INTEGER NOT NULL,           -- Foreign Key to Lemmas
    molex_id TEXT NOT NULL UNIQUE,       -- Original MOLEX identifier
    text TEXT NOT NULL,                  -- The word form
    pos TEXT,                            -- Part of Speech (nullable - some entries lack POS)
    tags TEXT,                           -- Linguistic annotations (nullable)
    FOREIGN KEY (lemma_id) REFERENCES Lemmas(id)
)
```

### 4. **Key Features**

- **Duplicate Prevention**: Uses a set to track lemma_id values and avoid duplicating lemmas
- **Error Handling**: Implements try-except with rollback on errors
- **Data Integrity**: Enforces foreign key constraints
- **Clean Output**: JSON format excludes internal IDs and MOLEX IDs as required

### 5. **Query Method: get_candidates(search_word)**

Returns a list of JSON objects with the format:
```json
{
  "form": {
    "text": "aprilmopje",
    "pos": "NOU-C",
    "tags": "number=sg"
  },
  "lemma": {
    "text": "aprilmopje",
    "pos": "NOU-C",
    "tags": "gender=n,number=sg"
  }
}
```

## Database Statistics

| Metric | Count |
|--------|-------|
| Total Lines Processed | 751,448 |
| Unique Lemmas | 223,152 |
| Total Forms | 751,448 |
| Unique Lemma POS Tags | 12 |
| Unique Form POS Tags | ~12 |

### POS Tags Found
- AA (Adjective)
- ADP (Adposition)
- ADV (Adverb)
- COLL (Collocation)
- CONJ (Conjunction)
- INT (Interjection)
- NOU-C (Common Noun)
- NOU-P (Proper Noun)
- NUM (Numeral)
- PD (Pronoun/Determiner)
- RES (Reserved)
- VRB (Verb)

## Usage

### Basic Usage

```python
from src.Lexicon import GiGaNT_Lexicon

# Create and populate database
db = GiGaNT_Lexicon("LexiconData/molex.db")
db.parse_and_populate("molex_22_02_2022.tsv")

# Query for word candidates
candidates = db.get_candidates("halaleten")
for candidate in candidates:
    print(candidate)

db.close()
```

### Run as Script
```bash
python BuildDictionary.py
```

This will:
1. Create `molex.db` in the script directory
2. Parse the TSV file
3. Populate all tables
4. Display statistics
5. Show an example query result for "halaleten"

## Example Output

```
Reading molex_22_02_2022.tsv...
----------------------------------------------------------------------
Processed 751,448 lines
Inserted 223,152 lemmas
Inserted 751,448 forms
----------------------------------------------------------------------

Example query - searching for 'halaleten':
----------------------------------------------------------------------
{
  "form": {
    "text": "halaleten",
    "pos": "NOU-C",
    "tags": "number=sg"
  },
  "lemma": {
    "text": "halaleten",
    "pos": "NOU-C",
    "tags": "gender=n,number=sg"
  }
}

Database created successfully at: /.../DutchNLP/molex.db
```

## Implementation Details

### Regex Pattern for Tag Splitting
```python
r"^([^(]+)(?:\((.+)\))?$"
```
- `^([^(]+)` - Captures everything before the first opening bracket (the POS tag)
- `(?:\((.+)\))?` - Optionally captures content inside brackets (the tags)
- `$` - End of string

### Data Validation
- Handles rows with fewer than 6 columns (skips them)
- Handles missing POS annotations (Form.pos is nullable)
- Validates UTF-8 encoding throughout

## Files Created

1. **BuildDictionary.py** - Main script with MolexDatabase class
2. **molex.db** - SQLite database (created on first run)
---

**Status**: ✅ Complete and functional
**Date Created**: 2026-04-27

