#!/usr/bin/env python3
"""
Parse molex_22_02_2022.tsv and initialize a SQLite database.

This script:
1. Parses the TSV file using csv.DictReader
2. Splits POS and Tags from linguistic annotations
3. Creates and populates Lemmas and Forms tables
4. Provides query methods to retrieve data in JSON format
"""

import csv
import os
import sqlite3
import json
import re
from pathlib import Path
from typing import List, Dict, Any


class GiGaNT_Lexicon:
    """Manages the MOLEX SQLite database for Dutch linguistic data."""

    def __init__(self):
        self.db_path = str(Path(__file__).parent / "LexiconData" / "molex.db")

        # Step 2: Load database from disk into RAM
        self.conn = sqlite3.connect(":memory:")

        self.conn.execute("PRAGMA journal_mode = OFF")
        self.conn.execute("PRAGMA synchronous = OFF")

        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        # Backup from disk into memory
        disk_conn = sqlite3.connect(self.db_path)
        disk_conn.backup(self.conn)
        disk_conn.close()

        print("Database loaded into RAM successfully")

    @staticmethod
    def _create_schema_on_conn(cursor, conn):
        """Create the database schema on a given connection."""
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS Lemmas
                       (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           molex_id TEXT NOT NULL UNIQUE,
                           text TEXT NOT NULL,
                           pos TEXT NOT NULL,
                           tags TEXT
                       )
                       """)

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS Forms
                       (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           lemma_id INTEGER NOT NULL,
                           molex_id
                           TEXT NOT NULL UNIQUE,
                           text TEXT NOT NULL,
                           pos TEXT,
                           tags TEXT,
                           FOREIGN KEY (lemma_id) REFERENCES Lemmas (id)
                           )
                       """)

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS Frequencies
                       (
                           word TEXT,
                           pos TEXT,
                           score REAL,
                           PRIMARY KEY (word, pos)
                       )
                       """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lemmas_text ON Lemmas(text)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_forms_text ON Forms(text)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_forms_lemma_id ON Forms(lemma_id)")

        conn.commit()

    @staticmethod
    def split_pos_tags(annotation: str) -> tuple:
        """
        Split a linguistic annotation into POS and tags.

        Args:
            annotation: String like "NOU-C(gender=n,number=sg)" or "CONJ"

        Returns:
            Tuple of (pos, tags) where tags is the bracket content or None
        """
        if not annotation:
            return None, None

        match = re.match(r"^([^(]+)(?:\((.+)\))?$", annotation.strip())
        if match:
            pos = match.group(1)
            tags = match.group(2) if match.group(2) else None
            return pos, tags
        return annotation.strip(), None

    def parse_and_populate(self, tsv_file: str = "LexiconData/molex_22_02_2022.tsv"):
        """
        Parse TSV file and populate the database.

        Args:
            tsv_file: Path to the molex_22_02_2022.tsv file
        """
        tsv_path = Path(tsv_file)
        if not tsv_path.exists():
            raise FileNotFoundError(f"TSV file not found: {tsv_file}")

        lemma_ids_seen = set()
        line_count = 0
        lemma_count = 0
        form_count = 0

        print(f"Reading {tsv_path.name}...")
        print("-" * 70)

        try:
            with open(tsv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")

                for row in reader:
                    # Expected columns:
                    # 0: Lemma id
                    # 1: Lemma
                    # 2: Lemmawoordsoort (POS + tags)
                    # 3: Woordvorm id
                    # 4: Woordvorm
                    # 5: Woordvormwoordsoort (POS + tags)

                    if len(row) < 6:
                        continue

                    line_count += 1

                    lemma_molex_id = row[0].strip()
                    lemma_text = row[1].strip()
                    lemma_annotation = row[2].strip()
                    form_molex_id = row[3].strip()
                    form_text = row[4].strip()
                    form_annotation = row[5].strip()

                    # Insert Lemma if not already seen
                    if lemma_molex_id not in lemma_ids_seen:
                        lemma_pos, lemma_tags = self.split_pos_tags(lemma_annotation)
                        self.cursor.execute(
                            """
                            INSERT INTO Lemmas (molex_id, text, pos, tags)
                            VALUES (?, ?, ?, ?)
                            """,
                            (lemma_molex_id, lemma_text, lemma_pos, lemma_tags),
                        )
                        lemma_ids_seen.add(lemma_molex_id)
                        lemma_count += 1

                    # Get the lemma_id for the foreign key
                    self.cursor.execute(
                        "SELECT id FROM Lemmas WHERE molex_id = ?", (lemma_molex_id,)
                    )
                    lemma_db_id = self.cursor.fetchone()[0]

                    # Insert Form
                    form_pos, form_tags = self.split_pos_tags(form_annotation)
                    self.cursor.execute(
                        """
                        INSERT INTO Forms (lemma_id, molex_id, text, pos, tags)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (lemma_db_id, form_molex_id, form_text, form_pos, form_tags),
                    )
                    form_count += 1

        except Exception as e:
            self.conn.rollback()
            print(f"Error reading file: {e}")
            raise

        self.conn.commit()
        print(f"Processed {line_count:,} lines")
        print(f"Inserted {lemma_count:,} lemmas")
        print(f"Inserted {form_count:,} forms")
        print("-" * 70)

    def get_candidates(self, search_word: str) -> List[Dict[str, Any]]:
        search_word = search_word.lower()
        self.cursor.execute(
            """
            SELECT DISTINCT 
                f.text as form_text,
                f.pos as form_pos,
                f.tags as form_tags,
                f.molex_id as form_molex_id,
                l.text as lemma_text,
                l.pos as lemma_pos,
                l.tags as lemma_tags,
                l.molex_id as lemma_molex_id
            FROM Forms f
            JOIN Lemmas l ON f.lemma_id = l.id
            WHERE f.text = ?
            """,
            (search_word,),
        )

        results = []
        for row in self.cursor.fetchall():
            result = {
                "form_text": row["form_text"],
                "form_pos": row["form_pos"],
                "form_tags": row["form_tags"],
                "form_molex_id": row["form_molex_id"],
                "lemma_text": row["lemma_text"],
                "lemma_pos": row["lemma_pos"],
                "lemma_tags": row["lemma_tags"],
                "lemma_molex_id": row["lemma_molex_id"]
            }
            results.append(result)

        return results
    
    def get_candidates_by_POS(self, search_word: str, pos: str) -> List[Dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT DISTINCT 
                f.text as form_text,
                f.pos as form_pos,
                f.tags as form_tags,
                l.text as lemma_text,
                l.pos as lemma_pos,
                l.tags as lemma_tags
            FROM Forms f
            JOIN Lemmas l ON f.lemma_id = l.id
            WHERE f.text = ? AND f.pos = ?
            """,
            (search_word.lower(), pos),
        )

        results = []
        for row in self.cursor.fetchall():
            result = {
                "form_text": row["form_text"],
                "form_pos": row["form_pos"],
                "form_tags": row["form_tags"],
                "lemma_text": row["lemma_text"],
                "lemma_pos": row["lemma_pos"],
                "lemma_tags": row["lemma_tags"],
            }
            results.append(result)

        return results

    def getSeparableParts(self) -> List[str]:
        """Get a list of all unique tags that indicate separability."""
        self.cursor.execute(
            """
            SELECT F.text as form
            FROM forms AS F JOIN main.Lemmas L on L.id = F.lemma_id
            WHERE L.tags LIKE '%sep=yes%' AND F.tags = 'finiteness=fin,mood=ind,tense=pres,NA=sg,PA=1,separated=yes'
            """
        )
        separableParts = set()
        for row in self.cursor.fetchall():
            separableParts.add(row["form"].split()[1])  # Get the second part of the form (the separable part)
        return list(separableParts)


    def createSeparablePartsFile(self, output_file: str = None):
        """Create a text file containing all unique tags that indicate separability."""
        if output_file is None:
            output_file = str(Path(__file__).parent / "LexiconData" / "separable_parts.txt")
        separable_verbs = self.getSeparableParts()
        with open(output_file, "w", encoding="utf-8") as f:
            for verb in separable_verbs:
                f.write(verb + "\n")
        print(f"Separable verbs written to {output_file}")

    def get_frequency(self, word: str, pos: str) -> float:
        """
        Get the frequency score for a word with a specific POS.

        - If word is single: direct lookup
        - If word has multiple parts: treat as potential separated verb and look up lemma

        Args:
            word: The word to look up (e.g., "aankomen" or "kom aan")
            pos: The part of speech

        Returns:
            The frequency score if found, otherwise 0.7 as default
        """
        word_count = len(word.split())

        if word_count == 1:
            # Single word - direct lookup
            result = self._get_frequency_direct(word, pos)
            return result if result is not None else 0.7

        else:
            # Multiple words - check if it's a separated verb
            if pos == "VRB":
                # Find the lemma for this separated form
                self.cursor.execute(
                    """
                    SELECT l.text, l.pos
                    FROM Forms f
                    JOIN Lemmas l ON f.lemma_id = l.id
                    WHERE f.text = ? 
                      AND f.pos = ?
                      AND f.tags LIKE '%separated=yes%'
                      AND l.tags LIKE '%sep=yes%'
                    LIMIT 1
                    """,
                    (word, pos),
                )

                row = self.cursor.fetchone()
                if row:
                    # Found separated verb - look up the lemma frequency
                    lemma_text = row[0]
                    result = self._get_frequency_direct(lemma_text, pos)
                    return result if result is not None else 0.7

            # Fallback: try direct lookup anyway
            result = self._get_frequency_direct(word, pos)
            return result if result is not None else 0.7


    def _get_frequency_direct(self, word: str, pos: str) -> float:
        """
        Internal method to directly look up frequency without fallback.

        Returns:
            The frequency score if found, otherwise None
        """
        self.cursor.execute(
            """
            SELECT score FROM Frequencies
            WHERE word = ? AND pos = ?
            """,
            (word, pos)
        )
        result = self.cursor.fetchone()
        return result['score'] if result else None


    def close(self):
        """Close the database connection."""
        self.conn.close()