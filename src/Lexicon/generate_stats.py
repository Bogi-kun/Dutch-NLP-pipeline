#!/usr/bin/env python3
"""
Extract all possible POS values and tags from the molex.db database.
"""

import sqlite3
import json
from pathlib import Path
from collections import defaultdict

def extract_pos_and_tags(db_path):
    """
    Connect to molex.db and extract:
    - All unique POS values from lemmas and forms
    - All unique tags from lemmas
    - All unique tags from forms
    """
    
    # Check if database exists
    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        return
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Extract unique POS from Lemmas
        cursor.execute("SELECT DISTINCT pos FROM Lemmas ORDER BY pos")
        lemma_pos = [row['pos'] for row in cursor.fetchall()]
        
        # Extract unique POS from Forms
        cursor.execute("SELECT DISTINCT pos FROM Forms WHERE pos IS NOT NULL ORDER BY pos")
        form_pos = [row['pos'] for row in cursor.fetchall()]
        
        # Extract unique tags from Lemmas
        cursor.execute("SELECT DISTINCT tags FROM Lemmas WHERE tags IS NOT NULL ORDER BY tags")
        lemma_tags = [row['tags'] for row in cursor.fetchall()]
        
        # Extract unique tags from Forms
        cursor.execute("SELECT DISTINCT tags FROM Forms WHERE tags IS NOT NULL ORDER BY tags")
        form_tags = [row['tags'] for row in cursor.fetchall()]
        
        # Split all tags by "." delimiter and collect into sets
        lemma_tag_components = set()
        for tag in lemma_tags:
            if tag:
                components = tag.split(',')
                lemma_tag_components.update(components)
        
        form_tag_components = set()
        for tag in form_tags:
            if tag:
                components = tag.split(',')
                form_tag_components.update(components)
        
        # Combine all tag components
        all_tag_components = sorted(lemma_tag_components | form_tag_components)
        
        # Combine and deduplicate all POS values
        all_pos = sorted(set(lemma_pos + form_pos))
        
        # Print results
        print("=" * 80)
        print("EXTRACTED LINGUISTIC DATA FROM molex.db")
        print("=" * 80)
        
        print("\n[UNIQUE POS VALUES]")
        print("-" * 80)
        for pos in all_pos:
            print(f"  {pos}")
        print(f"Total unique POS values: {len(all_pos)}")
        
        # Count occurrences for each POS in lemmas and forms
        print("\n[POS VALUE STATISTICS - Lemmas vs Forms]")
        print("-" * 80)
        pos_stats = {}
        
        for pos in all_pos:
            # Count lemmas with this POS
            cursor.execute(
                """
                SELECT COUNT(L.id) as count
                FROM Lemmas L
                WHERE L.pos = ?
                """,
                (pos,)
            )
            pos_lemma_count = cursor.fetchone()['count']
            
            # Count forms with this POS
            cursor.execute(
                """
                SELECT COUNT(F.id) as count
                FROM Forms F
                WHERE F.pos = ?
                """,
                (pos,)
            )
            pos_form_count = cursor.fetchone()['count']
            
            pos_stats[pos] = {
                "lemma_count": pos_lemma_count,
                "form_count": pos_form_count,
                "total_count": pos_lemma_count + pos_form_count
            }

            print(f"\n  POS: {pos}")
            print(f"    Lemmas: {pos_lemma_count}")
            print(f"    Forms: {pos_form_count}")
            print(f"    Total: {pos_lemma_count + pos_form_count}")
        
        print("\n[UNIQUE LEMMA TAGS]")
        print("-" * 80)
        for tag in lemma_tags:
            print(f"  {tag}")
        print(f"Total unique lemma tags: {len(lemma_tags)}")
        
        print("\n[UNIQUE LEMMA TAG COMPONENTS (split by '.')]")
        print("-" * 80)
        for component in sorted(lemma_tag_components):
            print(f"  {component}")
        print(f"Total unique lemma tag components: {len(lemma_tag_components)}")
        
        print("\n[UNIQUE FORM TAGS]")
        print("-" * 80)
        for tag in form_tags:
            print(f"  {tag}")
        print(f"Total unique form tags: {len(form_tags)}")
        
        print("\n[UNIQUE FORM TAG COMPONENTS (split by '.')]")
        print("-" * 80)
        for component in sorted(form_tag_components):
            print(f"  {component}")
        print(f"Total unique form tag components: {len(form_tag_components)}")
        
        print("\n[ALL TAG COMPONENTS (Lemmas + Forms combined)]")
        print("-" * 80)
        for component in all_tag_components:
            print(f"  {component}")
        print(f"Total unique combined tag components: {len(all_tag_components)}")
        
        # Count occurrences for lemmas and forms separately
        print("\n[TAG COMPONENT STATISTICS - Lemmas vs Forms]")
        print("-" * 80)
        tag_component_stats = {}
        
        for component in all_tag_components:
            # Count lemmas with this tag component
            cursor.execute(
                """
                SELECT COUNT(L.id) as count
                FROM Lemmas L
                WHERE L.tags LIKE ?
                """,
                (f'%{component}%',)
            )
            component_lemma_count = cursor.fetchone()['count']
            
            # Count forms with this tag component
            cursor.execute(
                """
                SELECT COUNT(F.id) as count
                FROM Forms F
                WHERE F.tags LIKE ?
                """,
                (f'%{component}%',)
            )
            component_form_count = cursor.fetchone()['count']
            
            tag_component_stats[component] = {
                "lemma_count": component_lemma_count,
                "form_count": component_form_count,
                "total_count": component_lemma_count + component_form_count
            }
            
            print(f"\n  Component: {component}")
            print(f"    Lemmas: {component_lemma_count}")
            print(f"    Forms: {component_form_count}")
            print(f"    Total: {component_lemma_count + component_form_count}")
        
        # Summary statistics
        print("\n[SUMMARY STATISTICS]")
        print("-" * 80)
        cursor.execute("SELECT COUNT(DISTINCT molex_id) FROM Lemmas")
        lemma_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT molex_id) FROM Forms")
        form_count = cursor.fetchone()[0]
        
        print(f"Total unique lemmas: {lemma_count}")
        print(f"Total unique word forms: {form_count}")
        print(f"Total unique POS values: {len(all_pos)}")
        print(f"Total unique lemma tags: {len(lemma_tags)}")
        print(f"Total unique form tags: {len(form_tags)}")
        
        # Export to JSON for easier processing
        output_data = {
            "pos_values": all_pos,
            "pos_stats": pos_stats,
            "lemma_tags": lemma_tags,
            "lemma_tag_components": sorted(list(lemma_tag_components)),
            "form_tags": form_tags,
            "form_tag_components": sorted(list(form_tag_components)),
            "all_tag_components": all_tag_components,
            "tag_component_stats": tag_component_stats,
            "statistics": {
                "total_lemmas": lemma_count,
                "total_forms": form_count,
                "unique_pos_count": len(all_pos),
                "unique_lemma_tags_count": len(lemma_tags),
                "unique_lemma_tag_components_count": len(lemma_tag_components),
                "unique_form_tags_count": len(form_tags),
                "unique_form_tag_components_count": len(form_tag_components),
                "total_combined_tag_components": len(all_tag_components)
            }
        }
        
        # Save to JSON file
        output_file = Path(__file__).parent / "LexiconData" / "lexicon_stats.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults exported to: {output_file}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    extract_pos_and_tags()

