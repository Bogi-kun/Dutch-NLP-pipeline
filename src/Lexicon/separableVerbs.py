#!/usr/bin/env python3
"""
Display one separable verb example for each suffix in separable_parts.txt

This script:
1. Reads all suffixes from separable_parts.txt
2. For each suffix, queries the database to find one separated verb form
3. Displays the results in a formatted table
"""

from pathlib import Path


def get_separable_parts():
    """Read separable parts from file."""
    separable_parts_path = Path(__file__).parent / "LexiconData" / "separable_parts.txt"

    if not separable_parts_path.exists():
        print(f"Error: {separable_parts_path} not found!")
        return []

    with open(separable_parts_path, "r", encoding="utf-8") as f:
        suffixes = [line.strip() for line in f if line.strip()]

    return suffixes