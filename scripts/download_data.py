#!/usr/bin/env python
"""
download_data.py  –  Instructions and helper for Paderborn dataset.

The Paderborn bearing dataset requires manual download from the university
website. This script prints instructions and verifies the downloaded file.

Usage
-----
    python scripts/download_data.py
"""

import os
import sys
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ZIP_NAME = 'paderborn-db.zip'
ZIP_PATH = os.path.join(DATA_DIR, ZIP_NAME)

EXPECTED_FOLDERS = [
    'K001', 'K002', 'K003', 'K004', 'K005', 'K006',  # healthy
    'KI04', 'KI14', 'KI16',                            # inner fault
    'KA04', 'KA15', 'KA16',                            # outer fault
    'KB23',                                             # mixed fault
]

INSTRUCTIONS = f"""
=====================================================================
  PADERBORN BEARING DATASET – Download Instructions
=====================================================================

1. Visit:
   https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/

2. Request access (free for research use).

3. Download the full dataset zip file.

4. Rename or copy the file to:
   {ZIP_PATH}

5. Re-run this script to verify:
   python scripts/download_data.py

=====================================================================
"""


def verify_zip(zip_path: str) -> bool:
    """Check that expected bearing folders exist in the zip."""
    print(f"Verifying: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
        found = [f for f in EXPECTED_FOLDERS if any(n.startswith(f + '/') for n in names)]
        missing = [f for f in EXPECTED_FOLDERS if f not in found]

        print(f"  Found {len(found)}/{len(EXPECTED_FOLDERS)} expected folders")
        if missing:
            print(f"  Missing: {missing}")
            return False
        print("  ✓ All required bearing folders found!")
        return True
    except zipfile.BadZipFile:
        print("  ✗ File is not a valid zip archive.")
        return False
    except FileNotFoundError:
        print(f"  ✗ File not found: {zip_path}")
        return False


if __name__ == '__main__':
    if not os.path.exists(ZIP_PATH):
        print(INSTRUCTIONS)
        sys.exit(1)
    else:
        ok = verify_zip(ZIP_PATH)
        sys.exit(0 if ok else 1)
