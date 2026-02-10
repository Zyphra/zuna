#!/usr/bin/env python3
"""
Tutorial: PT to FIF Conversion
"""
import zuna

INPUT_DIR = "data/3_pt_output"   # Folder with .pt files
OUTPUT_DIR = "data/4_fif_output"  # Where to save .fif files

results = zuna.pt_directory_to_fif(INPUT_DIR, OUTPUT_DIR)

print(f"Reconstructed {results['successful']}/{results['total']} original FIF files successfully")
print(f"(Multiple PT files from same source were stitched together)")

# Print errors if any
if results['errors']:
    print("\nErrors encountered:")
    for error in results['errors']:
        print(f"  File(s): {error['files']}")
        print(f"  Error: {error['error']}")
        print()
