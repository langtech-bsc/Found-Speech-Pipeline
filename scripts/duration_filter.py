#!/usr/bin/env python3
import json
import os
import sys

def filter_and_cleanup(json_path):
    """
    - Loads the JSON at json_path.
    - For each segment in each top‐level entry, computes duration = end - start.
    - If not (duration > 2 and duration <= 30):
        • Deletes the file at segment_path (if it exists).
        • Omits that segment from the filtered JSON.
    - Overwrites the original JSON with the filtered version.
    """
    # 1. Load the existing JSON.
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Iterate over each top-level item.
    for item_key, item_content in data.items():
        if 'results' not in item_content:
            continue

        filtered_results = []
        for segment in item_content['results']:
            start = segment.get('start')
            end = segment.get('end')
            seg_path = segment.get('segment_path')

            # If either start or end is missing, keep the segment by default.
            if start is None or end is None:
                filtered_results.append(segment)
                continue

            duration = end - start

            # Remove segments that do NOT satisfy 2 < duration <= 30
            if not (duration > 2 and duration <= 30):
                if seg_path and os.path.isfile(seg_path):
                    try:
                        os.remove(seg_path)
                        print(f"Deleted file: {seg_path}")
                    except Exception as e:
                        print(f"Warning: could not delete {seg_path}: {e}")
                # Skip adding this segment to filtered_results
            else:
                # Duration is in the allowed range; keep it.
                filtered_results.append(segment)

        # Replace the "results" list with the filtered one.
        item_content['results'] = filtered_results

    # 3. Overwrite the original JSON file with the filtered data.
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Filtering complete. Updated JSON written to: {json_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 duration_filter.py /path/to/final_output.json")
        sys.exit(1)

    json_file = sys.argv[1]
    if not os.path.isfile(json_file):
        print(f"Error: file not found: {json_file}")
        sys.exit(1)

    filter_and_cleanup(json_file)