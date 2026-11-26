import json
import os
from krajjat.tool_functions import sort_files_trailing_index, read_json
from constants import *

def merge_recording(original_path, output_path, subject, recording, verbosity=1):
    """Merges the independent frame files to single recording files."""

    # Define file output and sort files in input
    output_file = op.normpath(op.join(output_path, subject, recording + ".json"))
    input_path = op.normpath(op.join(original_path, subject, recording))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    input_files = sort_files_trailing_index(input_path, verbosity=verbosity)

    if verbosity >= 1:
        print(f"Merging {len(input_files)} frames from: {input_path}")
        print(f"Output: {output_file}")

    # Read frame 0 to add the metadata common to all files
    frame_0 = read_json(op.join(input_path, input_files[0]))
    merged = {k: v for k, v in frame_0.items() if k in METADATA_KEYS}
    merged["Poses"] = []

    # Read each frame
    for i, frame_path in enumerate(input_files):
        merged["Poses"].append(read_json(op.join(input_path, frame_path)))
    if verbosity == 1:
        print("100% - Done.")

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f)

    return output_path

if __name__ == "__main__":
    for subject in ALL_SUBJECTS:
        recordings = os.listdir(op.join(PATH_MOCAP_SINGLE_FILES, subject))
        for recording in recordings:
            merge_recording(PATH_MOCAP_SINGLE_FILES, PATH_MOCAP_ORIGINAL, subject, recording, verbosity=1)