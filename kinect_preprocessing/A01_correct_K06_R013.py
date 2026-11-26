import json
import os
from pathlib import Path
import numpy as np
from constants import *


def correct_kinect_timestamps(input_dir, output_dir, frame_prefix="frame_", frame_suffix=".json"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = len(os.listdir(input_dir))

    print(f"Loading timestamps from {total_frames} frames...")

    # Load all timestamps
    timestamps = []
    frames_data = []

    for i in range(total_frames):
        filepath = input_dir / f"{frame_prefix}{i}{frame_suffix}"

        with open(filepath, 'r', encoding='utf-16-le') as f:
            frame_data = json.load(f)
            timestamps.append(frame_data['Timestamp'])
            frames_data.append(frame_data)

    timestamps = np.array(timestamps, dtype=np.int64)

    print(f"Original timestamps loaded. Range: {timestamps[0]} to {timestamps[-1]}")

    deltas = np.diff(timestamps)

    if np.sum(deltas < 0) == 1:
        corruption_start = np.argmax(deltas < 0)
        print(f"Corruption found at frame {corruption_start}, correcting...")
        print(f"  Frame {corruption_start}: {timestamps[corruption_start]}")
        print(f"  Frame {corruption_start + 1}: {timestamps[corruption_start + 1]}")

        deltas_before_corruption = deltas[:corruption_start]
        average_delta = np.mean(deltas_before_corruption)

        print(f"\nCalculated average delta from frames 1-{corruption_start - 1}: {average_delta:.2f}")
        print(f"  Min delta: {deltas_before_corruption.min()}")
        print(f"  Max delta: {deltas_before_corruption.max()}")
        print(f"  Std delta: {np.std(deltas_before_corruption):.2f}")

        # Detect the time jump at frame 861
        if corruption_start < len(timestamps):
            time_jump = timestamps[corruption_start + 1] - timestamps[corruption_start]
            print(f"\nTime jump between frames {corruption_start} and {corruption_start + 1}: {time_jump:.2f}")
            print(f"  Expected delta: ~{average_delta:.2f}")
            print(f"  Actual delta: {time_jump:.2f}")
            print(f"  Difference: {time_jump - average_delta:.2f}")

        # Create corrected timestamps
        corrected_timestamps = timestamps.copy()
        corrected_timestamps[corruption_start + 1] = timestamps[corruption_start] + average_delta

        print(f"\nCorrecting frame {corruption_start + 1}:")
        print(f"  Original: {timestamps[corruption_start + 1]}")
        print(f"  Corrected: {corrected_timestamps[corruption_start + 1]}")

        # For frames 862-970 (indices 861-969), add individual deltas
        for i in range(corruption_start + 2, total_frames):
            # Calculate the delta from the corrupted timestamps
            individual_delta = timestamps[i] - timestamps[i - 1]
            # Add it to the corrected previous timestamp
            corrected_timestamps[i] = corrected_timestamps[i - 1] + individual_delta

        print(f"\nCorrected timestamps for frames {corruption_start}-{total_frames}")

        # Verify correction
        corrected_deltas_after = np.diff(corrected_timestamps[corruption_start - 1:corruption_start + 10])
        print(f"\nDeltas around corruption point (frame {corruption_start - 1} to {corruption_start + 9}):")
        print(f"  {corrected_deltas_after}")

        # Save corrected files
        print(f"\nSaving corrected files to {output_dir}...")

        for i in range(total_frames):
            # Update timestamp in the frame data
            frames_data[i]['Timestamp'] = int(corrected_timestamps[i])

            # Save to output directory
            output_filepath = output_dir / f"{frame_prefix}{i}{frame_suffix}"
            with open(output_filepath, 'w', encoding='utf-16-le') as f:
                json.dump(frames_data[i], f, indent=2)

            if (i + 1) % 100 == 0:
                print(f"  Saved {i + 1}/{total_frames} frames...")

        print(f"✓ All {total_frames} frames saved successfully!")

        # Calculate statistics
        all_corrected_deltas = np.diff(corrected_timestamps)

        summary = {'total_frames': total_frames, 'corruption_start_frame': corruption_start,
            'average_delta_before_corruption': float(average_delta),
            'corrected_frames': total_frames - corruption_start,
            'final_delta_stats': {'mean': float(np.mean(all_corrected_deltas)),
                'std': float(np.std(all_corrected_deltas)), 'min': float(np.min(all_corrected_deltas)),
                'max': float(np.max(all_corrected_deltas))},
            'original_time_range': {'start': int(timestamps[0]), 'end': int(timestamps[-1]),
                'duration': int(timestamps[-1] - timestamps[0])},
            'corrected_time_range': {'start': int(corrected_timestamps[0]), 'end': int(corrected_timestamps[-1]),
                'duration': int(corrected_timestamps[-1] - corrected_timestamps[0])}}

        # Save summary
        summary_path = output_dir / 'correction_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Summary saved to {summary_path}")

        print("\n" + "=" * 60)
        print("CORRECTION SUMMARY")
        print("=" * 60)
        print(f"Total frames processed: {summary['total_frames']}")
        print(f"Frames corrected: {summary['corrected_frames']}")
        print(f"Average delta (frames 1-{corruption_start - 1}): {summary['average_delta_before_corruption']:.2f}")
        print(f"\nOriginal duration: {summary['original_time_range']['duration']:,} ticks")
        print(f"Corrected duration: {summary['corrected_time_range']['duration']:,} ticks")
        print(f"\nFinal delta statistics:")
        print(f"  Mean: {summary['final_delta_stats']['mean']:.2f}")
        print(f"  Std:  {summary['final_delta_stats']['std']:.2f}")
        print(f"  Min:  {summary['final_delta_stats']['min']:.2f}")
        print(f"  Max:  {summary['final_delta_stats']['max']:.2f}")

        return summary

    return 0


if __name__ == "__main__":
    input_directory = op.join(PATH_MOCAP, "01_Original_gesture", "K06", "R013")
    output_directory =  op.join(PATH_MOCAP, "01_Original_gesture", "K06", "R013b")

    correct_kinect_timestamps(input_dir=input_directory, output_dir=output_directory)
