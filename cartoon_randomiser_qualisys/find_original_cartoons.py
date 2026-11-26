"""
Script to identify which Wile E. Coyote cartoon a video clip comes from.
Uses cross-correlation via the find_delay package to match audio.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple
from datetime import timedelta
import warnings

# Third-party imports
try:
    from moviepy import VideoFileClip, AudioFileClip
    from find_delay import find_delay
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install find-delay moviepy")
    raise


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert timestamp string in format 'hh:mm:ss' to seconds.
    
    Args:
        timestamp: Time in format 'hh:mm:ss'
    
    Returns:
        Time in seconds as float
    """
    parts = timestamp.split(':')
    if len(parts) != 3:
        raise ValueError(f"Timestamp must be in format 'hh:mm:ss', got: {timestamp}")
    
    hours, minutes, seconds = map(float, parts)
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds to timestamp string in format 'hh:mm:ss.ms'.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted timestamp string
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def extract_audio_segment(video_path: str, start_time: str, end_time: str, 
                          output_path: str) -> str:
    """
    Extract audio from a video file between specified timestamps.
    
    Args:
        video_path: Path to video file
        start_time: Start timestamp in 'hh:mm:ss' format
        end_time: End timestamp in 'hh:mm:ss' format
        output_path: Path for output WAV file
    
    Returns:
        Path to the extracted audio WAV file
    """
    start_sec = timestamp_to_seconds(start_time)
    end_sec = timestamp_to_seconds(end_time)
    
    # Load video and extract audio segment
    with VideoFileClip(video_path) as video:
        audio = video.audio
        if audio is None:
            raise ValueError(f"No audio track found in {video_path}")
        
        # Extract the segment
        audio_segment = audio.subclipped(start_sec, end_sec)
        
        # Write to WAV file
        audio_segment.write_audiofile(output_path, codec='pcm_s16le', 
                                     fps=44100, nbytes=2, logger=None)
    
    return output_path


def extract_full_audio(video_path: str, output_path: str) -> str:
    """
    Extract complete audio from a video file.
    
    Args:
        video_path: Path to video file
        output_path: Path for output WAV file
    
    Returns:
        Path to the extracted audio WAV file
    """
    with VideoFileClip(video_path) as video:
        audio = video.audio
        if audio is None:
            raise ValueError(f"No audio track found in {video_path}")
        
        # Write to WAV file
        audio.write_audiofile(output_path, codec='pcm_s16le', 
                            fps=44100, nbytes=2, logger=None)
    
    return output_path


def find_cartoon_match(clip_path: str, start_time: str, end_time: str, 
                      cartoons_folder: str, temp_dir: str = None) -> List[Tuple[str, float, float]]:
    """
    Find which cartoon video contains the given clip using audio matching.
    
    Args:
        clip_path: Path to the video clip to identify
        start_time: Start timestamp of the relevant portion in 'hh:mm:ss' format
        end_time: End timestamp of the relevant portion in 'hh:mm:ss' format
        cartoons_folder: Path to folder containing cartoon videos
        temp_dir: Optional temporary directory for audio files (default: system temp)
    
    Returns:
        List of tuples: (cartoon_name, match_timestamp_seconds, correlation_value)
        sorted by correlation value (best match first), top 10 results
    """
    # Setup temporary directory
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='cartoon_match_')
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Using temporary directory: {temp_dir}")
    print(f"Extracting audio from clip: {clip_path}")
    print(f"Time range: {start_time} to {end_time}\n")
    
    # Extract audio from the clip
    clip_audio_path = os.path.join(temp_dir, "clip_audio.wav")
    extract_audio_segment(clip_path, start_time, end_time, clip_audio_path)
    
    # Get all video files in cartoons folder
    cartoons_path = Path(cartoons_folder)
    video_extensions = {'.avi', '.mp4', '.mkv', '.mov', '.wmv'}
    cartoon_files = [
        f for f in cartoons_path.iterdir() 
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    if not cartoon_files:
        raise ValueError(f"No video files found in {cartoons_folder}")
    
    print(f"Found {len(cartoon_files)} cartoon videos to analyze\n")
    
    results = []
    
    # Process each cartoon
    for i, cartoon_file in enumerate(cartoon_files, 1):
        cartoon_name = cartoon_file.name
        print(f"[{i}/{len(cartoon_files)}] Processing: {cartoon_name}")
        
        try:
            # Extract audio from cartoon
            cartoon_audio_path = os.path.join(temp_dir, f"cartoon_{i}.wav")
            extract_full_audio(str(cartoon_file), cartoon_audio_path)
            
            # Find delay using cross-correlation
            # return_delay_format="timedelta" returns a timedelta object
            # return_correlation_value=True returns the correlation coefficient
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                delay, correlation = find_delay(
                    cartoon_audio_path,
                    clip_audio_path,
                    return_delay_format="timedelta",
                    return_correlation_value=True,
                    mono_channel="average"  # Average stereo to mono if needed
                )
            
            # Convert timedelta to seconds
            if delay is not None:
                delay_seconds = delay.total_seconds()
                print(f"  → Match at {seconds_to_timestamp(delay_seconds)} "
                      f"(correlation: {correlation:.4f})")
                
                results.append((cartoon_name, delay_seconds, correlation))
            else:
                print(f"  → No match found")
            
            # Clean up cartoon audio file to save space
            if os.path.exists(cartoon_audio_path):
                os.remove(cartoon_audio_path)
                
        except Exception as e:
            print(f"  → Error processing {cartoon_name}: {str(e)}")
            continue
    
    # Sort by correlation value (highest first) and get top 10
    results.sort(key=lambda x: x[2], reverse=True)
    top_10 = results[:10]
    
    # Clean up clip audio
    if os.path.exists(clip_audio_path):
        os.remove(clip_audio_path)
    
    return top_10


def print_results(results: List[Tuple[str, float, float]]) -> None:
    """
    Print the top matching cartoons in a formatted table.
    
    Args:
        results: List of tuples (cartoon_name, timestamp_seconds, correlation)
    """
    print("\n" + "="*80)
    print("TOP 10 CARTOON MATCHES")
    print("="*80)
    
    if not results:
        print("No matches found.")
        return
    
    print(f"{'Rank':<6} {'Cartoon Name':<40} {'Timestamp':<15} {'Correlation':<12}")
    print("-" * 80)
    
    for rank, (cartoon_name, timestamp_sec, correlation) in enumerate(results, 1):
        timestamp_str = seconds_to_timestamp(timestamp_sec)
        print(f"{rank:<6} {cartoon_name:<40} {timestamp_str:<15} {correlation:<12.6f}")
    
    print("="*80)
    print(f"\nBest match: {results[0][0]}")
    print(f"Found at timestamp: {seconds_to_timestamp(results[0][1])}")
    print(f"Correlation coefficient: {results[0][2]:.6f}")


# Example usage
if __name__ == "__main__":
    # Example parameters - replace with actual values
    
    clip_path = "F025.mp4"
    start_time = "00:00:26"  # 5 seconds into the clip
    end_time = "00:00:49"    # 15 seconds into the clip
    cartoons_folder = "Wile E. Coyote"
    
    # Run the analysis
    try:
        results = find_cartoon_match(clip_path, start_time, end_time, cartoons_folder)
        print_results(results)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
