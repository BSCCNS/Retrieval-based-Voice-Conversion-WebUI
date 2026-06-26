"""
Concatenate groups of 50 MP3 files into 20 WAV files at 16kHz.
Adds fade-out/fade-in crossfades at the join points between clips.

Requirements:
    pip install pydub numpy
    Also requires ffmpeg to be installed on your system.

Usage:
    python concatenate_mp3s.py --input_dir /path/to/mp3s --output_dir /path/to/output
"""

import argparse
import os
import glob
from typing import List
import numpy as np
from pydub import AudioSegment

# ── Configuration ────────────────────────────────────────────────────────────
FILES_PER_GROUP   = 50       # Number of MP3s per output WAV
OUTPUT_SAMPLE_RATE = 16_000  # Hz
CROSSFADE_MS      = 100      # Fade-out + fade-in duration at each join (ms)
# ─────────────────────────────────────────────────────────────────────────────


def load_and_normalize(path: str) -> AudioSegment:
    """Load an MP3 and convert to mono 16 kHz."""
    audio = AudioSegment.from_mp3(path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(OUTPUT_SAMPLE_RATE)
    audio = audio.set_sample_width(2)   # 16-bit PCM
    return audio


def concatenate_with_crossfade(segments: List[AudioSegment], crossfade_ms: int) -> AudioSegment:
    """
    Join a list of AudioSegments with a smooth fade-out / fade-in at every join.

    pydub's built-in .append(seg, crossfade=N) overlaps the last N ms of the
    current clip with the first N ms of the next one (cross-dissolve).  That is
    exactly the "fade-out then fade-in at boundary" behaviour requested.
    """
    if not segments:
        raise ValueError("No segments to concatenate.")

    result = segments[0]
    for seg in segments[1:]:
        # Guard: if a clip is shorter than the crossfade window, skip the fade
        fade = min(crossfade_ms, len(result), len(seg))
        result = result.append(seg, crossfade=fade)

    return result


def process_group(mp3_paths: List[str], output_path: str) -> None:
    """Load, concatenate with crossfade, and export one group."""
    segments = []
    for p in mp3_paths:
        try:
            segments.append(load_and_normalize(p))
        except Exception as e:
            print(f"  ⚠  Skipping {os.path.basename(p)}: {e}")

    if not segments:
        print(f"  ✗  No valid audio for {output_path}, skipping.")
        return

    combined = concatenate_with_crossfade(segments, CROSSFADE_MS)
    combined.export(output_path, format="wav")
    duration_s = len(combined) / 1000
    print(f"  ✓  {output_path}  ({len(segments)} clips, {duration_s:.1f} s)")


def main():
    parser = argparse.ArgumentParser(description="Batch-concatenate MP3s into 16 kHz WAV files.")
    parser.add_argument("--input_dir",  required=True, help="Folder containing the 1000 MP3 files.")
    parser.add_argument("--output_dir", required=True, help="Folder where the 20 WAV files will be saved.")
    parser.add_argument("--files_per_group", type=int, default=FILES_PER_GROUP,
                        help=f"MP3s per output file (default: {FILES_PER_GROUP}).")
    parser.add_argument("--crossfade_ms", type=int, default=CROSSFADE_MS,
                        help=f"Crossfade length in milliseconds (default: {CROSSFADE_MS}).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect and sort all MP3 files
    mp3_files = sorted(glob.glob(os.path.join(args.input_dir, "*.mp3")))
    if not mp3_files:
        raise FileNotFoundError(f"No .mp3 files found in: {args.input_dir}")

    total = len(mp3_files)
    n_groups = (total + args.files_per_group - 1) // args.files_per_group  # ceiling division
    print(f"Found {total} MP3 files → {n_groups} output WAV files "
          f"(~{args.files_per_group} files each, crossfade={args.crossfade_ms} ms)\n")

    for i in range(n_groups):
        start = i * args.files_per_group
        end   = start + args.files_per_group
        group = mp3_files[start:end]

        out_name = f"combined_{i+1:03d}.wav"
        out_path = os.path.join(args.output_dir, out_name)

        print(f"[{i+1}/{n_groups}] Processing files {start+1}–{min(end, total)} …")
        process_group(group, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
