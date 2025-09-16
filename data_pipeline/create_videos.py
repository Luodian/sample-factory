#!/usr/bin/env python3
"""Create videos from sampled frames using imageio."""

import os
import sys
from pathlib import Path
import imageio
import numpy as np
from glob import glob
from natsort import natsorted

def create_video_from_frames(frames_dir, output_path, fps=10):
    """Create a video from PNG frames in a directory."""
    # Find all PNG files
    frame_pattern = os.path.join(frames_dir, "**", "*.png")
    frame_files = glob(frame_pattern, recursive=True)

    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return False

    # Sort frames naturally (handle episode_000, frame_000001 ordering)
    frame_files = natsorted(frame_files)

    print(f"Found {len(frame_files)} frames")
    print(f"First frame: {frame_files[0]}")
    print(f"Last frame: {frame_files[-1]}")

    try:
        # Create video writer
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                   pixelformat='yuv420p', quality=8)

        # Read and write frames
        for i, frame_file in enumerate(frame_files):
            if i % 100 == 0:
                print(f"Processing frame {i}/{len(frame_files)}")

            frame = imageio.imread(frame_file)
            writer.append_data(frame)

        writer.close()

        # Check output
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"Video created: {output_path}")
        print(f"Size: {file_size:.2f} MB")

        return True

    except Exception as e:
        print(f"Error creating video: {e}")
        return False

def main():
    # Configuration
    frames_base_dir = "/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/sampled_frames_multi_checkpoint"
    video_dir = f"{frames_base_dir}_videos"
    fps = 10

    print(f"=== Creating Videos from Frames ===")
    print(f"Frames directory: {frames_base_dir}")
    print(f"Output directory: {video_dir}")
    print(f"FPS: {fps}")
    print()

    # Create output directory
    os.makedirs(video_dir, exist_ok=True)

    # Find all checkpoint directories
    env_dirs = glob(os.path.join(frames_base_dir, "*_ckpt*"))
    env_dirs = [d for d in env_dirs if os.path.isdir(d)]

    if not env_dirs:
        print("No checkpoint directories found")
        return

    print(f"Found {len(env_dirs)} checkpoint directories")

    success_count = 0
    for env_dir in env_dirs:
        env_name = os.path.basename(env_dir)
        output_path = os.path.join(video_dir, f"{env_name}.mp4")

        print(f"\nProcessing {env_name}...")

        if create_video_from_frames(env_dir, output_path, fps):
            success_count += 1
        else:
            print(f"Failed to create video for {env_name}")

    print(f"\n=== Summary ===")
    print(f"Successfully created {success_count}/{len(env_dirs)} videos")

    # List created videos
    videos = glob(os.path.join(video_dir, "*.mp4"))
    if videos:
        print(f"\nCreated videos:")
        for video in videos:
            size = os.path.getsize(video) / (1024 * 1024)
            print(f"  {os.path.basename(video)}: {size:.2f} MB")

if __name__ == "__main__":
    main()