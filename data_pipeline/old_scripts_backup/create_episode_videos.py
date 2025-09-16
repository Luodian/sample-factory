#!/usr/bin/env python3
"""
Create separate videos for each episode from sampled frames.
Each episode directory will be converted to its own video file.
"""

import os
import sys
import argparse
import imageio.v2 as imageio
from glob import glob
from natsort import natsorted
from multiprocessing import Pool, cpu_count
from pathlib import Path


def create_episode_video(args):
    """Create a video from frames in an episode directory."""
    episode_dir, video_dir, fps = args

    # Extract environment and episode info from path
    # Path format: .../atari_alien_ckpt2000093184/atari_alien/episode_000_rand0.3
    path_parts = episode_dir.split('/')
    env_checkpoint = path_parts[-3]  # e.g., atari_alien_ckpt2000093184
    env_name = path_parts[-2]  # e.g., atari_alien
    episode_name = path_parts[-1]  # e.g., episode_000_rand0.3

    # Create subdirectory for this env/checkpoint
    env_video_dir = os.path.join(video_dir, env_checkpoint)
    os.makedirs(env_video_dir, exist_ok=True)

    output_path = os.path.join(env_video_dir, f'{episode_name}.mp4')

    # Skip if video already exists
    if os.path.exists(output_path):
        print(f'  Skipping {env_checkpoint}/{episode_name}.mp4 (already exists)')
        return True

    # Find all frames in this episode
    frame_files = glob(os.path.join(episode_dir, '*.png'))
    frame_files = natsorted(frame_files)

    if not frame_files:
        print(f'  No frames found for {env_checkpoint}/{episode_name}')
        return False

    try:
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                   pixelformat='yuv420p', quality=8,
                                   macro_block_size=1)

        for frame_file in frame_files:
            frame = imageio.imread(frame_file)
            writer.append_data(frame)

        writer.close()

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f'  Created: {env_checkpoint}/{episode_name}.mp4 ({len(frame_files)} frames, {size_mb:.1f} MB)')
        return True

    except Exception as e:
        print(f'  Failed to create video for {env_checkpoint}/{episode_name}: {e}')
        return False


def main():
    parser = argparse.ArgumentParser(description='Create videos from episode frames')
    parser.add_argument('--frames-dir', type=str,
                        default='/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/sampled_frames_multi_checkpoint',
                        help='Directory containing frame directories')
    parser.add_argument('--video-dir', type=str, default=None,
                        help='Output directory for videos (default: frames_dir + "_videos")')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for videos')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--env-filter', type=str, default=None,
                        help='Filter for specific environment (e.g., "atari_alien")')

    args = parser.parse_args()

    frames_dir = args.frames_dir
    video_dir = args.video_dir or f"{frames_dir}_videos"

    print(f'Creating episode videos from: {frames_dir}')
    print(f'Output directory: {video_dir}')
    print(f'FPS: {args.fps}')
    print(f'Workers: {args.workers}')

    os.makedirs(video_dir, exist_ok=True)

    # Find all episode directories
    episode_dirs = []
    env_checkpoint_dirs = glob(os.path.join(frames_dir, '*_ckpt*'))

    # Apply environment filter if specified
    if args.env_filter:
        env_checkpoint_dirs = [d for d in env_checkpoint_dirs if args.env_filter in d]

    for env_ckpt_dir in env_checkpoint_dirs:
        if not os.path.isdir(env_ckpt_dir):
            continue
        # Find environment subdirectory
        env_subdirs = glob(os.path.join(env_ckpt_dir, '*'))
        for env_subdir in env_subdirs:
            if not os.path.isdir(env_subdir):
                continue
            # Find episode directories
            ep_dirs = glob(os.path.join(env_subdir, 'episode_*'))
            episode_dirs.extend(ep_dirs)

    if not episode_dirs:
        print('No episode directories found')
        return

    print(f'Found {len(episode_dirs)} episode directories')

    # Process videos in parallel
    num_workers = min(args.workers, cpu_count())
    print(f'Creating videos using {num_workers} workers...')

    with Pool(num_workers) as pool:
        args_list = [(ep_dir, video_dir, args.fps) for ep_dir in episode_dirs]
        results = pool.map(create_episode_video, args_list)

    success_count = sum(results)
    print(f'\nSuccessfully created {success_count}/{len(episode_dirs)} episode videos')

    # Print summary statistics
    total_videos = len(glob(os.path.join(video_dir, '**', '*.mp4'), recursive=True))
    print(f'Total videos in output directory: {total_videos}')


if __name__ == '__main__':
    main()