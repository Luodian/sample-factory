#!/usr/bin/env python3
"""
Sample frames from all available Atari environment checkpoints.
This script processes all checkpoints and saves frames with human-readable actions.
"""

import os
import sys
import glob
import subprocess
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Register Atari environments at import time
try:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass

# ANSI color codes
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


def extract_env_name(experiment_name):
    """Extract environment name from experiment directory name."""
    # Format: edbeeching_atari_2B_atari_alien_1111 -> atari_alien
    # Also handle typos like edbenching
    name = experiment_name.replace('edbeeching_atari_2B_', '').replace('edbenching_atari_2B_', '')
    # Remove seed suffixes
    for suffix in ['_1111', '_2222', '_3333']:
        name = name.replace(suffix, '')
    return name


def sample_environment(args):
    """Sample frames from a single environment."""
    experiment, checkpoint_dir, output_base_dir, frames_per_env, max_episodes, device, randomness = args
    
    env_name = extract_env_name(experiment)
    env_output_dir = os.path.join(output_base_dir, env_name)
    
    # Calculate frames per episode
    frames_per_episode = frames_per_env // max_episodes if max_episodes > 0 else frames_per_env
    
    # Skip if already processed with enough episodes
    if os.path.exists(env_output_dir):
        existing_episodes = [d for d in os.listdir(env_output_dir) 
                           if os.path.isdir(os.path.join(env_output_dir, d)) and d.startswith('episode_')]
        if len(existing_episodes) >= max_episodes:
            total_frames = sum([len(glob.glob(os.path.join(env_output_dir, ep, 'frame_*.png'))) 
                              for ep in existing_episodes])
            return env_name, 'skipped', total_frames
    
    os.makedirs(env_output_dir, exist_ok=True)
    total_frames_saved = 0
    
    # Sample each episode separately
    for episode_idx in range(max_episodes):
        # Create episode-specific directory with randomness in name
        episode_dir = os.path.join(env_output_dir, f'episode_{episode_idx:03d}_rand{randomness:.1f}')
        
        # Skip if this episode already exists
        if os.path.exists(episode_dir):
            existing_frames = glob.glob(os.path.join(episode_dir, 'frame_*.png'))
            total_frames_saved += len(existing_frames)
            continue
        
        # Create temp directory for this episode
        temp_dir = os.path.join(env_output_dir, f'temp_episode_{episode_idx}')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Prepare the command for a single episode
        # Use the current Python interpreter to ensure correct environment
        cmd = [
            sys.executable, '-m', 'sf_examples.atari.enjoy_atari',
            '--env', env_name,
            '--experiment', experiment,
            '--train_dir', checkpoint_dir,
            '--device', device,
            '--save_frames',
            '--frames_dir', temp_dir,
            '--video_frames', str(frames_per_episode),
            '--max_num_frames', str(frames_per_episode),  # IMPORTANT: Hard limit on frames
            '--max_num_episodes', '1',  # Only one episode at a time
            '--no_render'
        ]
        
        # Add randomness control (0.0 = deterministic, >0.1 = stochastic)
        if randomness < 0.1:
            cmd.extend(['--eval_deterministic', 'True'])
        else:
            cmd.extend(['--eval_deterministic', 'False'])
    
        try:
            # Run the command with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout per episode
            )
            
            # Check for errors in the result
            if result.returncode != 0:
                print(f"Error running {env_name}: {result.stderr[:500]}")
            
            # Move frames from temp to episode directory
            if os.path.exists(temp_dir):
                saved_frames = glob.glob(os.path.join(temp_dir, 'frame_*.png'))
                if saved_frames:
                    # Rename temp to final episode directory
                    os.rename(temp_dir, episode_dir)
                    total_frames_saved += len(saved_frames)
                else:
                    # Clean up empty temp directory
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
        except subprocess.TimeoutExpired:
            # Clean up temp directory on timeout
            print(f"Timeout for {env_name}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            continue
        except Exception as e:
            # Clean up temp directory on error
            print(f"Exception for {env_name}: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            continue
    
    if total_frames_saved > 0:
        return env_name, 'success', total_frames_saved
    else:
        return env_name, 'failed', 0


def main():
    parser = argparse.ArgumentParser(description='Sample frames from all Atari environment checkpoints')
    parser.add_argument('--checkpoint-dir', default='/opt/tiger/sample-factory/checkpoints',
                        help='Directory containing checkpoints')
    parser.add_argument('--output-dir', default='/opt/tiger/sample-factory/sampled_frames',
                        help='Output directory for sampled frames')
    parser.add_argument('--frames-per-env', type=int, default=100,
                        help='Number of frames to sample per environment (across all episodes)')
    parser.add_argument('--max-episodes', type=int, default=1,
                        help='Maximum episodes to run per environment')
    parser.add_argument('--randomness', type=float, default=0.2,
                        help='Randomness level (0.0=deterministic, 1.0=fully stochastic)')
    parser.add_argument('--parallel', type=int, default=4,
                        help='Number of parallel processes')
    parser.add_argument('--device', default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--seed', default='1111',
                        help='Seed suffix to use (1111, 2222, or 3333)')
    parser.add_argument('--specific-env', default=None,
                        help='Sample only a specific environment')
    
    args = parser.parse_args()
    
    print(f"{BLUE}{'='*50}{NC}")
    print(f"{BLUE}Sample Factory - Multi-Environment Frame Sampler{NC}")
    print(f"{BLUE}{'='*50}{NC}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of experiments
    if args.specific_env:
        # Check if specific_env already contains seed
        if any(seed in args.specific_env for seed in ['1111', '2222', '3333']):
            pattern = args.specific_env
        else:
            pattern = f"*{args.specific_env}*{args.seed}"
    else:
        pattern = f"*_{args.seed}"
    
    experiment_dirs = sorted(glob.glob(os.path.join(args.checkpoint_dir, pattern)))
    
    if not experiment_dirs:
        print(f"{RED}No checkpoint directories found matching pattern: {pattern}{NC}")
        return 1
    
    # Extract experiment names
    experiments = [os.path.basename(d) for d in experiment_dirs]
    
    print(f"{GREEN}Found {len(experiments)} environment(s) to sample{NC}")
    print(f"Output directory: {YELLOW}{args.output_dir}{NC}")
    print(f"Frames per environment: {args.frames_per_env}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Randomness: {args.randomness} ({'deterministic' if args.randomness < 0.1 else 'stochastic'})")
    print(f"Parallel processes: {args.parallel}\n")
    
    # Prepare arguments for parallel processing
    task_args = [
        (exp, args.checkpoint_dir, args.output_dir, 
         args.frames_per_env, args.max_episodes, args.device, args.randomness)
        for exp in experiments
    ]
    
    # Process environments in parallel
    results = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all tasks
        future_to_env = {executor.submit(sample_environment, arg): arg[0] for arg in task_args}
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_env), 1):
            env_name, status, frame_count = future.result()
            
            # Print progress
            if status == 'success':
                print(f"[{i}/{len(experiments)}] {GREEN}✓{NC} {env_name}: {frame_count} frames saved")
                success_count += 1
            elif status == 'skipped':
                print(f"[{i}/{len(experiments)}] {YELLOW}⏭{NC}  {env_name}: Already sampled ({frame_count} frames)")
                skipped_count += 1
            elif status == 'timeout':
                print(f"[{i}/{len(experiments)}] {RED}⏱{NC}  {env_name}: Timeout")
                failed_count += 1
            else:
                print(f"[{i}/{len(experiments)}] {RED}✗{NC} {env_name}: {status}")
                failed_count += 1
    
    # Print summary
    print(f"\n{BLUE}{'='*50}{NC}")
    print(f"{BLUE}SUMMARY{NC}")
    print(f"{BLUE}{'='*50}{NC}")
    print(f"{GREEN}Successful: {success_count}{NC}")
    print(f"{YELLOW}Skipped: {skipped_count}{NC}")
    print(f"{RED}Failed: {failed_count}{NC}")
    
    # Show sample of created content with episode structure
    if success_count > 0:
        print(f"\n{BLUE}Sample of created structure:{NC}")
        env_dirs = sorted(glob.glob(os.path.join(args.output_dir, '*')))[:3]
        for env_dir in env_dirs:
            if os.path.isdir(env_dir):
                env_name = os.path.basename(env_dir)
                episode_dirs = sorted([d for d in os.listdir(env_dir) 
                                     if os.path.isdir(os.path.join(env_dir, d)) and d.startswith('episode_')])
                if episode_dirs:
                    print(f"\n  {env_name}/")
                    for ep_dir in episode_dirs[:3]:
                        frame_count = len(glob.glob(os.path.join(env_dir, ep_dir, 'frame_*.png')))
                        action_count = len(glob.glob(os.path.join(env_dir, ep_dir, 'action_*.txt')))
                        print(f"    ├── {ep_dir}: {frame_count} frames, {action_count} actions")
    
    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())