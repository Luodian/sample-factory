#!/usr/bin/env python3
"""
Process a single Atari environment: sample frames and convert to parquet.
"""

import os
import sys
import argparse
import subprocess
import glob
import shutil
from pathlib import Path
import pandas as pd

# Import the sample_all_envs functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sample_all_envs import sample_environment, extract_env_name

# Import convert_to_training_data functions  
from convert_to_training_data import process_episode, generate_enhanced_description

def sample_single_env(env_name, checkpoint_dir, output_dir, frames_per_env, max_episodes, randomness, device='cpu'):
    """Sample frames from a single environment."""
    
    # Find the checkpoint for this environment (prefer seed 1111)
    checkpoint_pattern = f"{checkpoint_dir}/edbeeching_atari_2B_{env_name}_1111"
    if not os.path.exists(checkpoint_pattern):
        # Try other seeds
        checkpoint_pattern = f"{checkpoint_dir}/edbeeching_atari_2B_{env_name}_*"
        checkpoints = glob.glob(checkpoint_pattern)
        if not checkpoints:
            print(f"No checkpoint found for {env_name}")
            return False
        experiment = os.path.basename(checkpoints[0])
    else:
        experiment = os.path.basename(checkpoint_pattern)
    
    print(f"Sampling {env_name} using checkpoint {experiment}...")
    
    # Use the sample_environment function from sample_all_envs.py
    args = (experiment, checkpoint_dir, output_dir, frames_per_env, max_episodes, device, randomness)
    env_name_result, status, frame_count = sample_environment(args)
    
    if status == 'success':
        print(f"Successfully sampled {frame_count} frames for {env_name}")
        return True
    else:
        print(f"Failed to sample {env_name}: {status}")
        return False


def convert_env_to_parquet(env_name, frames_dir, output_file):
    """Convert a single environment's frames to parquet."""
    
    env_dir = os.path.join(frames_dir, env_name)
    if not os.path.exists(env_dir):
        print(f"No frames found for {env_name}")
        return False
    
    # Get all episode directories
    episode_dirs = sorted([d for d in os.listdir(env_dir) 
                          if os.path.isdir(os.path.join(env_dir, d)) and d.startswith('episode_')])
    
    if not episode_dirs:
        print(f"No episodes found for {env_name}")
        return False
    
    training_examples = []
    
    # Process each episode
    for ep_dir in episode_dirs:
        episode_path = os.path.join(env_dir, ep_dir)
        
        # Process the episode
        ep_data = process_episode(episode_path)
        
        if not ep_data:
            continue
        
        # Extract randomness from episode name (e.g., episode_000_rand0.2)
        ep_name = os.path.basename(episode_path)
        randomness = 0.0
        if '_rand' in ep_name:
            try:
                randomness = float(ep_name.split('_rand')[1])
            except:
                pass
        
        # Create inputs list with interleaved text and image_gen entries
        inputs = []
        images = []
        
        # Generate enhanced description
        enhanced_desc = generate_enhanced_description(env_name)
        
        # Add environment description
        inputs.append({
            "type": "text",
            "has_loss": 0,
            "text": enhanced_desc
        })
        
        # Interleave frames and actions
        for i, dp in enumerate(ep_data):
            # Add image
            images.append(dp['image'])
            
            # Add image generation marker
            inputs.append({
                "type": "image_gen",
                "has_loss": 1,
                "image_index": i
            })
            
            # Add action (except for last frame)
            if i < len(ep_data) - 1:
                inputs.append({
                    "type": "text",
                    "has_loss": 0,
                    "text": dp['action'].lower()
                })
        
        # Create training example
        example = {
            'inputs': inputs,
            'images': images,
            'images_front': images[0:1] if images else [],
            'environment': env_name,
            'episode': ep_name,
            'num_frames': len(ep_data),
            'randomness': randomness
        }
        
        training_examples.append(example)
    
    if not training_examples:
        print(f"No valid episodes found for {env_name}")
        return False
    
    # Convert to DataFrame
    df = pd.DataFrame(training_examples)
    
    # Save to parquet
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(df)} episodes for {env_name} to {output_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Process individual Atari environment')
    parser.add_argument('--env-name', required=True, help='Environment name (e.g., atari_alien)')
    parser.add_argument('--checkpoint-dir', default='/opt/tiger/atari_2B/checkpoints',
                        help='Directory containing checkpoints')
    parser.add_argument('--frames-dir', default='/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/sampled_frames',
                        help='Directory to save sampled frames')
    parser.add_argument('--output-dir', default='/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/parquet_files',
                        help='Directory to save parquet files')
    parser.add_argument('--frames-per-env', type=int, default=32,
                        help='Number of frames per episode')
    parser.add_argument('--max-episodes', type=int, default=100,
                        help='Maximum episodes to sample')
    parser.add_argument('--randomness', type=float, default=0.2,
                        help='Randomness level')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--skip-sampling', action='store_true',
                        help='Skip sampling and only convert existing frames')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.frames_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Sample frames (unless skipped)
    if not args.skip_sampling:
        success = sample_single_env(
            args.env_name,
            args.checkpoint_dir,
            os.path.join(args.frames_dir, args.env_name),
            args.frames_per_env,
            args.max_episodes,
            args.randomness,
            args.device
        )
        
        if not success:
            print(f"Failed to sample {args.env_name}")
            return 1
    
    # Step 2: Convert to parquet
    output_file = os.path.join(args.output_dir, f"{args.env_name}.parquet")
    success = convert_env_to_parquet(
        args.env_name,
        args.frames_dir,
        output_file
    )
    
    if not success:
        print(f"Failed to convert {args.env_name} to parquet")
        return 1
    
    print(f"Successfully processed {args.env_name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())