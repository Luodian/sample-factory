#!/usr/bin/env python3
"""
Convert sampled frames and actions to training data format.
Creates a parquet file with interleaved images and text (actions).
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import json
import base64
from io import BytesIO

def clean_action_text(action_text):
    """
    Clean action text by removing the digit prefix.
    E.g., "8: DOWNRIGHT" -> "DOWNRIGHT"
    """
    if ": " in action_text:
        return action_text.split(": ", 1)[1].strip()
    return action_text.strip()


def load_image_as_base64(image_path):
    """Load an image and convert to base64 string."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (optional)
            max_size = (256, 256)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_base64
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_image_as_bytes(image_path):
    """Load an image as bytes."""
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def process_episode(episode_dir):
    """
    Process a single episode directory.
    Returns a list of dictionaries with frame/action pairs.
    """
    data_points = []
    
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(episode_dir, 'frame_*.png')))
    
    for frame_file in frame_files:
        # Extract frame number
        frame_num = os.path.basename(frame_file).replace('frame_', '').replace('.png', '')
        
        # Find corresponding action file
        action_file = os.path.join(episode_dir, f'action_{frame_num}.txt')
        
        if not os.path.exists(action_file):
            continue
        
        # Load action text
        with open(action_file, 'r') as f:
            action_raw = f.read().strip()
        
        # Clean action text (remove digit prefix)
        action_clean = clean_action_text(action_raw)
        
        # Load image as bytes
        image_bytes = load_image_as_bytes(frame_file)
        if image_bytes is None:
            continue
        
        # Create data point
        data_point = {
            'frame_path': frame_file,
            'frame_number': int(frame_num),
            'action': action_clean,
            'action_raw': action_raw,
            'image': image_bytes,
            'episode': os.path.basename(episode_dir)
        }
        
        data_points.append(data_point)
    
    return data_points


def create_interleaved_format(data_points, env_name):
    """
    Create interleaved image-text format for training.
    Returns a list of training examples matching the actual training format.
    """
    training_examples = []
    
    # Group by episode
    episodes = {}
    for dp in data_points:
        ep_name = dp['episode']
        if ep_name not in episodes:
            episodes[ep_name] = []
        episodes[ep_name].append(dp)
    
    # Sort each episode by frame number
    for ep_name in episodes:
        episodes[ep_name].sort(key=lambda x: x['frame_number'])
    
    # Create training examples for each episode
    for ep_name, ep_data in episodes.items():
        # Extract randomness from episode name
        randomness = ep_name.split('_rand')[-1] if '_rand' in ep_name else '0.0'
        
        # Create inputs list with interleaved text and image_gen entries
        inputs = []
        images = []
        
        # Add environment description - clean dictionary with only needed fields
        inputs.append({
            "type": "text",
            "has_loss": 0,
            "text": f"Atari {env_name.replace('atari_', '').title()} Environment (randomness={randomness})"
        })
        
        # Interleave frames and actions
        for i, dp in enumerate(ep_data):
            # Add image
            images.append(dp['image'])
            
            # Add image generation marker - clean dictionary with only needed fields
            inputs.append({
                "type": "image_gen",
                "has_loss": 1,
                "image_index": i  # Keep as regular int, will be preserved properly
            })
            
            # Add action (except for last frame) - clean dictionary with only needed fields
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
            'images_front': images[0:1] if images else [],  # First frame as front image
            'environment': env_name,
            'episode': ep_name,
            'num_frames': len(ep_data),
            'randomness': randomness
        }
        
        training_examples.append(example)
    
    return training_examples


def main():
    parser = argparse.ArgumentParser(description='Convert sampled frames to training data')
    parser.add_argument('--input-dir', required=True,
                        help='Input directory containing sampled frames')
    parser.add_argument('--output-file', required=True,
                        help='Output parquet file path')
    parser.add_argument('--env-filter', default=None,
                        help='Filter for specific environment')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Maximum number of episodes to process')
    parser.add_argument('--format', choices=['simple', 'interleaved'], default='interleaved',
                        help='Output format type')
    
    args = parser.parse_args()
    
    print(f"Converting frames from: {args.input_dir}")
    print(f"Output file: {args.output_file}")
    
    # Find all environment directories
    if args.env_filter:
        env_dirs = glob.glob(os.path.join(args.input_dir, f'*{args.env_filter}*'))
    else:
        env_dirs = glob.glob(os.path.join(args.input_dir, '*'))
    
    env_dirs = [d for d in env_dirs if os.path.isdir(d)]
    
    if not env_dirs:
        print("No environment directories found!")
        return 1
    
    print(f"Found {len(env_dirs)} environment(s)")
    
    all_data = []
    
    # Process each environment
    for env_dir in env_dirs:
        env_name = os.path.basename(env_dir)
        print(f"\nProcessing environment: {env_name}")
        
        # Find episode directories
        episode_dirs = sorted(glob.glob(os.path.join(env_dir, 'episode_*')))
        
        if args.max_episodes:
            episode_dirs = episode_dirs[:args.max_episodes]
        
        print(f"  Found {len(episode_dirs)} episode(s)")
        
        env_data_points = []
        
        # Process each episode
        for ep_dir in episode_dirs:
            ep_name = os.path.basename(ep_dir)
            print(f"    Processing {ep_name}...", end='')
            
            ep_data = process_episode(ep_dir)
            env_data_points.extend(ep_data)
            
            print(f" {len(ep_data)} frames")
        
        if args.format == 'interleaved':
            # Create interleaved format
            training_examples = create_interleaved_format(env_data_points, env_name)
            all_data.extend(training_examples)
        else:
            # Simple format - one row per frame
            for dp in env_data_points:
                all_data.append({
                    'environment': env_name,
                    'episode': dp['episode'],
                    'frame_number': dp['frame_number'],
                    'action': dp['action'],
                    'action_raw': dp['action_raw'],
                    'image': dp['image']
                })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Post-process inputs to ensure clean format (remove None values from dictionaries)
    if 'inputs' in df.columns and args.format == 'interleaved':
        def clean_input_entry(entry):
            """Remove None values from input dictionaries."""
            if entry['type'] == 'text':
                return {'type': 'text', 'has_loss': entry['has_loss'], 'text': entry['text']}
            else:  # image_gen
                return {'type': 'image_gen', 'has_loss': entry['has_loss'], 'image_index': int(entry['image_index'])}
        
        df['inputs'] = df['inputs'].apply(lambda inputs: [clean_input_entry(inp) for inp in inputs])
    
    print(f"\nTotal data points: {len(df)}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Save to parquet
    df.to_parquet(args.output_file, engine='pyarrow', compression='snappy')
    
    print(f"\nSaved to: {args.output_file}")
    
    # Print sample of the data
    if args.format == 'interleaved':
        print("\nSample training example:")
        if len(df) > 0:
            sample = df.iloc[0]
            print(f"  Environment: {sample['environment']}")
            print(f"  Episode: {sample['episode']}")
            print(f"  Num frames: {sample['num_frames']}")
            print(f"  Randomness: {sample['randomness']}")
            print(f"  Number of inputs: {len(sample['inputs'])}")
            print(f"  Number of images: {len(sample['images'])}")
            
            # Show first few input entries
            print(f"\n  First 5 input entries:")
            for i, inp in enumerate(sample['inputs'][:5]):
                if inp['type'] == 'text':
                    print(f"    [{i}] text (loss={inp['has_loss']}): {inp['text'][:50]}...")
                else:
                    print(f"    [{i}] image_gen (loss={inp['has_loss']}): index={inp['image_index']}")
    else:
        print("\nFirst 5 rows:")
        print(df[['environment', 'episode', 'frame_number', 'action']].head())
    
    return 0


if __name__ == '__main__':
    sys.exit(main())