#!/usr/bin/env python3
"""
Sample frames from VizDoom environments using random actions.
This script creates VizDoom environments and samples frames with random actions.
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse
import cv2

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec
from sample_factory.utils.attr_dict import AttrDict


def sample_vizdoom_env(env_name, output_dir, num_frames=16, seed=42):
    """Sample frames from a VizDoom environment using random actions."""
    
    # Find the environment spec
    env_spec = None
    for spec in DOOM_ENVS:
        if spec.name == env_name:
            env_spec = spec
            break
    
    if env_spec is None:
        print(f"Environment {env_name} not found!")
        print(f"Available environments: {[spec.name for spec in DOOM_ENVS]}")
        return False
    
    # Create a minimal config for the environment
    cfg = AttrDict({
        'env_frameskip': 4,
        'res_w': 128,
        'res_h': 72,
        'wide_aspect_ratio': True,
        'pixel_format': 'HWC',
        'fps': 35,
        'experiment': 'vizdoom_random_sampling',
        'record_to': None,
    })
    
    # Create the environment
    print(f"Creating environment: {env_name}")
    # make_doom_env_from_spec expects (_env_name, cfg, env_config) as positional args
    env_config = None  # We don't need env_config for simple sampling
    env = make_doom_env_from_spec(env_spec, env_name, cfg, env_config)
    
    # Create output directory for this environment
    env_output_dir = Path(output_dir) / env_name
    env_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(seed)
    
    # Reset environment
    obs, info = env.reset()
    
    frames_saved = 0
    actions_saved = []
    
    print(f"Sampling {num_frames} frames from {env_name}...")
    
    while frames_saved < num_frames:
        # Sample random action
        action = env.action_space.sample()
        
        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Save frame
        frame_path = env_output_dir / f"frame_{frames_saved:04d}.png"
        
        # Convert observation to uint8 if needed
        if obs.dtype != np.uint8:
            obs_save = (obs * 255).astype(np.uint8)
        else:
            obs_save = obs
        
        # Save the frame
        cv2.imwrite(str(frame_path), cv2.cvtColor(obs_save, cv2.COLOR_RGB2BGR))
        
        # Save action as text
        action_path = env_output_dir / f"action_{frames_saved:04d}.txt"
        with open(action_path, 'w') as f:
            f.write(f"Action: {action}\n")
            if hasattr(env_spec.action_space, 'n'):
                f.write(f"Action space size: {env_spec.action_space.n}\n")
        
        frames_saved += 1
        actions_saved.append(action)
        
        # Reset if episode ends
        if done:
            obs, info = env.reset()
            print(f"  Episode ended, resetting... ({frames_saved}/{num_frames} frames saved)")
    
    env.close()
    
    print(f"Successfully saved {frames_saved} frames to {env_output_dir}")
    print(f"Action statistics: min={min(actions_saved)}, max={max(actions_saved)}, unique={len(set(actions_saved))}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Sample frames from VizDoom environments with random actions')
    parser.add_argument('--env', default='doom_basic', 
                        help='VizDoom environment name (e.g., doom_basic, doom_health_gathering)')
    parser.add_argument('--output-dir', default='sampled_vizdoom_frames',
                        help='Output directory for sampled frames')
    parser.add_argument('--num-frames', type=int, default=16,
                        help='Number of frames to sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--list-envs', action='store_true',
                        help='List all available VizDoom environments')
    
    args = parser.parse_args()
    
    # Register VizDoom components
    register_vizdoom_components()
    
    if args.list_envs:
        print("Available VizDoom environments:")
        for spec in DOOM_ENVS:
            print(f"  - {spec.name}")
        return 0
    
    # Sample from the environment
    success = sample_vizdoom_env(
        args.env,
        args.output_dir,
        args.num_frames,
        args.seed
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())