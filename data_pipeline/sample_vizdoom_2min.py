#!/usr/bin/env python3
"""
Simple VizDoom sampling for 2 minutes at 1 FPS.
"""

import os
import sys
import torch
import numpy as np
import imageio.v2 as imageio
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.utils import log

def sample_vizdoom_long(checkpoint_path, output_dir, env_name='doom_battle', total_seconds=120, fps=1, device='cpu'):
    """Sample VizDoom for 2 minutes at 1 FPS."""

    # Register components
    register_vizdoom_components()

    # Load checkpoint with weights_only=False (safe for our trusted models)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Parse config with minimal args to avoid conflicts
    import sys
    old_argv = sys.argv
    sys.argv = ['sample_vizdoom_2min.py', '--env', env_name, '--experiment', 'test']

    parser, cfg = parse_sf_args(evaluation=True)
    add_doom_env_args(parser)
    cfg = parse_full_cfg(parser, argv=['--env', env_name, '--experiment', 'test'])
    doom_override_defaults(parser)

    sys.argv = old_argv

    # Use config from checkpoint if available
    if 'cfg' in checkpoint:
        checkpoint_cfg = checkpoint['cfg']
        cfg.__dict__.update(checkpoint_cfg.__dict__)

    # Update config for evaluation
    cfg.env = env_name
    cfg.device = device
    cfg.num_workers = 1
    cfg.num_envs_per_worker = 1
    cfg.batch_size = 1
    cfg.rollout = 128
    cfg.use_rnn = True
    cfg.recurrence = cfg.rollout

    # Create environment
    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode='rgb_array'
    )

    # Create actor-critic
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

    # Convert old checkpoint keys to new format
    state_dict = checkpoint['model']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('fc_layers', 'mlp_layers')
        new_state_dict[new_key] = value

    # Load with strict=False to allow mismatches
    actor_critic.load_state_dict(new_state_dict, strict=False)
    actor_critic.eval()
    actor_critic.to(device)

    # Initialize RNN states
    rnn_size = get_rnn_size(cfg)
    rnn_states = torch.zeros([env.num_agents, rnn_size], dtype=torch.float32, device=device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    env_dir = os.path.join(output_dir, env_name)
    os.makedirs(env_dir, exist_ok=True)

    # Calculate frames needed
    total_frames_needed = total_seconds * fps
    frame_skip = 35 // fps  # VizDoom runs at ~35 FPS internally

    log.info(f"Sampling {total_seconds} seconds ({total_frames_needed} frames) at {fps} FPS from {env_name}...")
    log.info(f"Frame skip: {frame_skip} (capturing every {frame_skip}th frame)")

    # Sampling loop
    all_frames = []
    total_frames = 0
    episode_count = 0
    max_episodes = 20  # Allow more episodes to reach target

    while total_frames < total_frames_needed and episode_count < max_episodes:
        obs, _ = env.reset()
        done = [False]
        episode_reward = 0
        episode_frames = []
        frame_counter = 0

        # Reset RNN states for new episode
        rnn_states = torch.zeros([env.num_agents, rnn_size], dtype=torch.float32, device=device)

        while not done[0] and total_frames < total_frames_needed:
            with torch.no_grad():
                # Convert observations to float if needed
                if isinstance(obs, dict):
                    obs_dict = {}
                    for key, value in obs.items():
                        if hasattr(value, 'dtype') and value.dtype == torch.uint8:
                            obs_dict[key] = value.float() / 255.0
                        else:
                            obs_dict[key] = value
                else:
                    obs_dict = {'obs': obs}

                # Get action from policy
                policy_outputs = actor_critic(obs_dict, rnn_states)

                # Extract action
                if hasattr(policy_outputs, 'actions'):
                    actions = policy_outputs.actions
                else:
                    actions = policy_outputs['actions']

                # Update RNN states
                if hasattr(policy_outputs, 'new_rnn_states'):
                    rnn_states = policy_outputs.new_rnn_states
                elif 'new_rnn_states' in policy_outputs:
                    rnn_states = policy_outputs['new_rnn_states']

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated
            episode_reward += reward[0] if isinstance(reward, (list, tuple)) else reward

            # Capture frame at specified FPS
            if frame_counter % frame_skip == 0:
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)
                    all_frames.append(frame)
                    total_frames += 1

                    if total_frames % 10 == 0:
                        log.info(f"Captured {total_frames}/{total_frames_needed} frames")

            frame_counter += 1

        episode_count += 1
        # Convert tensor to float if needed
        if hasattr(episode_reward, 'item'):
            episode_reward = episode_reward.item()
        log.info(f"Episode {episode_count}: reward = {episode_reward:.2f}, frames captured = {len(episode_frames)}")

        # Save episode video
        if episode_frames:
            video_path = os.path.join(env_dir, f"episode_{episode_count:02d}.mp4")
            writer = imageio.get_writer(video_path, fps=fps)
            for frame in episode_frames:
                writer.append_data(frame)
            writer.close()
            log.info(f"Saved episode video to {video_path}")

    # Save complete video
    if all_frames:
        full_video_path = os.path.join(env_dir, f"{env_name}_complete_{total_seconds}s_{fps}fps.mp4")
        writer = imageio.get_writer(full_video_path, fps=fps)
        for frame in all_frames:
            writer.append_data(frame)
        writer.close()
        log.info(f"Saved complete video to {full_video_path}")

        # Save frames as parquet
        try:
            import pandas as pd
            import pyarrow.parquet as pq

            frame_data = []
            for i, frame in enumerate(all_frames):
                frame_data.append({
                    'frame_idx': i,
                    'timestamp': i / fps,
                    'frame': frame.tobytes(),
                    'shape': frame.shape,
                    'dtype': str(frame.dtype)
                })

            df = pd.DataFrame(frame_data)
            parquet_path = os.path.join(env_dir, f"{env_name}_{total_seconds}s_{fps}fps.parquet")
            df.to_parquet(parquet_path, compression='snappy')
            log.info(f"Saved {len(all_frames)} frames to {parquet_path}")
        except ImportError:
            log.warning("pandas/pyarrow not available, skipping parquet export")

    env.close()
    return all_frames, episode_count

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample 2 minutes from VizDoom at 1 FPS")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="/tmp/vizdoom_2min", help="Output directory")
    parser.add_argument("--env", type=str, default="doom_battle", help="Environment name")
    parser.add_argument("--seconds", type=int, default=120, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second")
    parser.add_argument("--device", type=str, default="cpu", help="Device")

    args = parser.parse_args()

    frames, episodes = sample_vizdoom_long(
        args.checkpoint,
        args.output,
        args.env,
        args.seconds,
        args.fps,
        args.device
    )

    log.info(f"Sampling complete: {len(frames)} frames from {episodes} episodes")