#!/usr/bin/env python3
"""
Sample VizDoom at high resolution (up to 1920x1080).
"""

import os
import sys
import torch
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import make_doom_env_impl, doom_env_by_name
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.utils import log

# Available resolutions from observation_space.py
RESOLUTIONS = [
    "160x120", "200x125", "200x150", "256x144", "256x160", "256x192",
    "320x180", "320x200", "320x240", "320x256", "400x225", "400x250",
    "400x300", "512x288", "512x320", "512x384", "640x360", "640x400",
    "640x480", "800x450", "800x500", "800x600", "1024x576", "1024x640",
    "1024x768", "1280x720", "1280x800", "1280x960", "1280x1024", "1400x787",
    "1400x875", "1400x1050", "1600x900", "1600x1000", "1600x1200", "1920x1080"
]

def sample_vizdoom_hires(checkpoint_path, output_dir, env_name='doom_battle',
                         resolution='1280x720', num_frames=120, fps=30, device='cpu'):
    """Sample VizDoom at high resolution."""

    # Register components
    register_vizdoom_components()

    # Load checkpoint with weights_only=False (safe for our trusted models)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Parse config with minimal args to avoid conflicts
    old_argv = sys.argv
    sys.argv = ['sample_vizdoom_hires.py', '--env', env_name, '--experiment', 'test']

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

    # Create environment with high resolution
    log.info(f"Creating environment with resolution: {resolution}")
    env_spec = doom_env_by_name(env_name)
    env = make_doom_env_impl(
        env_spec,
        cfg=cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode='rgb_array',
        custom_resolution=resolution  # Set custom high resolution
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
    num_agents = getattr(env, 'num_agents', 1)  # Default to 1 if not specified
    rnn_states = torch.zeros([num_agents, rnn_size], dtype=torch.float32, device=device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    env_dir = os.path.join(output_dir, f"{env_name}_{resolution}")
    os.makedirs(env_dir, exist_ok=True)

    log.info(f"Sampling {num_frames} frames at {resolution} resolution, {fps} FPS from {env_name}...")

    # Sampling loop
    all_frames = []
    total_frames = 0
    episode_count = 0
    max_episodes = 5

    while total_frames < num_frames and episode_count < max_episodes:
        obs, _ = env.reset()
        done = [False]
        episode_frames = []
        episode_reward = 0

        # Reset RNN states for new episode
        rnn_states = torch.zeros([num_agents, rnn_size], dtype=torch.float32, device=device)

        while not done[0] and total_frames < num_frames:
            with torch.no_grad():
                # Convert observations to tensor format
                if isinstance(obs, dict):
                    obs_dict = {}
                    for key, value in obs.items():
                        # Convert numpy arrays to tensors
                        if isinstance(value, np.ndarray):
                            value = torch.from_numpy(value).to(device)
                        if hasattr(value, 'dtype') and value.dtype == torch.uint8:
                            obs_dict[key] = value.float() / 255.0
                        else:
                            obs_dict[key] = value
                else:
                    # Convert numpy array to tensor if needed
                    if isinstance(obs, np.ndarray):
                        obs = torch.from_numpy(obs).to(device)
                    if obs.dtype == torch.uint8:
                        obs = obs.float() / 255.0
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

            # Capture frame
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)
                all_frames.append(frame)
                total_frames += 1

                if total_frames % 30 == 0:
                    log.info(f"Captured {total_frames}/{num_frames} frames at {resolution}")

        episode_count += 1
        # Convert tensor to float if needed
        if hasattr(episode_reward, 'item'):
            episode_reward = episode_reward.item()
        log.info(f"Episode {episode_count}: reward = {episode_reward:.2f}, frames = {len(episode_frames)}")

        # Save episode video
        if episode_frames:
            video_path = os.path.join(env_dir, f"episode_{episode_count:02d}_{resolution}.mp4")
            writer = imageio.get_writer(video_path, fps=fps)
            for frame in episode_frames:
                writer.append_data(frame)
            writer.close()
            log.info(f"Saved episode video to {video_path}")

    # Save complete video
    if all_frames:
        full_video_path = os.path.join(env_dir, f"{env_name}_complete_{resolution}_{fps}fps.mp4")
        writer = imageio.get_writer(full_video_path, fps=fps)
        for frame in all_frames:
            writer.append_data(frame)
        writer.close()
        log.info(f"Saved complete video to {full_video_path}")

        # Save sample frames as images
        frames_dir = os.path.join(env_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Save first, middle, and last frames
        sample_indices = [0, len(all_frames)//2, len(all_frames)-1]
        for idx in sample_indices[:min(3, len(all_frames))]:
            frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
            imageio.imwrite(frame_path, all_frames[idx])
            log.info(f"Saved sample frame to {frame_path}")

    env.close()
    log.info(f"High-resolution sampling complete: {total_frames} frames at {resolution}")
    return all_frames, episode_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample VizDoom at high resolution")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="/tmp/vizdoom_hires", help="Output directory")
    parser.add_argument("--env", type=str, default="doom_battle", help="Environment name")
    parser.add_argument("--resolution", type=str, default="1280x720",
                       choices=RESOLUTIONS, help="Resolution to sample at")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to capture")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output video")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")

    args = parser.parse_args()

    frames, episodes = sample_vizdoom_hires(
        args.checkpoint,
        args.output,
        args.env,
        args.resolution,
        args.frames,
        args.fps,
        args.device
    )

    log.info(f"Sampling complete: {len(frames)} frames from {episodes} episodes at {args.resolution}")