#!/usr/bin/env python3
"""Sample 2 minutes of gameplay from VizDoom models at 1 FPS."""

import os
import sys
import torch
import imageio
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.utils import log

def sample_vizdoom_long(checkpoint_path, output_dir="/tmp/vizdoom_long_samples", fps=1, duration_seconds=120, device="cuda"):
    """Sample long gameplay from VizDoom model."""

    # Register VizDoom components
    register_vizdoom_components()

    # Parse arguments
    parser, cfg = parse_sf_args(argv=["--env", "doom_battle", "--experiment", "test"], evaluation=True)
    add_doom_env_args(parser)
    cfg = parse_full_cfg(parser, argv=["--env", "doom_battle", "--experiment", "test"])
    doom_override_defaults(cfg)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get('cfg', cfg)

    # Update config for evaluation
    cfg.num_workers = 1
    cfg.num_envs_per_worker = 1
    cfg.batch_size = 1
    cfg.rollout = 128
    cfg.use_rnn = True
    cfg.device = device

    # Extract environment name from checkpoint
    env_name = checkpoint_path.split('/')[-2] if '/' in checkpoint_path else "doom_battle"
    if env_name == "doom_duel_selfplay":
        env_name = "doom_duel"
    cfg.env = env_name

    # Create environment
    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode="rgb_array"
    )

    # Get actor-critic model
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

    # Load weights with key conversion
    state_dict = checkpoint['model']
    new_state_dict = {}
    for key, value in state_dict.items():
        # Convert fc_layers to mlp_layers for compatibility
        new_key = key.replace('fc_layers', 'mlp_layers')
        new_state_dict[new_key] = value

    actor_critic.load_state_dict(new_state_dict, strict=False)
    actor_critic.eval()
    actor_critic.to(device)

    # Initialize RNN states
    rnn_size = get_rnn_size(cfg)
    rnn_states = torch.zeros([env.num_agents, rnn_size], dtype=torch.float32, device=device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    env_output_dir = os.path.join(output_dir, env_name)
    os.makedirs(env_output_dir, exist_ok=True)

    # Calculate total frames needed
    total_frames_needed = duration_seconds * fps
    frame_skip = 35 // fps  # VizDoom runs at ~35 FPS internally

    log.info(f"Sampling {duration_seconds} seconds ({total_frames_needed} frames) at {fps} FPS from {env_name}")
    log.info(f"Using frame skip of {frame_skip} to achieve {fps} FPS output")

    # Sampling loop
    all_frames = []
    total_frames = 0
    episode_count = 0
    max_episodes = 10

    while total_frames < total_frames_needed and episode_count < max_episodes:
        obs, _ = env.reset()
        done = [False]
        episode_frames = []
        episode_reward = 0
        frame_counter = 0

        # Reset RNN states for new episode
        rnn_states = torch.zeros([env.num_agents, rnn_size], dtype=torch.float32, device=device)

        while not done[0] and total_frames < total_frames_needed:
            with torch.no_grad():
                # Convert observations to tensor format
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

            # Save frame at specified FPS
            if frame_counter % frame_skip == 0:
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)
                    all_frames.append(frame)
                    total_frames += 1

            frame_counter += 1

        episode_count += 1
        log.info(f"Episode {episode_count}: reward = {episode_reward:.2f}, frames captured = {len(episode_frames)}")

        # Save episode video
        if episode_frames:
            video_path = os.path.join(env_output_dir, f"{env_name}_episode_{episode_count}.mp4")
            writer = imageio.get_writer(video_path, fps=fps)
            for frame in episode_frames:
                writer.append_data(frame)
            writer.close()
            log.info(f"Saved episode video to {video_path}")

    # Save complete video with all frames
    if all_frames:
        full_video_path = os.path.join(env_output_dir, f"{env_name}_full_{duration_seconds}s_{fps}fps.mp4")
        writer = imageio.get_writer(full_video_path, fps=fps)
        for frame in all_frames:
            writer.append_data(frame)
        writer.close()
        log.info(f"Saved full video to {full_video_path}")

        # Save frames as parquet
        import pandas as pd
        import pyarrow as pa
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
        parquet_path = os.path.join(env_output_dir, f"{env_name}_{duration_seconds}s_{fps}fps.parquet")
        df.to_parquet(parquet_path, compression='snappy')
        log.info(f"Saved {len(all_frames)} frames to {parquet_path}")

    return all_frames, episode_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample long gameplay from VizDoom models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="/tmp/vizdoom_long_samples", help="Output directory")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for output video")
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    frames, episodes = sample_vizdoom_long(
        args.checkpoint,
        args.output,
        args.fps,
        args.duration,
        args.device
    )

    log.info(f"Sampling complete: {len(frames)} frames from {episodes} episodes")