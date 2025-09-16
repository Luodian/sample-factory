#!/usr/bin/env python3
"""
Simple VizDoom sampling that handles the fc_layers -> mlp_layers conversion.
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

def sample_vizdoom(checkpoint_path, output_dir, env_name='doom_battle', num_frames=256, device='cpu'):
    """Sample VizDoom with checkpoint key conversion."""

    # Register components
    register_vizdoom_components()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Setup environment
    argv = [
        '--env', env_name,
        '--device', device,
        '--no_render',
        '--eval_env_frameskip', '4',
    ]

    parser, _ = parse_sf_args(argv=argv, evaluation=True)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)

    # Update config from checkpoint if available
    if 'cfg' in checkpoint:
        saved_cfg = checkpoint['cfg']
        for key in ['encoder_type', 'encoder_custom']:
            if hasattr(saved_cfg, key):
                setattr(cfg, key, getattr(saved_cfg, key))

    # Create environment
    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode="rgb_array"
    )

    # Create model
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

    # Handle key conversion
    state_dict = checkpoint['model']
    new_state_dict = {}
    for key, value in state_dict.items():
        # Convert fc_layers to mlp_layers
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

    # Sampling loop
    frames = []
    total_frames = 0
    episode_count = 0
    max_episodes = 10

    log.info(f"Starting to sample {num_frames} frames from {env_name}...")

    while total_frames < num_frames and episode_count < max_episodes:
        obs, _ = env.reset()
        done = [False]
        episode_reward = 0

        # Reset RNN states for new episode
        rnn_states = torch.zeros([env.num_agents, rnn_size], dtype=torch.float32, device=device)

        while not done[0] and total_frames < num_frames:
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
            episode_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward

            # Render and save frame
            frame = env.render()
            if frame is not None:
                if len(frame.shape) == 4:  # Batched
                    frame = frame[0]

                # Ensure uint8
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)

                frames.append(frame)

                # Save individual frame
                frame_path = os.path.join(output_dir, f'frame_{total_frames:06d}.png')
                imageio.imwrite(frame_path, frame)

                total_frames += 1

        episode_count += 1
        reward_value = float(episode_reward.item()) if hasattr(episode_reward, 'item') else float(episode_reward)
        log.info(f"Episode {episode_count}: reward = {reward_value:.2f}, frames = {total_frames}")

    # Save video
    if frames:
        video_path = os.path.join(output_dir, f'{env_name}_sample.mp4')
        writer = imageio.get_writer(video_path, fps=30, codec='libx264',
                                   pixelformat='yuv420p', quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        log.info(f"Saved video to {video_path}")

    env.close()

    return total_frames, episode_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='doom_battle')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--frames', type=int, default=256)
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()

    frames, episodes = sample_vizdoom(
        args.checkpoint,
        args.output_dir,
        args.env,
        args.frames,
        args.device
    )

    print(f"Sampled {frames} frames over {episodes} episodes")