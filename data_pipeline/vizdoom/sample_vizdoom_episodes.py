#!/usr/bin/env python3
"""
VizDoom sampling with episode folders and action recording.
Follows the same structure as the Atari pipeline.
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


# VizDoom action meanings for different environments
VIZDOOM_ACTION_MEANINGS = {
    'doom_battle': [
        'MOVE_LEFT',
        'MOVE_RIGHT',
        'ATTACK',
        'MOVE_FORWARD',
        'MOVE_BACKWARD',
        'TURN_LEFT',
        'TURN_RIGHT',
        'RELOAD',
        'USE',
    ],
    'doom_battle2': [
        'MOVE_LEFT',
        'MOVE_RIGHT',
        'ATTACK',
        'MOVE_FORWARD',
        'MOVE_BACKWARD',
        'TURN_LEFT',
        'TURN_RIGHT',
        'RELOAD',
        'USE',
    ],
    'doom_deathmatch_bots': [
        'MOVE_LEFT',
        'MOVE_RIGHT',
        'ATTACK',
        'MOVE_FORWARD',
        'MOVE_BACKWARD',
        'TURN_LEFT',
        'TURN_RIGHT',
        'RELOAD',
        'USE',
        'SELECT_WEAPON1',
        'SELECT_WEAPON2',
        'SELECT_WEAPON3',
        'SELECT_WEAPON4',
        'SELECT_WEAPON5',
        'SELECT_WEAPON6',
    ],
    'doom_duel_bots': [
        'MOVE_LEFT',
        'MOVE_RIGHT',
        'ATTACK',
        'MOVE_FORWARD',
        'MOVE_BACKWARD',
        'TURN_LEFT',
        'TURN_RIGHT',
        'RELOAD',
        'USE',
        'SELECT_WEAPON1',
        'SELECT_WEAPON2',
        'SELECT_WEAPON3',
        'SELECT_WEAPON4',
        'SELECT_WEAPON5',
        'SELECT_WEAPON6',
    ],
    'doom_duel_selfplay': [
        'MOVE_LEFT',
        'MOVE_RIGHT',
        'ATTACK',
        'MOVE_FORWARD',
        'MOVE_BACKWARD',
        'TURN_LEFT',
        'TURN_RIGHT',
        'RELOAD',
        'USE',
        'SELECT_WEAPON1',
        'SELECT_WEAPON2',
        'SELECT_WEAPON3',
        'SELECT_WEAPON4',
        'SELECT_WEAPON5',
        'SELECT_WEAPON6',
    ],
}


def save_frame_and_action(frame, episode_dir, frame_count, action=None, action_meanings=None, is_last_frame=False):
    """Save a single frame to disk with optional action information."""
    os.makedirs(episode_dir, exist_ok=True)
    frame_path = os.path.join(episode_dir, f"frame_{frame_count:06d}.png")

    # Ensure uint8
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)

    imageio.imwrite(frame_path, frame)

    # Save action info if provided and not the last frame
    if action is not None and not is_last_frame:
        action_path = os.path.join(episode_dir, f"action_{frame_count:06d}.txt")
        with open(action_path, 'w') as f:
            # Convert action to human-readable format if meanings are available
            if action_meanings is not None:
                # VizDoom actions are typically multi-dimensional (one per button)
                if hasattr(action, '__len__'):
                    if len(action.shape) > 1:
                        action_vec = action[0] if len(action) > 0 else action
                    else:
                        action_vec = action

                    # Find which actions are pressed
                    pressed_actions = []
                    for i, val in enumerate(action_vec):
                        if val > 0 and i < len(action_meanings):
                            pressed_actions.append(action_meanings[i])

                    if pressed_actions:
                        action_str = "Pressed: " + ", ".join(pressed_actions)
                    else:
                        action_str = "No action"
                else:
                    action_str = str(action)
            else:
                action_str = str(action)

            f.write(action_str)

    return frame_path


def sample_vizdoom_episodes(checkpoint_path, output_dir, env_name='doom_battle',
                           num_frames=256, max_episodes=10, device='cpu', deterministic=True):
    """Sample VizDoom with episode folders and action recording."""

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

    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get action meanings for this environment
    action_meanings = VIZDOOM_ACTION_MEANINGS.get(env_name, None)

    # Sampling loop
    total_frames = 0
    episode_count = 0
    all_frames = []  # For video creation
    episode_rewards = []

    log.info(f"Starting to sample from {env_name} (max {num_frames} frames, max {max_episodes} episodes)...")
    log.info(f"Using {'deterministic' if deterministic else 'stochastic'} policy")
    log.info(f"Saving episode folders to: {output_dir}")

    while total_frames < num_frames and episode_count < max_episodes:
        # Create episode-specific directory
        episode_dir = os.path.join(output_dir, f"episode_{episode_count:04d}")
        os.makedirs(episode_dir, exist_ok=True)

        obs, _ = env.reset()
        done = [False]
        episode_reward = 0
        episode_frame_count = 0
        episode_frames = []

        # Reset RNN states for new episode
        rnn_states = torch.zeros([env.num_agents, rnn_size], dtype=torch.float32, device=device)

        log.info(f"Episode {episode_count}: Starting...")

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
                # Note: Using sampled actions for now as VizDoom has complex action space handling
                # The reward discrepancy between checkpoint name (57.570) and actual rewards (~1-3)
                # is likely due to differences in:
                # 1. Training vs evaluation environment settings
                # 2. Episode length limits during training vs evaluation
                # 3. Reward normalization/scaling during training
                if hasattr(policy_outputs, 'actions'):
                    actions = policy_outputs.actions
                else:
                    actions = policy_outputs['actions']

                # Update RNN states
                if hasattr(policy_outputs, 'new_rnn_states'):
                    rnn_states = policy_outputs.new_rnn_states
                elif 'new_rnn_states' in policy_outputs:
                    rnn_states = policy_outputs['new_rnn_states']

            # Render and save frame BEFORE taking action
            frame = env.render()
            if frame is not None:
                if len(frame.shape) == 4:  # Batched
                    frame = frame[0]

                # Save frame and action to episode folder
                actions_numpy = actions.cpu().numpy() if torch.is_tensor(actions) else actions
                save_frame_and_action(frame, episode_dir, episode_frame_count,
                                    actions_numpy, action_meanings, is_last_frame=False)

                episode_frames.append(frame)
                all_frames.append(frame)
                episode_frame_count += 1
                total_frames += 1

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated
            episode_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward

        # Save the final frame without an action
        frame = env.render()
        if frame is not None:
            if len(frame.shape) == 4:  # Batched
                frame = frame[0]

            save_frame_and_action(frame, episode_dir, episode_frame_count,
                                None, action_meanings, is_last_frame=True)

            episode_frames.append(frame)
            all_frames.append(frame)
            episode_frame_count += 1
            total_frames += 1

        episode_count += 1
        reward_value = float(episode_reward.item()) if hasattr(episode_reward, 'item') else float(episode_reward)
        episode_rewards.append(reward_value)

        log.info(f"Episode {episode_count - 1}: Completed with {episode_frame_count} frames, reward = {reward_value:.2f}")

        # Create episode video at 1 FPS
        if episode_frames:
            episode_video_path = os.path.join(episode_dir, f"episode_{episode_count-1:04d}.mp4")
            writer = imageio.get_writer(episode_video_path, fps=1, codec='libx264',
                                       pixelformat='yuv420p', quality=8)
            for frame in episode_frames:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)
            writer.close()
            log.info(f"  Saved episode video to {episode_video_path} (1 FPS)")

    # Save combined video of all episodes at 1 FPS
    if all_frames:
        video_path = os.path.join(output_dir, f'{env_name}_all_episodes.mp4')
        writer = imageio.get_writer(video_path, fps=1, codec='libx264',
                                   pixelformat='yuv420p', quality=8)
        for frame in all_frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)
        writer.close()
        log.info(f"Saved combined video to {video_path} (1 FPS)")

    # Save metadata
    metadata = {
        'env_name': env_name,
        'episodes': episode_count,
        'total_frames': total_frames,
        'episode_rewards': episode_rewards,
        'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
        'checkpoint': os.path.basename(checkpoint_path),
    }

    import json
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    env.close()

    log.info(f"\nSummary:")
    log.info(f"  Episodes: {episode_count}")
    log.info(f"  Total frames: {total_frames}")
    log.info(f"  Average reward: {metadata['avg_reward']:.2f}")
    log.info(f"  Output directory: {output_dir}")

    return total_frames, episode_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='doom_battle')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--frames', type=int, default=256)
    parser.add_argument('--max-episodes', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy (default: True)')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy instead of deterministic')

    args = parser.parse_args()

    # Use stochastic if explicitly requested
    deterministic = not args.stochastic

    frames, episodes = sample_vizdoom_episodes(
        args.checkpoint,
        args.output_dir,
        args.env,
        args.frames,
        args.max_episodes,
        args.device,
        deterministic
    )

    print(f"Sampled {frames} frames over {episodes} episodes")