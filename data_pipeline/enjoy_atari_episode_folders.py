import os
import sys
import time
from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import BatchedVecEnv, make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import experiment_dir, log

from sf_examples.atari.train_atari import parse_atari_args, register_atari_components


def save_frame_to_disk(frame, episode_dir, frame_count, action=None, action_meanings=None, is_last_frame=False):
    """Save a single frame to disk with optional action information."""
    os.makedirs(episode_dir, exist_ok=True)
    frame_path = os.path.join(episode_dir, f"frame_{frame_count:06d}.png")

    # Convert frame format if needed
    if frame.shape[0] == 3:  # CHW format
        frame = frame.transpose(1, 2, 0)  # Convert to HWC

    # Convert RGB to BGR for cv2
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(frame_path, frame_bgr)

    # Save action info if provided and not the last frame
    if action is not None and not is_last_frame:
        action_path = os.path.join(episode_dir, f"action_{frame_count:06d}.txt")
        with open(action_path, 'w') as f:
            # Convert action to human-readable format if meanings are available
            if action_meanings is not None:
                if hasattr(action, '__len__'):
                    if len(action.shape) > 0:
                        action_idx = int(action[0]) if len(action) > 0 else 0
                    else:
                        action_idx = int(action)
                else:
                    action_idx = int(action)

                if 0 <= action_idx < len(action_meanings):
                    action_str = f"{action_idx}: {action_meanings[action_idx]}"
                else:
                    action_str = f"{action_idx}: UNKNOWN"
            else:
                action_str = str(action)

            f.write(action_str)

    return frame_path


def render_frame(cfg, env, episode_dir, episode_frame_count, action=None, action_meanings=None, is_last_frame=False):
    """Render and save a frame to the episode-specific directory."""
    if cfg.save_frames:
        frame = env.render()
        if frame is not None:
            save_frame_to_disk(frame, episode_dir, episode_frame_count, action, action_meanings, is_last_frame)


def make_env(cfg: Config, render_mode: Optional[str] = None) -> BatchedVecEnv:
    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )
    return env


def load_state_dict(cfg: Config, actor_critic: ActorCritic, device: torch.device) -> None:
    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    if checkpoint_dict:
        actor_critic.load_state_dict(checkpoint_dict["model"])
    else:
        raise RuntimeError("Could not load checkpoint")


def enjoy(cfg: Config) -> Tuple[StatusCode, float]:
    verbose = False

    cfg = load_from_checkpoint(cfg)

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1

    render_mode = "rgb_array" if cfg.save_frames else None

    env = make_env(cfg, render_mode=render_mode)
    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    load_state_dict(cfg, actor_critic, device)

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    def max_frames_reached(frames):
        return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    reward_list = []

    # Setup base frames directory if saving frames
    base_frames_dir = None
    action_meanings = None
    if cfg.save_frames:
        base_frames_dir = os.path.join(experiment_dir(cfg=cfg), cfg.frames_dir)
        os.makedirs(base_frames_dir, exist_ok=True)
        log.info(f"Saving frames to {base_frames_dir}")

        # Try to get action meanings for human-readable output
        try:
            if hasattr(env.unwrapped, 'get_action_meanings'):
                action_meanings = env.unwrapped.get_action_meanings()
                log.info(f"Action meanings: {action_meanings}")
        except Exception as e:
            log.debug(f"Could not get action meanings: {e}")

    num_episodes = 0
    total_frame_count = 0

    while num_episodes < cfg.max_num_episodes and not max_frames_reached(num_frames):
        # Reset for new episode
        obs, infos = env.reset()
        action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
        rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
        episode_reward = torch.zeros(env.num_agents, dtype=torch.float32)
        finished_episode = [False for _ in range(env.num_agents)]

        # Create episode-specific directory
        episode_dir = None
        if cfg.save_frames:
            episode_dir = os.path.join(base_frames_dir, f"episode_{num_episodes:04d}")
            os.makedirs(episode_dir, exist_ok=True)
            log.info(f"Starting episode {num_episodes}, saving to {episode_dir}")

        episode_frame_count = 0

        with torch.no_grad():
            episode_done = False
            while not episode_done and not max_frames_reached(num_frames):
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)

                # sample actions from the distribution by default
                actions = policy_outputs["actions"]

                if cfg.eval_deterministic:
                    action_distribution = actor_critic.action_distribution()
                    actions = argmax_actions(action_distribution)

                # actions shape should be [num_agents, num_actions] even if it's [1, 1]
                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                actions = preprocess_actions(env_info, actions)

                rnn_states = policy_outputs["new_rnn_states"]

                for _ in range(render_action_repeat):
                    # Convert actions to numpy array for saving
                    actions_numpy = actions.cpu().numpy() if torch.is_tensor(actions) else actions

                    # Check if this will be the last frame
                    is_last = cfg.max_num_frames is not None and num_frames >= cfg.max_num_frames

                    # Save the current frame and action
                    if cfg.save_frames and episode_dir:
                        render_frame(cfg, env, episode_dir, episode_frame_count, actions_numpy, action_meanings, is_last)

                    episode_frame_count += 1
                    total_frame_count += 1

                    # Execute the action to transition to next state
                    obs, rew, terminated, truncated, infos = env.step(actions)
                    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
                    dones = make_dones(terminated, truncated)
                    infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                    episode_reward += rew.float()
                    num_frames += 1

                    if num_frames % 100 == 0:
                        log.debug(f"Num frames {num_frames}...")

                    dones = dones.cpu().numpy()
                    for agent_i, done_flag in enumerate(dones):
                        if done_flag:
                            finished_episode[agent_i] = True
                            rew_val = episode_reward[agent_i].item()
                            episode_rewards[agent_i].append(rew_val)

                            true_objective = rew_val
                            if isinstance(infos, (list, tuple)):
                                true_objective = infos[agent_i].get("true_objective", rew_val)
                            true_objectives[agent_i].append(true_objective)

                            if verbose:
                                log.info(
                                    "Episode %d finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
                                    num_episodes,
                                    agent_i,
                                    num_frames,
                                    episode_reward[agent_i],
                                    true_objectives[agent_i][-1],
                                )

                            reward_list.append(true_objective)

                    # if episode terminated synchronously for all agents, save the final frame without action
                    if all(dones):
                        # Save the final frame without an action
                        if cfg.save_frames and episode_dir:
                            render_frame(cfg, env, episode_dir, episode_frame_count, None, action_meanings, is_last_frame=True)
                        episode_frame_count += 1
                        episode_done = True
                        break

                    if all(finished_episode):
                        episode_done = True
                        break

        # Episode complete
        num_episodes += 1
        log.info(f"Episode {num_episodes - 1} complete: {episode_frame_count} frames saved, reward: {episode_reward[0]:.3f}")

    env.close()

    if cfg.save_frames:
        log.info(f"Saved {num_episodes} episodes with {total_frame_count} total frames to {base_frames_dir}")

    total_episodes = sum([len(episode_rewards[i]) for i in range(env.num_agents)])
    if total_episodes > 0:
        avg_reward = sum([sum(episode_rewards[i]) for i in range(env.num_agents)]) / total_episodes
    else:
        avg_reward = 0.0
    return ExperimentStatus.SUCCESS, avg_reward


def main():
    """Script entry point."""
    register_atari_components()
    cfg = parse_atari_args(evaluation=True)

    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())