import os
import time
from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import gymnasium as gym
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
from sample_factory.huggingface.huggingface_utils import generate_model_card, generate_replay_video, push_to_hf
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log


def visualize_policy_inputs(normalized_obs: Dict[str, Tensor]) -> None:
    """
    Display actual policy inputs after all wrappers and normalizations using OpenCV imshow.
    """
    import cv2

    if "obs" not in normalized_obs.keys():
        return

    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent
    obs = obs[0]
    if obs.dim() != 3:
        # this function is only for RGB images
        return

    # convert to HWC
    obs = obs.permute(1, 2, 0)
    # convert to numpy
    obs = obs.cpu().numpy()
    # convert to uint8
    obs = cv2.normalize(
        obs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1
    )  # this will be different frame-by-frame but probably good enough to give us an idea?
    scale = 5
    obs = cv2.resize(obs, (obs.shape[1] * scale, obs.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("policy inputs", obs)
    cv2.waitKey(delay=1)


def save_frame_to_disk(frame, frames_dir, frame_count, action=None, action_meanings=None):
    """Save a single frame to disk with optional action information."""
    os.makedirs(frames_dir, exist_ok=True)
    frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
    
    # Convert frame format if needed
    if frame.shape[0] == 3:  # CHW format
        frame = frame.transpose(1, 2, 0)  # Convert to HWC
    
    # Convert RGB to BGR for cv2
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(frame_path, frame_bgr)
    
    # Save action info if provided
    if action is not None:
        action_path = os.path.join(frames_dir, f"action_{frame_count:06d}.txt")
        with open(action_path, 'w') as f:
            # Convert action to human-readable format if meanings are available
            if action_meanings is not None:
                # Handle both single actions and batch actions
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

def render_frame(cfg, env, video_frames, num_episodes, last_render_start, frame_count=0, frames_dir=None, action=None, action_meanings=None) -> float:
    render_start = time.time()

    if cfg.save_video or cfg.save_frames:
        need_video_frame = len(video_frames) < cfg.video_frames or cfg.video_frames < 0 and num_episodes == 0
        if need_video_frame:
            frame = env.render()
            if frame is not None:
                if cfg.save_frames:
                    # Save individual frame to disk
                    save_frame_to_disk(frame, frames_dir, frame_count, action, action_meanings)
                if cfg.save_video:
                    # Append to video frames list
                    video_frames.append(frame.copy())
    else:
        if not cfg.no_render:
            target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
            current_delay = render_start - last_render_start
            time_wait = target_delay - current_delay

            if time_wait > 0:
                # log.info("Wait time %.3f", time_wait)
                time.sleep(time_wait)

            try:
                env.render()
            except (gym.error.Error, TypeError) as ex:
                debug_log_every_n(1000, f"Exception when calling env.render() {str(ex)}")

    return render_start


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
    
    # Add epsilon-greedy exploration parameter
    epsilon = getattr(cfg, 'epsilon_greedy', 0.0)
    if epsilon > 0:
        log.info(f"Using epsilon-greedy exploration with epsilon={epsilon}")

    cfg.num_envs = 1

    render_mode = "human"
    if cfg.save_video or cfg.save_frames:
        render_mode = "rgb_array"
    elif cfg.no_render:
        render_mode = None

    env = make_env(cfg, render_mode=render_mode)
    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    load_state_dict(cfg, actor_critic, device)

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    reward_list = []

    obs, infos = env.reset()
    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]

    video_frames = []
    num_episodes = 0
    frame_count = 0
    
    # Setup frames directory if saving frames
    frames_dir = None
    action_meanings = None
    if cfg.save_frames:
        frames_dir = os.path.join(experiment_dir(cfg=cfg), cfg.frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        log.info(f"Saving frames to {frames_dir}")
        
        # Try to get action meanings for human-readable output
        try:
            if hasattr(env.unwrapped, 'get_action_meanings'):
                action_meanings = env.unwrapped.get_action_meanings()
                log.info(f"Action meanings: {action_meanings}")
        except Exception as e:
            log.debug(f"Could not get action meanings: {e}")

    with torch.no_grad():
        while not max_frames_reached(num_frames):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

            if not cfg.no_render:
                visualize_policy_inputs(normalized_obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)

            # sample actions from the distribution by default
            actions = policy_outputs["actions"]

            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)
            
            # Apply epsilon-greedy exploration
            epsilon = getattr(cfg, 'epsilon_greedy', 0.0)
            if epsilon > 0 and np.random.random() < epsilon:
                # Take random action
                if hasattr(env.action_space, 'n'):
                    # Discrete action space
                    actions = torch.tensor([[np.random.randint(0, env.action_space.n)]], dtype=torch.long, device=device)
                else:
                    # Continuous action space
                    actions = torch.tensor([env.action_space.sample()], dtype=torch.float32, device=device)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            for _ in range(render_action_repeat):
                # Convert actions to numpy array for saving
                actions_numpy = actions.cpu().numpy() if torch.is_tensor(actions) else actions
                
                # Save the current frame (state before action)
                last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start, frame_count, frames_dir, None, action_meanings)
                frame_count += 1
                
                # Save the action that will be executed from this frame
                if cfg.save_frames and frames_dir is not None:
                    action_path = os.path.join(frames_dir, f"action_{frame_count-1:06d}.txt")
                    with open(action_path, 'w') as f:
                        if action_meanings is not None and hasattr(actions_numpy, '__len__'):
                            action_idx = int(actions_numpy[0]) if len(actions_numpy) > 0 else 0
                            if 0 <= action_idx < len(action_meanings):
                                action_str = f"{action_idx}: {action_meanings[action_idx]}"
                            else:
                                action_str = f"{action_idx}: UNKNOWN"
                        else:
                            action_str = str(actions_numpy)
                        f.write(action_str)
                
                # Execute the action to transition to next state
                obs, rew, terminated, truncated, infos = env.step(actions)
                action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
                dones = make_dones(terminated, truncated)
                infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                if episode_reward is None:
                    episode_reward = rew.float().clone()
                else:
                    episode_reward += rew.float()

                num_frames += 1
                if num_frames % 100 == 0:
                    log.debug(f"Num frames {num_frames}...")

                dones = dones.cpu().numpy()
                for agent_i, done_flag in enumerate(dones):
                    if done_flag:
                        finished_episode[agent_i] = True
                        rew = episode_reward[agent_i].item()
                        episode_rewards[agent_i].append(rew)

                        true_objective = rew
                        if isinstance(infos, (list, tuple)):
                            true_objective = infos[agent_i].get("true_objective", rew)
                        true_objectives[agent_i].append(true_objective)

                        if verbose:
                            log.info(
                                "Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
                                agent_i,
                                num_frames,
                                episode_reward[agent_i],
                                true_objectives[agent_i][-1],
                            )
                        rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                        episode_reward[agent_i] = 0

                        if cfg.use_record_episode_statistics:
                            # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                            if "episode" in infos[agent_i].keys():
                                num_episodes += 1
                                reward_list.append(infos[agent_i]["episode"]["r"])
                        else:
                            num_episodes += 1
                            reward_list.append(true_objective)

                # if episode terminated synchronously for all agents, save the final frame
                if all(dones):
                    # Save the final frame (result of the last action)
                    render_frame(cfg, env, video_frames, num_episodes, last_render_start, frame_count, frames_dir, None, action_meanings)
                    frame_count += 1
                    time.sleep(0.05)

                if all(finished_episode):
                    finished_episode = [False] * env.num_agents
                    avg_episode_rewards_str, avg_true_objective_str = "", ""
                    for agent_i in range(env.num_agents):
                        avg_rew = np.mean(episode_rewards[agent_i])
                        avg_true_obj = np.mean(true_objectives[agent_i])

                        if not np.isnan(avg_rew):
                            if avg_episode_rewards_str:
                                avg_episode_rewards_str += ", "
                            avg_episode_rewards_str += f"#{agent_i}: {avg_rew:.3f}"
                        if not np.isnan(avg_true_obj):
                            if avg_true_objective_str:
                                avg_true_objective_str += ", "
                            avg_true_objective_str += f"#{agent_i}: {avg_true_obj:.3f}"

                    log.info(
                        "Avg episode rewards: %s, true rewards: %s", avg_episode_rewards_str, avg_true_objective_str
                    )
                    log.info(
                        "Avg episode reward: %.3f, avg true_objective: %.3f",
                        np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]),
                        np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]),
                    )

                # VizDoom multiplayer stuff
                # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                #     key = f'PLAYER{player}_FRAGCOUNT'
                #     if key in infos[0]:
                #         log.debug('Score for player %d: %r', player, infos[0][key])

            if num_episodes >= cfg.max_num_episodes:
                break

    env.close()

    if cfg.save_video:
        if cfg.fps > 0:
            fps = cfg.fps
        else:
            fps = 30
        generate_replay_video(experiment_dir(cfg=cfg), video_frames, fps, cfg)
    
    if cfg.save_frames:
        log.info(f"Saved {frame_count} frames to {frames_dir}")

    if cfg.push_to_hub:
        generate_model_card(
            experiment_dir(cfg=cfg),
            cfg.algo,
            cfg.env,
            cfg.hf_repository,
            reward_list,
            cfg.enjoy_script,
            cfg.train_script,
        )
        push_to_hf(experiment_dir(cfg=cfg), cfg.hf_repository)

    total_episodes = sum([len(episode_rewards[i]) for i in range(env.num_agents)])
    if total_episodes > 0:
        avg_reward = sum([sum(episode_rewards[i]) for i in range(env.num_agents)]) / total_episodes
    else:
        avg_reward = 0.0
    return ExperimentStatus.SUCCESS, avg_reward
