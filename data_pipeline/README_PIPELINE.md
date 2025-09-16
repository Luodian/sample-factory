# Atari Sampling Pipeline

## Overview
Simplified pipeline for sampling frames from Atari environments using their best checkpoints and creating videos.

## Scripts

### 1. `sample_single_env.sh`
Test script for sampling a single environment.

**Usage:**
```bash
./sample_single_env.sh <env_name> [suffix]

# Example:
./sample_single_env.sh atari_mspacman 3333
```

**Parameters:**
- `MAX_EPISODES`: Number of episodes to sample (default: 1)
- `MAX_FRAMES`: Maximum frames to sample (default: 240)
- `FPS`: Video framerate (default: 30)
- `DEVICE`: CPU or CUDA device (default: cpu)

**Output:**
- Frames: `/tmp/atari_samples/frames_<env_name>/`
- Video: `/tmp/atari_samples/<env_name>.mp4`

### 2. `multi_checkpoint_pipeline_best.sh`
Main pipeline for processing multiple environments in parallel using their best checkpoints.

**Usage:**
```bash
# Run with default settings (4 parallel jobs)
./multi_checkpoint_pipeline_best.sh

# Custom parallel jobs and episodes
PARALLEL_JOBS=8 MAX_EPISODES=5 ./multi_checkpoint_pipeline_best.sh
```

**Parameters:**
- `PARALLEL_JOBS`: Number of environments to process in parallel (default: 4)
- `MAX_EPISODES`: Episodes per environment (default: 10)
- `MAX_FRAMES`: Maximum frames per environment (default: 2400)
- `FPS`: Video framerate (default: 30)

**Output:**
- Frames: `/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/sampled_frames_best/`
- Videos: `/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/sampled_videos_best/`

## Environment Configuration

The `multi_checkpoint_pipeline_best.sh` script contains a list of all 57 Atari environments. You can:

1. **Comment out environments** you don't want to process:
```bash
declare -a ENV_CONFIGS=(
    "atari_alien:3333"
    # "atari_amidar:3333"  # Skip this one
    "atari_assault:3333"
    ...
)
```

2. **Change checkpoint versions** (1111, 2222, or 3333):
```bash
"atari_mspacman:3333"  # Use best checkpoint
"atari_breakout:2222"  # Use middle checkpoint
"atari_pong:1111"      # Use early checkpoint
```

## Available Environments

All 57 Atari environments are configured in the script:
- atari_alien, atari_amidar, atari_assault, atari_asterix, atari_asteroid
- atari_atlantis, atari_bankheist, atari_battlezone, atari_beamrider
- atari_berzerk, atari_bowling, atari_boxing, atari_breakout
- atari_centipede, atari_choppercommand, atari_crazyclimber
- atari_defender, atari_demonattack, atari_doubledunk, atari_enduro
- atari_fishingderby, atari_freeway, atari_frostbite, atari_gopher
- atari_gravitar, atari_hero, atari_icehockey, atari_jamesbond
- atari_kangaroo, atari_kongfumaster, atari_krull, atari_montezuma
- **atari_mspacman**, atari_namethisgame, atari_phoenix, atari_pitfall
- atari_pong, atari_privateye, atari_qbert, atari_riverraid
- atari_roadrunner, atari_robotank, atari_seaquest, atari_skiing
- atari_solaris, atari_spaceinvaders, atari_stargunner, atari_surround
- atari_tennis, atari_timepilot, atari_tutankham, atari_upndown
- atari_venture, atari_videopinball, atari_wizardofwor
- atari_yarsrevenge, atari_zaxxon

## Checkpoint Suffixes

- **1111**: Early training checkpoint
- **2222**: Middle training checkpoint
- **3333**: Best/final training checkpoint (recommended)

## Example Commands

```bash
# Test single environment
./sample_single_env.sh atari_mspacman 3333

# Run full pipeline with 8 parallel jobs
PARALLEL_JOBS=8 ./multi_checkpoint_pipeline_best.sh

# Sample 5 episodes per environment with 4 parallel jobs
MAX_EPISODES=5 PARALLEL_JOBS=4 ./multi_checkpoint_pipeline_best.sh
```

## Notes

- The pipeline automatically uses the `best` checkpoint for each environment
- Videos are created using ffmpeg with H.264 encoding
- Frame sampling uses CPU by default (set `DEVICE=cuda` for GPU)
- Each environment's frames are saved before video creation