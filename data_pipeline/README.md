# Sample Factory Data Pipeline

Clean and simplified pipeline for sampling frames and creating videos from trained models.

## Directory Structure

```
data_pipeline/
├── README.md                              # This file
├── README_PIPELINE.md                     # Detailed pipeline documentation
├── multi_checkpoint_pipeline_best.sh      # Main parallel pipeline for all Atari envs
├── sample_single_env.sh                   # Test single environment
├── vizdoom/                              # VizDoom-specific scripts
│   ├── download_vizdoom_models.sh
│   ├── run_vizdoom_sampling.sh
│   ├── sample_vizdoom_simple.py
│   └── vizdoom_pipeline.sh
└── old_scripts_backup/                    # Deprecated scripts (for reference)
```

## Quick Start

### Single Environment Test
```bash
# Test with Ms. Pac-Man using best checkpoint
./sample_single_env.sh atari_mspacman 3333

# Test with custom settings
MAX_EPISODES=5 MAX_FRAMES=1000 ./sample_single_env.sh atari_breakout 3333
```

### Full Pipeline (All Environments)
```bash
# Run with default settings (4 parallel jobs, 10 episodes each)
./multi_checkpoint_pipeline_best.sh

# Run with custom settings
PARALLEL_JOBS=8 MAX_EPISODES=20 ./multi_checkpoint_pipeline_best.sh
```

## Key Features

- **Simplified**: Direct use of Sample Factory's `enjoy_atari` module
- **Parallel Processing**: Process multiple environments simultaneously
- **Best Checkpoints**: Automatically uses best performing checkpoints
- **Clear Configuration**: All 57 Atari environments listed in script for easy editing
- **Video Creation**: Automatic video generation using ffmpeg

## Output Locations

- **Test outputs**: `/tmp/atari_samples/`
- **Production outputs**: `/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/`
  - Frames: `sampled_frames_best/`
  - Videos: `sampled_videos_best/`

See `README_PIPELINE.md` for detailed documentation.