# VizDoom Sampling Setup Guide

## Overview
This guide explains how to set up and use the VizDoom sampling capabilities with Sample Factory.

## Prerequisites

VizDoom requires several system dependencies to be installed before the Python package can be built:

```bash
# On Debian/Ubuntu systems:
sudo apt-get update
sudo apt-get install -y \
    cmake \
    libboost-all-dev \
    libsdl2-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libpng-dev \
    libjpeg-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    zlib1g-dev \
    timidity \
    tar \
    nasm

# Install VizDoom Python package
uv pip install vizdoom --pre
```

## Files Created

1. **sample_vizdoom.py** - Main sampling script for VizDoom environments
   - Samples frames from trained VizDoom models
   - Supports various VizDoom environments (doom_battle, doom_deathmatch, etc.)
   - Can save videos of sampled episodes
   - Configurable exploration parameters

2. **vizdoom_multi_checkpoint_pipeline.sh** - Batch processing pipeline
   - Processes multiple checkpoints across different training stages
   - Automatically detects VizDoom experiments in checkpoint directory
   - Creates videos from sampled frames
   - Optionally converts to parquet format

3. **test_vizdoom_sampling.sh** - Test script to verify functionality
   - Tests VizDoom environment creation
   - Tests sampling with random policy
   - Creates test frames and videos

## Usage Examples

### Single Environment Sampling

```bash
# Sample frames from a trained model
python3 data_pipeline/sample_vizdoom.py \
    --env doom_battle \
    --checkpoint /path/to/checkpoint.pth \
    --output-dir /path/to/output \
    --frames 256 \
    --max-episodes 50 \
    --device cpu \
    --randomness 0.2 \
    --epsilon-greedy 0.2 \
    --save-video \
    --video-name doom_battle_sample
```

### Batch Processing Multiple Checkpoints

```bash
# Process all VizDoom experiments in a checkpoint directory
VIZDOOM_CHECKPOINT_DIR=/path/to/checkpoints \
VIZDOOM_FRAMES_DIR=/path/to/frames \
FRAMES_PER_ENV=256 \
MAX_EPISODES=50 \
./data_pipeline/vizdoom_multi_checkpoint_pipeline.sh

# Process specific environments only
SPECIFIC_ENVS="doom_battle,doom_deathmatch" \
MAX_ENVS=2 \
./data_pipeline/vizdoom_multi_checkpoint_pipeline.sh
```

## Available VizDoom Environments

Sample Factory supports various VizDoom environments:

- **doom_basic** - Basic navigation and shooting
- **doom_battle** - Battle scenario with monsters
- **doom_battle2** - Advanced battle scenario
- **doom_duel** - 1v1 duel scenarios
- **doom_deathmatch** - Deathmatch multiplayer
- **doom_duel_bots** - Duel with bot opponents
- **doom_deathmatch_bots** - Deathmatch with bots
- **doom_my_way_home** - Navigation puzzle
- **doom_deadly_corridor** - Corridor navigation and combat
- **doom_defend_the_center** - Defensive scenario
- **doom_defend_the_line** - Line defense scenario
- **doom_health_gathering** - Health pickup collection
- **doom_health_gathering_supreme** - Advanced health gathering
- **doom_two_colors_easy** - Two colors gathering task
- **doom_two_colors_hard** - Harder two colors task

## Configuration Parameters

### Sampling Parameters
- `--frames`: Number of frames to sample per environment (default: 256)
- `--max-episodes`: Maximum episodes to run (default: 50)
- `--randomness`: Probability of random actions (0.0-1.0)
- `--epsilon-greedy`: Epsilon for epsilon-greedy exploration
- `--eval-env-frameskip`: Frame skip for evaluation (default: 4)

### Pipeline Environment Variables
- `VIZDOOM_CHECKPOINT_DIR`: Directory containing VizDoom checkpoints
- `VIZDOOM_FRAMES_DIR`: Output directory for sampled frames
- `VIZDOOM_PARQUET_DIR`: Output directory for parquet files
- `FRAMES_PER_ENV`: Frames to sample per environment
- `MAX_EPISODES`: Maximum episodes per checkpoint
- `CREATE_VIDEOS`: Whether to create videos (0 or 1)
- `VIDEO_FPS`: FPS for generated videos

## Integration with Existing Pipeline

The VizDoom sampling scripts are designed to be compatible with the existing Atari pipeline structure:

1. Same directory structure for outputs
2. Compatible parquet conversion (if process_individual_env.py supports it)
3. Similar checkpoint selection strategy (early, mid, late training stages)
4. Video creation from sampled frames

## Troubleshooting

### VizDoom Installation Issues
If VizDoom fails to install, ensure all system dependencies are installed:
```bash
# Check missing dependencies
cmake --version
pkg-config --modversion sdl2
ldconfig -p | grep boost
```

### Environment Creation Errors
If environments fail to create:
1. Verify VizDoom is properly installed: `python3 -c "import vizdoom"`
2. Check that Sample Factory is properly configured
3. Ensure checkpoint paths are correct

### Memory Issues
For large-scale sampling:
- Reduce `FRAMES_PER_ENV` and `MAX_EPISODES`
- Process environments sequentially instead of in parallel
- Use CPU instead of GPU for sampling (`--device cpu`)

## Next Steps

1. Install VizDoom dependencies (see Prerequisites)
2. Test the setup: `./data_pipeline/test_vizdoom_sampling.sh`
3. Start sampling from your trained models
4. Process multiple checkpoints with the pipeline script

## Additional Resources

- VizDoom Documentation: https://vizdoom.farama.org/
- Sample Factory VizDoom Guide: https://www.samplefactory.dev/09-environment-integrations/vizdoom/
- Training VizDoom agents: Use `python -m sf_examples.vizdoom.train_vizdoom`