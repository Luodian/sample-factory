#!/bin/bash

# Quick script to sample a few frames from each environment
# This is a faster version for testing/preview purposes

echo "Quick sampling from all Atari environments..."
echo "This will sample 60 frames total (20 per episode, 3 episodes) from each environment"
echo "Episodes are saved in separate folders with randomness level in the name"
echo ""

# Use the Python script with optimized settings
python3 sample_all_envs.py \
    --frames-per-env 16 \
    --max-episodes 3 \
    --randomness 0.2 \
    --parallel 8 \
    --device cpu \
    --seed 1111 \
    --output-dir sampled_frames_organized