#!/bin/bash

# Simple script to sample frames and create video for a single environment
# Usage: ./sample_single_env.sh <env_name> [suffix]
# Example: ./sample_single_env.sh atari_mspacman 3333

ENV_NAME=${1:-"atari_mspacman"}
SUFFIX=${2:-"3333"}

CHECKPOINT_DIR="/opt/tiger/sample-factory/checkpoints"
OUTPUT_DIR="/tmp/atari_samples"
FRAMES_DIR="${OUTPUT_DIR}/frames_${ENV_NAME}"
VIDEO_PATH="${OUTPUT_DIR}/${ENV_NAME}.mp4"

# Parameters
MAX_EPISODES=${MAX_EPISODES:-1}
MAX_FRAMES=${MAX_FRAMES:-240}
FPS=${FPS:-30}
DEVICE=${DEVICE:-"cpu"}

EXPERIMENT="edbeeching_atari_2B_${ENV_NAME}_${SUFFIX}"

echo "=== Single Environment Sampling ==="
echo "Environment: $ENV_NAME"
echo "Checkpoint: $EXPERIMENT"
echo "Frames dir: $FRAMES_DIR"
echo "Output video: $VIDEO_PATH"
echo ""

# Clean and create directories
rm -rf "$FRAMES_DIR"
mkdir -p "$FRAMES_DIR"
mkdir -p "$OUTPUT_DIR"

# Check if checkpoint exists
if [ ! -d "${CHECKPOINT_DIR}/${EXPERIMENT}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT_DIR}/${EXPERIMENT}"
    exit 1
fi

# Sample frames
echo "Sampling frames..."
python -m sf_examples.atari.enjoy_atari \
    --env "$ENV_NAME" \
    --experiment "$EXPERIMENT" \
    --train_dir "$CHECKPOINT_DIR" \
    --device "$DEVICE" \
    --save_frames \
    --frames_dir "$FRAMES_DIR" \
    --max_num_frames "$MAX_FRAMES" \
    --max_num_episodes "$MAX_EPISODES" \
    --no_render \
    --load_checkpoint_kind best

# Check if frames were generated
FRAME_COUNT=$(find "$FRAMES_DIR" -name "*.png" 2>/dev/null | wc -l)
echo "Generated $FRAME_COUNT frames"

if [ "$FRAME_COUNT" -eq 0 ]; then
    echo "ERROR: No frames were generated"
    exit 1
fi

# Create video
echo "Creating video at $FPS fps..."
ffmpeg -y -framerate "$FPS" \
    -pattern_type glob -i "${FRAMES_DIR}/*.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 23 \
    "$VIDEO_PATH"

if [ -f "$VIDEO_PATH" ]; then
    VIDEO_SIZE=$(du -h "$VIDEO_PATH" | cut -f1)
    echo ""
    echo "Success! Video created:"
    echo "  Path: $VIDEO_PATH"
    echo "  Size: $VIDEO_SIZE"
    echo "  Frames: $FRAME_COUNT"
    echo ""
    echo "To play the video: ffplay $VIDEO_PATH"
else
    echo "ERROR: Failed to create video"
    exit 1
fi