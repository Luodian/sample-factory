#!/bin/bash

# Sample a single VizDoom environment with best checkpoint
# Usage: ./sample_single_vizdoom.sh doom_battle2

ENV_NAME=${1:-doom_battle2}
MODEL_DIR="${VIZDOOM_MODEL_DIR:-/mnt/bn/seed-aws-va/brianli/prod/contents/vizdoom_models}"
OUTPUT_BASE="${VIZDOOM_OUTPUT_BASE:-/mnt/bn/seed-aws-va/brianli/prod/contents/vizdoom}"
FRAMES_DIR="${OUTPUT_BASE}/sampled_frames_best/${ENV_NAME}"
VIDEO_DIR="${OUTPUT_BASE}/sampled_videos_best"
FRAMES_PER_ENV="${FRAMES_PER_ENV:-512}"
MAX_EPISODES="${MAX_EPISODES:-10}"
DEVICE="${DEVICE:-cpu}"

echo "=== Sampling single VizDoom environment: $ENV_NAME ==="

# Find best checkpoint
if [ "$ENV_NAME" == "doom_battle" ]; then
    CHECKPOINT="$MODEL_DIR/doom_battle/checkpoint_p0/best_000433355_3550044160_reward_57.570.pth"
elif [ "$ENV_NAME" == "doom_battle2" ]; then
    # Use the latest checkpoint for doom_battle2 (no best_ file available)
    CHECKPOINT="$MODEL_DIR/doom_battle2/checkpoint_p0/checkpoint_000402752_3299344384.pth"
elif [ "$ENV_NAME" == "doom_deathmatch_bots" ]; then
    CHECKPOINT="$MODEL_DIR/doom_deathmatch_bots/checkpoint_p0/checkpoint_000548577_2246971392.pth"
elif [ "$ENV_NAME" == "doom_duel_bots" ]; then
    # Has multiple best checkpoints, use the one with highest reward
    CHECKPOINT="$MODEL_DIR/doom_duel_bots/checkpoint_p1/best_000358179_1450700800_reward_493.143.pth"
elif [ "$ENV_NAME" == "doom_duel_selfplay" ]; then
    # Use best from checkpoint_p5 (highest reward)
    CHECKPOINT="$MODEL_DIR/doom_duel_selfplay/checkpoint_p5/best_000387727_1567629312_reward_370.568.pth"
else
    echo "ERROR: Unknown environment $ENV_NAME"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Using checkpoint: $(basename $CHECKPOINT)"
echo "Output directory: $FRAMES_DIR"
echo "Frames: $FRAMES_PER_ENV, Episodes: $MAX_EPISODES"

# Create directories
mkdir -p "$FRAMES_DIR"
mkdir -p "$VIDEO_DIR"

# Run sampling
cd /opt/tiger/sample-factory/data_pipeline/vizdoom
python3 sample_vizdoom_episodes.py \
    --env "$ENV_NAME" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$FRAMES_DIR" \
    --frames "$FRAMES_PER_ENV" \
    --max-episodes "$MAX_EPISODES" \
    --device "$DEVICE"

# Move video to central location if it exists
if [ -f "$FRAMES_DIR/${ENV_NAME}_all_episodes.mp4" ]; then
    mv "$FRAMES_DIR/${ENV_NAME}_all_episodes.mp4" "$VIDEO_DIR/${ENV_NAME}.mp4"
    echo "Video saved to: $VIDEO_DIR/${ENV_NAME}.mp4"
fi

# Show summary
echo ""
echo "=== Summary ==="
if [ -d "$FRAMES_DIR" ]; then
    episode_count=$(find "$FRAMES_DIR" -type d -name "episode_*" 2>/dev/null | wc -l)
    frame_count=$(find "$FRAMES_DIR" -name "frame_*.png" 2>/dev/null | wc -l)
    action_count=$(find "$FRAMES_DIR" -name "action_*.txt" 2>/dev/null | wc -l)
    echo "Episodes: $episode_count"
    echo "Frames: $frame_count"
    echo "Actions: $action_count"
fi

if [ -f "$FRAMES_DIR/metadata.json" ]; then
    echo ""
    echo "Metadata:"
    cat "$FRAMES_DIR/metadata.json"
fi