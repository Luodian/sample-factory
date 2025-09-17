#!/bin/bash

# VizDoom Best Checkpoint Sampling Pipeline
# Uses the best performing checkpoints for each environment

set -e

# Configuration
MODEL_DIR="${VIZDOOM_MODEL_DIR:-/mnt/bn/seed-aws-va/brianli/prod/contents/vizdoom_models}"
OUTPUT_BASE="${VIZDOOM_OUTPUT_BASE:-/mnt/bn/seed-aws-va/brianli/prod/contents/vizdoom}"
FRAMES_DIR="${OUTPUT_BASE}/sampled_frames_best"
VIDEO_DIR="${OUTPUT_BASE}/sampled_videos_best"

# Sampling parameters
FRAMES_PER_ENV="${FRAMES_PER_ENV:-512}"
MAX_EPISODES="${MAX_EPISODES:-10}"
DEVICE="${DEVICE:-cpu}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
FPS="${FPS:-1}"  # 1 FPS for slow playback as requested

echo "=== VizDoom Best Checkpoint Sampling Pipeline ==="
echo "Model directory: $MODEL_DIR"
echo "Frames output: $FRAMES_DIR"
echo "Videos output: $VIDEO_DIR"
echo "Frames per environment: $FRAMES_PER_ENV"
echo "Max episodes: $MAX_EPISODES"
echo "Device: $DEVICE"
echo "Parallel jobs: $PARALLEL_JOBS"
echo ""

# Create output directories
mkdir -p "$FRAMES_DIR"
mkdir -p "$VIDEO_DIR"

# Check if models directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Please ensure VizDoom models are downloaded first."
    exit 1
fi

# Define environments and find their best checkpoints
declare -A ENV_BEST_CHECKPOINTS

# Function to find best checkpoint for an environment
find_best_checkpoint() {
    local env_name=$1
    local model_base_dir="$MODEL_DIR/$env_name"

    if [ ! -d "$model_base_dir" ]; then
        echo ""
        return
    fi

    # Look for best_*.pth files in checkpoint subdirectories
    local best_checkpoint=$(find "$model_base_dir" -name "best_*.pth" -type f 2>/dev/null | head -1)

    # If no best checkpoint, look for any .pth file
    if [ -z "$best_checkpoint" ]; then
        best_checkpoint=$(find "$model_base_dir" -name "*.pth" -type f 2>/dev/null | head -1)
    fi

    echo "$best_checkpoint"
}

# Find best checkpoints for each environment
echo "Finding best checkpoints..."
for env in doom_battle doom_battle2 doom_deathmatch_bots doom_duel_bots doom_duel_selfplay; do
    checkpoint=$(find_best_checkpoint "$env")
    if [ -n "$checkpoint" ]; then
        ENV_BEST_CHECKPOINTS["$env"]="$checkpoint"
        echo "  $env: Found $(basename $checkpoint)"
    else
        echo "  $env: No checkpoint found"
    fi
done
echo ""

# Function to process a single environment
process_environment() {
    local env_name=$1
    local checkpoint_path=$2
    local env_frames_dir="${FRAMES_DIR}/${env_name}"
    local env_video_path="${VIDEO_DIR}/${env_name}.mp4"

    echo "[$(date '+%H:%M:%S')] Processing $env_name with best checkpoint"
    echo "[$(date '+%H:%M:%S')] Using: $(basename $checkpoint_path)"

    # Check if checkpoint exists
    if [ ! -f "$checkpoint_path" ]; then
        echo "[$(date '+%H:%M:%S')] ERROR: Checkpoint not found at $checkpoint_path"
        return 1
    fi

    # Clean up any existing frames for this env
    rm -rf "$env_frames_dir"
    mkdir -p "$env_frames_dir"

    # Run sampling with the best checkpoint (saves episodes to separate folders)
    echo "[$(date '+%H:%M:%S')] Sampling frames for $env_name..."

    cd /opt/tiger/sample-factory/data_pipeline/vizdoom
    python3 sample_vizdoom_episodes.py \
        --env "$env_name" \
        --checkpoint "$checkpoint_path" \
        --output-dir "$env_frames_dir" \
        --frames "$FRAMES_PER_ENV" \
        --max-episodes "$MAX_EPISODES" \
        --device "$DEVICE" \
        2>&1 | sed "s/^/[$env_name] /"

    local sampling_result=$?
    cd - > /dev/null

    if [ $sampling_result -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Successfully sampled frames for $env_name"

        # Count frames and episodes
        local episode_count=$(find "$env_frames_dir" -type d -name "episode_*" 2>/dev/null | wc -l)
        local frame_count=$(find "$env_frames_dir" -name "frame_*.png" 2>/dev/null | wc -l)
        local action_count=$(find "$env_frames_dir" -name "action_*.txt" 2>/dev/null | wc -l)
        echo "[$(date '+%H:%M:%S')] Generated $frame_count frames and $action_count actions across $episode_count episodes"

        # Move combined video to centralized video directory if it exists
        if [ -f "$env_frames_dir/${env_name}_all_episodes.mp4" ]; then
            mv "$env_frames_dir/${env_name}_all_episodes.mp4" "$env_video_path"
            local video_size=$(du -h "$env_video_path" | cut -f1)
            echo "[$(date '+%H:%M:%S')] Created video: ${env_name}.mp4 ($video_size) at 1 FPS"
        fi
    else
        echo "[$(date '+%H:%M:%S')] ERROR: Failed to sample frames for $env_name"
        return 1
    fi

    echo "[$(date '+%H:%M:%S')] Completed $env_name"
    return 0
}

# Export function and variables for parallel execution
export -f process_environment find_best_checkpoint
export MODEL_DIR FRAMES_DIR VIDEO_DIR FRAMES_PER_ENV MAX_EPISODES DEVICE FPS

# Start processing
START_TIME=$(date +%s)

echo "Starting processing of ${#ENV_BEST_CHECKPOINTS[@]} environments..."
echo "=========================================="

# Process environments
if [ "$PARALLEL_JOBS" -gt 1 ] && command -v parallel &> /dev/null; then
    echo "Processing in parallel with $PARALLEL_JOBS jobs..."

    # Create job list
    job_file=$(mktemp)
    for env_name in "${!ENV_BEST_CHECKPOINTS[@]}"; do
        echo "$env_name ${ENV_BEST_CHECKPOINTS[$env_name]}" >> "$job_file"
    done

    # Run in parallel
    cat "$job_file" | parallel --colsep ' ' -j "$PARALLEL_JOBS" process_environment {1} {2}

    rm "$job_file"
else
    echo "Processing sequentially..."

    success_count=0
    total_count=${#ENV_BEST_CHECKPOINTS[@]}

    for env_name in "${!ENV_BEST_CHECKPOINTS[@]}"; do
        checkpoint_path="${ENV_BEST_CHECKPOINTS[$env_name]}"

        process_environment "$env_name" "$checkpoint_path"

        if [ $? -eq 0 ]; then
            ((success_count++))
        fi
        echo ""
    done

    echo "Successfully processed: $success_count/$total_count environments"
fi

# Summary statistics
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "Pipeline completed in $((DURATION / 60))m $((DURATION % 60))s"
echo ""
echo "Output locations:"
echo "  Frames: $FRAMES_DIR"
echo "  Videos: $VIDEO_DIR"
echo ""

# Generate detailed summary report
echo "Generating summary report..."

python3 -c "
import os
import json
from glob import glob

frames_dir = '${FRAMES_DIR}'
video_dir = '${VIDEO_DIR}'
report_file = os.path.join('${OUTPUT_BASE}', 'best_checkpoint_report.txt')

with open(report_file, 'w') as f:
    f.write('VizDoom Best Checkpoint Sampling Report\\n')
    f.write('=' * 60 + '\\n\\n')

    total_frames = 0
    total_videos = 0

    # Check each environment
    environments = ['doom_battle', 'doom_battle2', 'doom_deathmatch_bots',
                   'doom_duel_bots', 'doom_duel_selfplay']

    for env_name in environments:
        env_frames_dir = os.path.join(frames_dir, env_name)

        f.write(f'Environment: {env_name}\\n')

        # Check if directory exists
        if os.path.exists(env_frames_dir):
            # Count frames
            frames = glob(os.path.join(env_frames_dir, 'frame_*.png'))
            f.write(f'  Frames: {len(frames)}\\n')
            total_frames += len(frames)

            # Check for metadata
            metadata_file = os.path.join(env_frames_dir, 'metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as mf:
                    metadata = json.load(mf)
                    f.write(f'  Episodes sampled: {metadata.get(\"episodes\", \"N/A\")}\\n')
                    f.write(f'  Total reward: {metadata.get(\"total_reward\", \"N/A\")}\\n')
        else:
            f.write(f'  Status: Not processed\\n')

        # Check for video
        video_file = os.path.join(video_dir, f'{env_name}.mp4')
        if os.path.exists(video_file):
            size_mb = os.path.getsize(video_file) / (1024 * 1024)
            f.write(f'  Video: {env_name}.mp4 ({size_mb:.1f} MB)\\n')
            total_videos += 1

        f.write('\\n')

    f.write('\\n' + '=' * 60 + '\\n')
    f.write(f'Summary:\\n')
    f.write(f'  Total frames: {total_frames}\\n')
    f.write(f'  Total videos: {total_videos}\\n')
    f.write(f'  Environments processed: {total_videos}/{len(environments)}\\n')

print(f'Report saved to: {report_file}')
"

# Display summary
echo ""
if [ -d "$FRAMES_DIR" ]; then
    echo "Summary per environment:"
    for env_dir in "$FRAMES_DIR"/*; do
        if [ -d "$env_dir" ]; then
            env_name=$(basename "$env_dir")
            episode_count=$(find "$env_dir" -type d -name "episode_*" 2>/dev/null | wc -l)
            frame_count=$(find "$env_dir" -name "frame_*.png" 2>/dev/null | wc -l)
            action_count=$(find "$env_dir" -name "action_*.txt" 2>/dev/null | wc -l)
            echo "  $env_name: $episode_count episodes, $frame_count frames, $action_count actions"
        fi
    done
fi

echo ""
if [ -d "$VIDEO_DIR" ]; then
    video_count=$(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l)
    echo "Total videos created: $video_count"
    echo ""
    echo "Videos:"
    ls -lh "$VIDEO_DIR"/*.mp4 2>/dev/null
fi

echo ""
echo "To view a video:"
echo "  ffplay ${VIDEO_DIR}/doom_battle.mp4"
echo ""
echo "To view the detailed report:"
echo "  cat ${OUTPUT_BASE}/best_checkpoint_report.txt"

exit 0