#!/bin/bash

# VizDoom Sampling Pipeline
# Downloads models and samples frames/videos from all VizDoom environments

set -e

# Configuration
MODEL_DIR="${VIZDOOM_MODEL_DIR:-/tmp/vizdoom_models}"
OUTPUT_DIR="${VIZDOOM_OUTPUT_DIR:-/tmp/vizdoom_samples}"
FRAMES_PER_ENV="${FRAMES_PER_ENV:-256}"
MAX_EPISODES="${MAX_EPISODES:-50}"
DEVICE="${DEVICE:-cpu}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"

echo "=== VizDoom Sampling Pipeline ==="
echo "Model directory: $MODEL_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Frames per environment: $FRAMES_PER_ENV"
echo "Max episodes: $MAX_EPISODES"
echo "Device: $DEVICE"
echo "Parallel jobs: $PARALLEL_JOBS"
echo ""

# Step 1: Download models if needed
echo "Step 1: Downloading VizDoom models..."
./download_vizdoom_models.sh
echo ""

# Step 2: Create output directory
mkdir -p "$OUTPUT_DIR"

# Define environments and their HuggingFace model mappings
declare -A ENV_MODELS=(
    ["doom_battle"]="$MODEL_DIR/doom_battle/checkpoint.pth"
    ["doom_battle2"]="$MODEL_DIR/doom_battle2/checkpoint.pth"
    ["doom_deathmatch_bots"]="$MODEL_DIR/doom_deathmatch_bots/checkpoint.pth"
    ["doom_duel_bots"]="$MODEL_DIR/doom_duel_bots/checkpoint.pth"
    ["doom_duel_selfplay"]="$MODEL_DIR/doom_duel_selfplay/checkpoint.pth"
)

# Function to process a single environment
process_environment() {
    local env_name=$1
    local checkpoint_path=$2
    local env_output_dir="${OUTPUT_DIR}/${env_name}"

    echo "[${env_name}] Starting sampling..."

    # Check if checkpoint exists
    if [ ! -f "$checkpoint_path" ]; then
        # Try to find any .pth file in the model directory
        local model_dir=$(dirname "$checkpoint_path")
        checkpoint_path=$(find "$model_dir" -name "*.pth" -type f | head -1)

        if [ -z "$checkpoint_path" ] || [ ! -f "$checkpoint_path" ]; then
            echo "[${env_name}] ERROR: Checkpoint not found"
            return 1
        fi
    fi

    # Create environment output directory
    mkdir -p "$env_output_dir"

    # Run sampling
    python3 sample_vizdoom_simple.py \
        --env "$env_name" \
        --checkpoint "$checkpoint_path" \
        --output-dir "$env_output_dir" \
        --frames "$FRAMES_PER_ENV" \
        --max-episodes "$MAX_EPISODES" \
        --device "$DEVICE" \
        --save-video \
        --video-name "${env_name}"

    if [ $? -eq 0 ]; then
        echo "[${env_name}] Successfully sampled frames and created video"

        # Count frames
        local frame_count=$(ls "$env_output_dir"/frame_*.png 2>/dev/null | wc -l)
        echo "[${env_name}] Generated $frame_count frames"

        # Check for video
        if [ -f "$env_output_dir/${env_name}.mp4" ]; then
            local video_size=$(du -h "$env_output_dir/${env_name}.mp4" | cut -f1)
            echo "[${env_name}] Video size: $video_size"
        fi
    else
        echo "[${env_name}] ERROR: Failed to sample"
        return 1
    fi

    echo "[${env_name}] Completed"
    return 0
}

export -f process_environment
export OUTPUT_DIR FRAMES_PER_ENV MAX_EPISODES DEVICE

# Step 3: Process environments
echo ""
echo "Step 2: Sampling frames and creating videos..."
echo ""

# Check if GNU parallel is available for parallel processing
if command -v parallel &> /dev/null && [ "$PARALLEL_JOBS" -gt 1 ]; then
    echo "Processing environments in parallel (${PARALLEL_JOBS} jobs)..."

    # Create job list
    job_file=$(mktemp)
    for env_name in "${!ENV_MODELS[@]}"; do
        echo "$env_name ${ENV_MODELS[$env_name]}" >> "$job_file"
    done

    # Run in parallel
    cat "$job_file" | parallel --colsep ' ' -j "$PARALLEL_JOBS" process_environment {1} {2}

    rm "$job_file"
else
    echo "Processing environments sequentially..."

    success_count=0
    total_count=${#ENV_MODELS[@]}

    for env_name in "${!ENV_MODELS[@]}"; do
        checkpoint_path="${ENV_MODELS[$env_name]}"

        process_environment "$env_name" "$checkpoint_path"

        if [ $? -eq 0 ]; then
            ((success_count++))
        fi
        echo ""
    done

    echo "=== Summary ==="
    echo "Successfully processed: $success_count/$total_count environments"
fi

# Step 4: Generate summary report
echo ""
echo "Step 3: Generating summary report..."

python3 -c "
import os
import json
from glob import glob

output_dir = '${OUTPUT_DIR}'
report_file = os.path.join(output_dir, 'summary_report.txt')

with open(report_file, 'w') as f:
    f.write('VizDoom Sampling Pipeline Report\\n')
    f.write('=' * 50 + '\\n\\n')

    total_frames = 0
    total_videos = 0

    # Check each environment
    for env_dir in sorted(glob(os.path.join(output_dir, '*/'))):
        env_name = os.path.basename(env_dir.rstrip('/'))

        # Skip if not an environment directory
        if not os.path.isdir(env_dir) or env_name == 'summary_report.txt':
            continue

        f.write(f'Environment: {env_name}\\n')

        # Count frames
        frames = glob(os.path.join(env_dir, 'frame_*.png'))
        f.write(f'  Frames: {len(frames)}\\n')
        total_frames += len(frames)

        # Check for video
        videos = glob(os.path.join(env_dir, '*.mp4'))
        if videos:
            for video in videos:
                size_mb = os.path.getsize(video) / (1024 * 1024)
                f.write(f'  Video: {os.path.basename(video)} ({size_mb:.1f} MB)\\n')
            total_videos += len(videos)

        # Check for metadata
        metadata_file = os.path.join(env_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as mf:
                metadata = json.load(mf)
                f.write(f'  Episodes: {metadata.get(\"episodes\", \"N/A\")}\\n')
                f.write(f'  Frameskip: {metadata.get(\"frameskip\", \"N/A\")}\\n')

        f.write('\\n')

    f.write('\\n' + '=' * 50 + '\\n')
    f.write(f'Total frames: {total_frames}\\n')
    f.write(f'Total videos: {total_videos}\\n')

print(f'Summary report saved to: {report_file}')
"

# Display summary
echo ""
echo "=== Pipeline Complete ==="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Contents:"
ls -la "$OUTPUT_DIR"/ | head -10
echo ""

if [ -f "$OUTPUT_DIR/summary_report.txt" ]; then
    echo "Summary:"
    cat "$OUTPUT_DIR/summary_report.txt"
fi

echo ""
echo "To view a sample video:"
echo "  ffplay ${OUTPUT_DIR}/doom_battle/doom_battle.mp4"
echo ""
echo "To view frames:"
echo "  ls ${OUTPUT_DIR}/*/frame_*.png"

exit 0