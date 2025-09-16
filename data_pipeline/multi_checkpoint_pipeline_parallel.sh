#!/bin/bash

# Parallel version of multi-checkpoint pipeline
# Uses GNU parallel or xargs to process multiple environments concurrently

# Configuration
CHECKPOINT_DIR="/opt/tiger/sample-factory/checkpoints"
FRAMES_DIR="/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/sampled_frames_multi_checkpoint"
PARQUET_DIR="/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/parquet_multi_checkpoint"
FRAMES_PER_ENV=128
MAX_EPISODES=100  # Episodes per checkpoint
RANDOMNESS=0.3
EPSILON_GREEDY=0.3

# Parallel configuration
PARALLEL_JOBS=${PARALLEL_JOBS:-8}  # Number of parallel jobs (default 8, max 23 cores available)
MAX_ENVS=${MAX_ENVS:-0}  # Limit number of environments (0 = all)
SPECIFIC_ENVS=${SPECIFIC_ENVS:-""}  # Specific environments to process

echo "=== Multi-Checkpoint Atari Pipeline (PARALLEL) ==="
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Frames output: $FRAMES_DIR"
echo "Parquet output: $PARQUET_DIR"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Using multiple checkpoints from different training stages"
echo ""

# Create output directories
mkdir -p "$FRAMES_DIR"
mkdir -p "$PARQUET_DIR"

# Function to find checkpoints at different training stages
find_checkpoints() {
    local env_name=$1
    local checkpoint_base="${CHECKPOINT_DIR}/edbeeching_atari_2B_${env_name}_1111"

    # Look for checkpoints at different training stages
    local checkpoints=()

    # Try to find early, mid, and late checkpoints
    if [ -d "$checkpoint_base" ]; then
        # Find checkpoint files inside per-policy subdirs
        for ckpt in $(ls -1 "$checkpoint_base"/checkpoint_p*/checkpoint_*.pth 2>/dev/null | sort -V); do
            checkpoints+=(\"$ckpt\")
        done
    fi

    echo "${checkpoints[@]}"
}

# Function to process a single environment (will be called in parallel)
process_environment() {
    local env_name=$1
    echo "[$(date '+%H:%M:%S')] Starting $env_name (PID: $$)"

    # Find available checkpoints
    local checkpoint_base="${CHECKPOINT_DIR}/edbeeching_atari_2B_${env_name}_1111"

    if [ ! -d "$checkpoint_base" ]; then
        echo "[$(date '+%H:%M:%S')] No checkpoints found for $env_name, skipping..."
        return 1
    fi

    # Find checkpoint files
    local checkpoints=($(ls -1 "$checkpoint_base"/checkpoint_p*/checkpoint_*.pth 2>/dev/null | sort -V))

    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] No checkpoint files found for $env_name, skipping..."
        return 1
    fi

    echo "[$(date '+%H:%M:%S')] Found ${#checkpoints[@]} checkpoints for $env_name"

    # Select checkpoints from different training stages
    local selected_checkpoints=()

    if [ ${#checkpoints[@]} -eq 1 ]; then
        selected_checkpoints=("${checkpoints[0]}")
    elif [ ${#checkpoints[@]} -eq 2 ]; then
        selected_checkpoints=("${checkpoints[0]}" "${checkpoints[1]}")
    else
        # Select early, mid, and late checkpoints
        local first_idx=0
        local mid_idx=$((${#checkpoints[@]} / 2))
        local last_idx=$((${#checkpoints[@]} - 1))

        selected_checkpoints=(
            "${checkpoints[$first_idx]}"
            "${checkpoints[$mid_idx]}"
            "${checkpoints[$last_idx]}"
        )
    fi

    # Process each selected checkpoint
    local checkpoint_num=1
    for checkpoint_path in "${selected_checkpoints[@]}"; do
        local checkpoint_name=$(basename "$checkpoint_path" .pth)
        echo "[$(date '+%H:%M:%S')] $env_name: Processing checkpoint $checkpoint_num/${#selected_checkpoints[@]}"

        # Create unique output directory for this checkpoint
        local checkpoint_suffix="${checkpoint_name##*_}"  # Get the number part
        local env_frames_dir="${FRAMES_DIR}/${env_name}_ckpt${checkpoint_suffix}"

        # Remove existing files if they exist
        if [ -d "$env_frames_dir" ]; then
            rm -rf "$env_frames_dir"
        fi

        # Sample frames using this specific checkpoint
        python3 -c "
import sys
import os
sys.path.append('${CHECKPOINT_DIR}/../data_pipeline')
from sample_all_envs import sample_environment

# Use the specific checkpoint
experiment = 'edbeeching_atari_2B_${env_name}_1111'
checkpoint_dir = '${CHECKPOINT_DIR}'
output_dir = '${env_frames_dir}'
frames_per_env = ${FRAMES_PER_ENV}
max_episodes = ${MAX_EPISODES}
device = 'cpu'
randomness = ${RANDOMNESS}
epsilon_greedy = ${EPSILON_GREEDY}

# Override to use specific checkpoint
import shutil
checkpoint_override_dir = os.path.join(checkpoint_dir, experiment)
os.makedirs(checkpoint_override_dir, exist_ok=True)

# Copy the specific checkpoint as the latest
latest_path = os.path.join(checkpoint_override_dir, 'checkpoint_latest.pth')
if os.path.exists(latest_path):
    os.remove(latest_path)
shutil.copy2('${checkpoint_path}', latest_path)

# Sample with this checkpoint
args = (experiment, checkpoint_dir, output_dir,
        frames_per_env, max_episodes, device, randomness, epsilon_greedy, False)
env_name, status, frame_count = sample_environment(args)

print(f'[{env_name}] Result: {status}, {frame_count} frames')
" 2>&1 | sed "s/^/[$env_name] /"

        checkpoint_num=$((checkpoint_num + 1))
    done

    echo "[$(date '+%H:%M:%S')] Completed $env_name with multiple checkpoints"
    return 0
}

# Export the function so it can be used by parallel/xargs
export -f process_environment
export CHECKPOINT_DIR FRAMES_DIR PARQUET_DIR FRAMES_PER_ENV MAX_EPISODES RANDOMNESS EPSILON_GREEDY

# List of environments to process
if [ -n "$SPECIFIC_ENVS" ]; then
    # Use specific environments if provided
    ENVS=$(echo "$SPECIFIC_ENVS" | tr ',' ' ')
    echo "Using specified environments: $ENVS"
else
    # Automatically detect all available environments with _1111 suffix
    ENVS=$(ls -1 "$CHECKPOINT_DIR" | grep "^edbeeching_atari_2B_.*_1111$" | sed 's/edbeeching_atari_2B_//' | sed 's/_1111$//' | tr '\n' ' ')

    if [ -z "$ENVS" ]; then
        echo "No environments found with _1111 suffix in $CHECKPOINT_DIR"
        exit 1
    fi

    # Apply MAX_ENVS limit if set
    if [ "$MAX_ENVS" -gt 0 ]; then
        ENVS=$(echo $ENVS | tr ' ' '\n' | head -n "$MAX_ENVS" | tr '\n' ' ')
        echo "Limiting to first $MAX_ENVS environments"
    fi
fi

ENV_COUNT=$(echo $ENVS | wc -w)
echo "Will process $ENV_COUNT environment(s) in parallel with $PARALLEL_JOBS jobs"
echo "Environments: $ENVS"
echo ""
echo "Starting parallel processing..."
echo "="

START_TIME=$(date +%s)

# Check if GNU parallel is available
if command -v parallel >/dev/null 2>&1; then
    echo "Using GNU parallel..."
    echo $ENVS | tr ' ' '\n' | parallel -j $PARALLEL_JOBS --eta --progress process_environment {}
else
    echo "GNU parallel not found, using xargs..."
    echo $ENVS | tr ' ' '\n' | xargs -n 1 -P $PARALLEL_JOBS -I {} bash -c 'process_environment "$@"' _ {}
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=== Parallel processing complete in $((DURATION / 60)) minutes $((DURATION % 60)) seconds ==="
echo "Frames saved to: $FRAMES_DIR"

# Create videos from frames - one video per episode
echo ""
echo "Creating videos from frames (one per episode)..."
VIDEO_DIR="${FRAMES_DIR}_videos"
VIDEO_FPS=${VIDEO_FPS:-10}

# Create videos using Python (also in parallel)
python3 -c "
import os
import sys
import imageio.v2 as imageio
from glob import glob
from natsort import natsorted
from multiprocessing import Pool, cpu_count

def create_episode_video(args):
    episode_dir, video_dir, fps = args
    # Extract environment and episode info from path
    # Path format: .../atari_alien_ckpt2000093184/atari_alien/episode_000_rand0.3
    path_parts = episode_dir.split('/')
    env_checkpoint = path_parts[-3]  # e.g., atari_alien_ckpt2000093184
    env_name = path_parts[-2]  # e.g., atari_alien
    episode_name = path_parts[-1]  # e.g., episode_000_rand0.3

    # Create subdirectory for this env/checkpoint
    env_video_dir = os.path.join(video_dir, env_checkpoint)
    os.makedirs(env_video_dir, exist_ok=True)

    output_path = os.path.join(env_video_dir, f'{episode_name}.mp4')

    # Find all frames in this episode
    frame_files = glob(os.path.join(episode_dir, '*.png'))
    frame_files = natsorted(frame_files)

    if not frame_files:
        print(f'No frames found for {env_checkpoint}/{episode_name}')
        return False

    try:
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                   pixelformat='yuv420p', quality=8,
                                   macro_block_size=1)

        for frame_file in frame_files:
            frame = imageio.imread(frame_file)
            writer.append_data(frame)

        writer.close()

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f'  Created: {env_checkpoint}/{episode_name}.mp4 ({len(frame_files)} frames, {size_mb:.1f} MB)')
        return True

    except Exception as e:
        print(f'  Failed to create video for {env_checkpoint}/{episode_name}: {e}')
        return False

frames_dir = '${FRAMES_DIR}'
video_dir = '${VIDEO_DIR}'
fps = ${VIDEO_FPS}

print(f'Creating episode videos in: {video_dir}')
os.makedirs(video_dir, exist_ok=True)

# Find all episode directories
episode_dirs = []
env_checkpoint_dirs = glob(os.path.join(frames_dir, '*_ckpt*'))

for env_ckpt_dir in env_checkpoint_dirs:
    if not os.path.isdir(env_ckpt_dir):
        continue
    # Find environment subdirectory
    env_subdirs = glob(os.path.join(env_ckpt_dir, '*'))
    for env_subdir in env_subdirs:
        if not os.path.isdir(env_subdir):
            continue
        # Find episode directories
        ep_dirs = glob(os.path.join(env_subdir, 'episode_*'))
        episode_dirs.extend(ep_dirs)

if not episode_dirs:
    print('No episode directories found')
    sys.exit(0)

# Process videos in parallel
num_workers = min(${PARALLEL_JOBS}, cpu_count())
print(f'Creating videos for {len(episode_dirs)} episodes using {num_workers} workers...')

with Pool(num_workers) as pool:
    args_list = [(ep_dir, video_dir, fps) for ep_dir in episode_dirs]
    results = pool.map(create_episode_video, args_list)

success_count = sum(results)
print(f'Successfully created {success_count}/{len(episode_dirs)} episode videos')
"

echo "Videos saved to: $VIDEO_DIR"
find "$VIDEO_DIR" -name "*.mp4" | wc -l | xargs echo "Total episode videos created:"

# Convert to parquet in parallel
echo ""
echo "Converting frames to parquet format (parallel)..."

convert_to_parquet() {
    local env_dir=$1
    local env_name=$(basename "$env_dir" | sed 's/_ckpt.*//')
    local checkpoint_suffix=$(basename "$env_dir" | sed 's/.*_ckpt//')
    local output_file="${PARQUET_DIR}/${env_name}_ckpt${checkpoint_suffix}.parquet"

    echo "[$(date '+%H:%M:%S')] Converting $env_name to parquet..."

    python3 -c "
import sys
import os
sys.path.append('${CHECKPOINT_DIR}/../data_pipeline')
from process_individual_env import convert_env_to_parquet

success = convert_env_to_parquet('${env_name}', '${env_dir}', '${output_file}')
if success:
    print(f'[$(date '+%H:%M:%S')] Successfully converted ${env_name} to parquet')
else:
    print(f'[$(date '+%H:%M:%S')] Failed to convert ${env_name} to parquet')
"
}

export -f convert_to_parquet
export PARQUET_DIR

# Convert all directories to parquet in parallel
find ${FRAMES_DIR} -maxdepth 1 -name "*_ckpt*" -type d | \
    xargs -n 1 -P $PARALLEL_JOBS -I {} bash -c 'convert_to_parquet "$@"' _ {}

echo ""
echo "=== Pipeline complete! ==="
ls -lh $PARQUET_DIR/*.parquet 2>/dev/null | wc -l | xargs echo "Total parquet files created:"

TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - START_TIME))
echo "Total pipeline time: $((TOTAL_DURATION / 60)) minutes $((TOTAL_DURATION % 60)) seconds"