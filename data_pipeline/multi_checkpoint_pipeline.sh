#!/bin/bash

# uv pip install "gymnasium[atari,accept-rom-license]" shimmy
# uv pip install ale-py autorom
# AutoROM --accept-license

# Configuration
CHECKPOINT_DIR="/opt/tiger/sample-factory/checkpoints"
FRAMES_DIR="/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/sampled_frames_multi_checkpoint_0916"
PARQUET_DIR="/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/parquet_multi_checkpoint_0916"
FRAMES_PER_ENV=128
MAX_EPISODES=100  # Episodes per checkpoint
RANDOMNESS=0.3
EPSILON_GREEDY=0.3

# Optional: limit number of environments to process (set to 0 for all)
MAX_ENVS=${MAX_ENVS:-0}  # Can be overridden by environment variable

# Optional: specific environments to process (comma-separated, e.g., "atari_alien,atari_assault")
SPECIFIC_ENVS=${SPECIFIC_ENVS:-""}  # Can be overridden by environment variable

echo "=== Multi-Checkpoint Atari Pipeline ==="
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Frames output: $FRAMES_DIR"
echo "Parquet output: $PARQUET_DIR"
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
        # Find checkpoint files inside per-policy subdirs (e.g., checkpoint_p0)
        for ckpt in $(ls -1 "$checkpoint_base"/checkpoint_p*/checkpoint_*.pth 2>/dev/null | sort -V); do
            checkpoints+=("$ckpt")
        done
    fi
    
    echo "${checkpoints[@]}"
}

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
echo "Will process $ENV_COUNT environment(s): $ENVS"
echo ""

# Process each environment
for env_name in $ENVS; do
    echo "Processing $env_name..."
    
    # Find available checkpoints
    checkpoints=($(find_checkpoints "$env_name"))
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "No checkpoints found for $env_name, skipping..."
        continue
    fi
    
    echo "Found ${#checkpoints[@]} checkpoints for $env_name"
    
    # Select checkpoints from different training stages
    # Use first (early), middle, and last (final) checkpoints
    selected_checkpoints=()
    
    if [ ${#checkpoints[@]} -eq 1 ]; then
        selected_checkpoints=("${checkpoints[0]}")
    elif [ ${#checkpoints[@]} -eq 2 ]; then
        selected_checkpoints=("${checkpoints[0]}" "${checkpoints[1]}")
    else
        # Select early, mid, and late checkpoints
        first_idx=0
        mid_idx=$((${#checkpoints[@]} / 2))
        last_idx=$((${#checkpoints[@]} - 1))
        
        selected_checkpoints=(
            "${checkpoints[$first_idx]}"
            "${checkpoints[$mid_idx]}"
            "${checkpoints[$last_idx]}"
        )
    fi
    
    echo "Selected ${#selected_checkpoints[@]} checkpoints from different training stages"
    
    # Process each selected checkpoint
    checkpoint_num=1
    for checkpoint_path in "${selected_checkpoints[@]}"; do
        checkpoint_name=$(basename "$checkpoint_path" .pth)
        echo "  [$checkpoint_num/${#selected_checkpoints[@]}] Processing checkpoint: $checkpoint_name"
        
        # Create unique output directory for this checkpoint
        checkpoint_suffix="${checkpoint_name##*_}"  # Get the number part
        env_frames_dir="${FRAMES_DIR}/${env_name}_ckpt${checkpoint_suffix}"
        
        # Remove existing files if they exist
        if [ -d "$env_frames_dir" ]; then
            rm -rf "$env_frames_dir"
        fi
        
        # Sample frames using this specific checkpoint
        python3 -c "
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('$0')))
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

print(f'Result: {status}, {frame_count} frames')
"
        
        checkpoint_num=$((checkpoint_num + 1))
    done
    
    echo "Completed $env_name with multiple checkpoints"
    echo ""
done

echo "=== Multi-checkpoint pipeline complete! ==="
echo "Frames saved to: $FRAMES_DIR"

# Also create combined videos from all frames per env/checkpoint
echo "Creating combined videos from frames..."
VIDEO_DIR="${FRAMES_DIR}_videos"
VIDEO_FPS=${VIDEO_FPS:-10}  # Default FPS for combined video

# Create videos using Python (more reliable than ffmpeg)
python3 -c "
import os
import sys
import imageio.v2 as imageio
from glob import glob
from natsort import natsorted

frames_dir = '${FRAMES_DIR}'
video_dir = '${VIDEO_DIR}'
fps = ${VIDEO_FPS}

print(f'Creating videos in: {video_dir}')
os.makedirs(video_dir, exist_ok=True)

# Find all checkpoint directories
env_dirs = glob(os.path.join(frames_dir, '*_ckpt*'))
env_dirs = [d for d in env_dirs if os.path.isdir(d)]

if not env_dirs:
    print('No checkpoint directories found')
    sys.exit(0)

success_count = 0
for env_dir in env_dirs:
    env_name = os.path.basename(env_dir)
    output_path = os.path.join(video_dir, f'{env_name}.mp4')

    # Find all frames
    frame_files = glob(os.path.join(env_dir, '**', '*.png'), recursive=True)
    frame_files = natsorted(frame_files)

    if not frame_files:
        print(f'No frames found for {env_name}')
        continue

    print(f'Creating video for {env_name}: {len(frame_files)} frames')

    try:
        # Create video with macro_block_size=1 to avoid resizing warnings
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                   pixelformat='yuv420p', quality=8,
                                   macro_block_size=1)

        for i, frame_file in enumerate(frame_files):
            if i % 100 == 0:
                print(f'  Processing frame {i}/{len(frame_files)}')
            frame = imageio.imread(frame_file)
            writer.append_data(frame)

        writer.close()

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f'  Created: {output_path} ({size_mb:.1f} MB)')
        success_count += 1

    except Exception as e:
        print(f'  Failed to create video for {env_name}: {e}')

print(f'Successfully created {success_count}/{len(env_dirs)} videos')
"

echo "Videos saved to: $VIDEO_DIR"
ls -lh "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l | xargs echo "Total videos created:"

# Now convert all sampled frames to parquet
echo "Converting frames to parquet format..."
for env_dir in $(ls -d ${FRAMES_DIR}/*_ckpt* 2>/dev/null); do
    env_name=$(basename "$env_dir" | sed 's/_ckpt.*//')
    checkpoint_suffix=$(basename "$env_dir" | sed 's/.*_ckpt//')
    output_file="${PARQUET_DIR}/${env_name}_ckpt${checkpoint_suffix}.parquet"
    
    echo "Converting $env_dir to $output_file"
    
    python3 -c "
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('$0')))
from process_individual_env import convert_env_to_parquet

success = convert_env_to_parquet('${env_name}', '${env_dir}', '$output_file')
if success:
    print(f'Successfully converted to parquet')
else:
    print(f'Failed to convert to parquet')
"
done

echo "=== Pipeline complete! ==="
ls -lh $PARQUET_DIR/*.parquet 2>/dev/null | wc -l | xargs echo "Total parquet files created:"
