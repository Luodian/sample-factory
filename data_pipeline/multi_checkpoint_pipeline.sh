#!/bin/bash

# Configuration
CHECKPOINT_DIR="/opt/tiger/atari_2B/checkpoints"
FRAMES_DIR="/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/sampled_frames_multi_checkpoint"
PARQUET_DIR="/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/parquet_multi_checkpoint"
FRAMES_PER_ENV=32
MAX_EPISODES=1000  # Episodes per checkpoint
RANDOMNESS=0.2
EPSILON_GREEDY=0.3

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
        # Find checkpoint files
        for ckpt in $(ls -1 "$checkpoint_base"/checkpoint_*.pth 2>/dev/null | sort -V); do
            checkpoints+=("$ckpt")
        done
    fi
    
    echo "${checkpoints[@]}"
}

# List of environments to process
ENVS="atari_alien atari_assault atari_asterix"

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
output_dir = '${FRAMES_DIR}'
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
args = (experiment, checkpoint_dir, output_dir + '_ckpt${checkpoint_suffix}', 
        frames_per_env, max_episodes, device, randomness, epsilon_greedy)
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

success = convert_env_to_parquet('$(basename $env_dir)', '$(dirname $env_dir)', '$output_file')
if success:
    print(f'Successfully converted to parquet')
else:
    print(f'Failed to convert to parquet')
"
done

echo "=== Pipeline complete! ==="
ls -lh $PARQUET_DIR/*.parquet 2>/dev/null | wc -l | xargs echo "Total parquet files created:"