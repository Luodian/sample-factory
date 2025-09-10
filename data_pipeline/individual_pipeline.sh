#!/bin/bash

# Exit immediately if any command fails
set -e
set -o pipefail

# Configuration
CHECKPOINT_DIR="/opt/tiger/atari_2B/checkpoints"
FRAMES_DIR="/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/sampled_frames_individual"
PARQUET_DIR="/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/parquet_individual"
FRAMES_PER_ENV=32
MAX_EPISODES=100
RANDOMNESS=0.2
PARALLEL_JOBS=8

echo "=== Individual Atari Environment Processing Pipeline ==="
echo "Configuration:"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Frames output: $FRAMES_DIR"
echo "  Parquet output: $PARQUET_DIR"
echo "  Frames per episode: $FRAMES_PER_ENV"
echo "  Max episodes per env: $MAX_EPISODES"
echo "  Randomness: $RANDOMNESS"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo ""

# Register Atari environments
echo "Registering Atari environments..."
python3 -c "import gymnasium as gym; import ale_py; gym.register_envs(ale_py); print('Atari environments registered successfully')" || {
    echo "ERROR: Failed to register Atari environments"
    exit 1
}
echo ""

# Get list of all Atari environments from checkpoints
echo "Discovering available environments..."
ENVS=$(ls $CHECKPOINT_DIR | grep "^edbeeching_atari_2B_" | sed 's/edbeeching_atari_2B_//' | sed 's/_[0-9]*$//' | sort -u)
ENV_COUNT=$(echo "$ENVS" | wc -w)
echo "Found $ENV_COUNT environments to process"
echo ""

# Create output directories
mkdir -p "$FRAMES_DIR"
mkdir -p "$PARQUET_DIR"

# Function to process a single environment
process_env() {
    local env_name=$1
    local env_num=$2
    local total=$3
    
    echo "[$env_num/$total] Processing $env_name..."
    
    # Check if parquet already exists
    if [ -f "$PARQUET_DIR/${env_name}.parquet" ]; then
        echo "[$env_num/$total] $env_name already processed, skipping..."
        return 0
    fi
    
    # Process the environment
    python3 process_individual_env.py \
        --env-name "$env_name" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --frames-dir "$FRAMES_DIR" \
        --output-dir "$PARQUET_DIR" \
        --frames-per-env $FRAMES_PER_ENV \
        --max-episodes $MAX_EPISODES \
        --randomness $RANDOMNESS \
        --device cpu 2>&1 | sed "s/^/[$env_name] /"
    
    if [ $? -eq 0 ]; then
        echo "[$env_num/$total] ✓ Successfully processed $env_name"
        return 0
    else
        echo "[$env_num/$total] ✗ Failed to process $env_name"
        return 1
    fi
}

# Export function and variables for parallel execution
export -f process_env
export CHECKPOINT_DIR FRAMES_DIR PARQUET_DIR FRAMES_PER_ENV MAX_EPISODES RANDOMNESS

# Process environments in parallel
echo "Starting parallel processing with $PARALLEL_JOBS workers..."
echo "----------------------------------------"

# Create a list of environments with index
env_num=1
success_count=0
failed_count=0

# Use GNU parallel if available, otherwise fall back to sequential processing
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for processing..."
    echo "$ENVS" | tr ' ' '\n' | nl -nln -w1 | \
        parallel -j $PARALLEL_JOBS --colsep ' ' process_env {2} {1} $ENV_COUNT
else
    echo "GNU parallel not found, processing sequentially..."
    for env_name in $ENVS; do
        process_env "$env_name" "$env_num" "$ENV_COUNT"
        if [ $? -eq 0 ]; then
            ((success_count++))
        else
            ((failed_count++))
        fi
        ((env_num++))
    done
fi

echo ""
echo "=== Processing Complete ==="
echo "Successfully processed environments"
echo "Parquet files saved to: $PARQUET_DIR"

# Optional: Combine all parquet files into one if needed
echo ""
echo "Creating combined parquet file..."
python3 -c "
import pandas as pd
import glob
import os

parquet_files = glob.glob('$PARQUET_DIR/*.parquet')
if parquet_files:
    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    output_file = '$PARQUET_DIR/combined_atari_training_data.parquet'
    combined_df.to_parquet(output_file, index=False)
    print(f'Combined {len(parquet_files)} files into {output_file}')
    print(f'Total episodes: {len(combined_df)}')
    print(f'Total frames: {combined_df[\"num_frames\"].sum()}')
else:
    print('No parquet files found to combine')
"

echo ""
echo "=== Pipeline Complete! ==="
exit 0