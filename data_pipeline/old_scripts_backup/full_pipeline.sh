#!/bin/bash

# Exit immediately if any command fails
set -e
set -o pipefail

# Configuration
CHECKPOINT_DIR="/opt/tiger/atari_2B/checkpoints"
FRAMES_DIR="/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/sampled_frames_individual"
PARQUET_DIR="/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/parquet_individual"
FRAMES_PER_ENV=32
MAX_EPISODES=3000
RANDOMNESS=0.2
PARALLEL_JOBS=32

# Complete pipeline for sampling Atari frames and converting to training data
echo "=== Atari Training Data Pipeline (Individual Processing) ==="
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

# Create output directories
mkdir -p "$FRAMES_DIR"
mkdir -p "$PARQUET_DIR"

# Get list of all Atari environments from checkpoints
echo "Discovering available environments..."
ENVS=$(ls $CHECKPOINT_DIR | grep "^edbeeching_atari_2B_" | sed 's/edbeeching_atari_2B_//' | sed 's/_[0-9]*$//' | sort -u)
ENV_COUNT=$(echo "$ENVS" | wc -w)
echo "Found $ENV_COUNT environments to process"
echo ""

# Function to process a single environment
process_env() {
    local env_name=$1
    local env_num=$2
    local total=$3
    
    echo "[$env_num/$total] Processing $env_name..."
    
    # Check if parquet already exists and remove it (overwrite mode)
    if [ -f "$PARQUET_DIR/${env_name}.parquet" ]; then
        echo "[$env_num/$total] Found existing parquet for $env_name, overwriting..."
        rm -f "$PARQUET_DIR/${env_name}.parquet"
    fi
    
    # Also clean up existing frames if they exist
    if [ -d "$FRAMES_DIR/${env_name}" ]; then
        echo "[$env_num/$total] Found existing frames for $env_name, removing..."
        rm -rf "$FRAMES_DIR/${env_name}"
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

# Process environments sequentially (since parallel was causing issues)
echo "Step 1: Processing environments individually..."
echo "----------------------------------------"
env_num=1
success_count=0
failed_count=0
failed_envs=""

for env_name in $ENVS; do
    process_env "$env_name" "$env_num" "$ENV_COUNT"
    if [ $? -eq 0 ]; then
        ((success_count++))
    else
        ((failed_count++))
        failed_envs="$failed_envs $env_name"
    fi
    ((env_num++))
done

echo ""
echo "=== Processing Summary ==="
echo "Successfully processed: $success_count environments"
echo "Failed: $failed_count environments"
if [ ! -z "$failed_envs" ]; then
    echo "Failed environments:$failed_envs"
fi
echo "Individual parquet files saved to: $PARQUET_DIR"
echo ""
echo "=== Pipeline complete! ==="
echo "Individual parquet files saved to: $PARQUET_DIR"
echo ""
echo "To list all generated parquet files:"
echo "  ls -lh $PARQUET_DIR/*.parquet"
echo ""
echo "To check a specific environment's data:"
echo "  python3 -c \"import pandas as pd; df = pd.read_parquet('$PARQUET_DIR/atari_alien.parquet'); print(df.info())\""
exit 0