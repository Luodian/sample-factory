#!/bin/bash

# Exit immediately if any command fails
set -e
set -o pipefail

# Complete pipeline for sampling Atari frames and converting to training data
echo "=== Atari Training Data Pipeline ==="
# Register Atari environments
echo "Registering Atari environments..."
python3 -c "import gymnasium as gym; import ale_py; gym.register_envs(ale_py); print('Atari environments registered successfully')" || {
    echo "ERROR: Failed to register Atari environments"
    exit 1
}
echo ""

# Step 1: Sample frames from all Atari environments
echo "Step 1: Sampling frames from Atari environments..."
echo "----------------------------------------"
python3 sample_all_envs.py \
    --checkpoint-dir /opt/tiger/atari_2B/checkpoints \
    --frames-per-env 60 \
    --max-episodes 3 \
    --randomness 0.2 \
    --parallel 8 \
    --output-dir /mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/sampled_frames_organized || {
    echo "ERROR: Frame sampling failed"
    exit 1
}

echo ""
echo "Step 2: Converting to training data format..."
echo "----------------------------------------"
# Step 2: Convert sampled frames to training parquet
python3 convert_to_training_data.py \
    --input-dir /mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/sampled_frames_organized \
    --output-file atari_training_data.parquet \
    --format interleaved || {
    echo "ERROR: Conversion to training data failed"
    exit 1
}

echo ""
echo "Step 3: Verifying output..."
echo "----------------------------------------"
# Step 3: Verify the output
python3 -c "
import pandas as pd
df = pd.read_parquet('atari_training_data.parquet')
print(f'Dataset shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print(f'Environments: {list(df[\"environment\"].unique())[:5]}...')
print(f'Total episodes: {len(df)}')

# Show structure of first row
row = df.iloc[0]
print()
print('First episode structure:')
print(f'  - Environment: {row[\"environment\"]}')
print(f'  - Episode: {row[\"episode\"]}')
print(f'  - Num frames: {row[\"num_frames\"]}')
print(f'  - Randomness: {row[\"randomness\"]}')
print(f'  - Number of inputs: {len(row[\"inputs\"])}')
print(f'  - Number of images: {len(row[\"images\"])}')
" || {
    echo "ERROR: Verification failed"
    exit 1
}

echo ""
echo "=== Pipeline complete! ==="
echo "Output saved to: atari_training_data.parquet"
exit 0