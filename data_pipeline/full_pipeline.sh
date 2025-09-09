#!/bin/bash

# Complete pipeline for sampling Atari frames and converting to training data

echo "=== Atari Training Data Pipeline ==="
echo ""

# Step 1: Sample frames from all Atari environments
echo "Step 1: Sampling frames from Atari environments..."
echo "----------------------------------------"
python3 sample_all_envs.py \
    --frames-per-env 60 \
    --max-episodes 3 \
    --randomness 0.2 \
    --parallel 8 \
    --output-dir sampled_frames_organized

echo ""
echo "Step 2: Converting to training data format..."
echo "----------------------------------------"
# Step 2: Convert sampled frames to training parquet
python3 data_pipeline/convert_to_training_data.py \
    --input-dir sampled_frames_organized \
    --output-file atari_training_data.parquet \
    --format interleaved

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
"

echo ""
echo "=== Pipeline complete! ==="
echo "Output saved to: atari_training_data.parquet"