#!/bin/bash

# Script to sample frames from all Atari environments with existing checkpoints
# This script will create frame samples for each environment found in the checkpoints directory

CHECKPOINT_DIR="/opt/tiger/sample-factory/checkpoints"
OUTPUT_BASE_DIR="/opt/tiger/sample-factory/sampled_frames"
FRAMES_PER_ENV=100  # Number of frames to sample per environment
MAX_EPISODES=1      # Number of episodes to sample

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Sample Factory - Multi-Environment Sampler"
echo "========================================="
echo ""

# Create base output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Get list of all experiments (use first seed only to avoid duplicates)
EXPERIMENTS=$(ls "$CHECKPOINT_DIR" | grep "_1111$" | sort)
TOTAL_ENVS=$(echo "$EXPERIMENTS" | wc -l)

if [ "$TOTAL_ENVS" -eq 0 ]; then
    echo -e "${RED}No checkpoint directories found ending with _1111${NC}"
    exit 1
fi

echo -e "${GREEN}Found $TOTAL_ENVS environments to sample${NC}"
echo ""

# Counter for progress
CURRENT=0
SUCCESS=0
FAILED=0

# Process each environment
for EXPERIMENT in $EXPERIMENTS; do
    CURRENT=$((CURRENT + 1))
    
    # Extract environment name from experiment name
    # Format: edbeeching_atari_2B_atari_alien_1111 -> atari_alien
    ENV_NAME=$(echo "$EXPERIMENT" | sed 's/edbeeching_atari_2B_//;s/_1111$//')
    
    echo -e "${YELLOW}[$CURRENT/$TOTAL_ENVS] Processing: $ENV_NAME${NC}"
    
    # Create output directory for this environment
    ENV_OUTPUT_DIR="$OUTPUT_BASE_DIR/$ENV_NAME"
    
    # Skip if already processed
    if [ -d "$ENV_OUTPUT_DIR" ] && [ "$(ls -A $ENV_OUTPUT_DIR 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  ⏭️  Skipping (already sampled)"
        continue
    fi
    
    # Run the sampling command
    python3 -m sf_examples.atari.enjoy_atari \
        --env="$ENV_NAME" \
        --experiment="$EXPERIMENT" \
        --train_dir="$CHECKPOINT_DIR" \
        --device=cpu \
        --save_frames \
        --frames_dir="$ENV_OUTPUT_DIR" \
        --video_frames=$FRAMES_PER_ENV \
        --max_num_episodes=$MAX_EPISODES \
        --no_render \
        --eval_deterministic \
        2>&1 | grep -E "(Saving frames to|Saved .* frames to)" &
    
    # Store the PID
    PID=$!
    
    # Wait for the process with a timeout
    TIMEOUT=30
    TIMER=0
    while [ $TIMER -lt $TIMEOUT ]; do
        if ! kill -0 $PID 2>/dev/null; then
            # Process finished
            wait $PID
            EXIT_CODE=$?
            break
        fi
        sleep 1
        TIMER=$((TIMER + 1))
    done
    
    # Check if timeout occurred
    if [ $TIMER -ge $TIMEOUT ]; then
        echo "  ⚠️  Timeout reached, killing process"
        kill -9 $PID 2>/dev/null
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Check if sampling was successful
    if [ -d "$ENV_OUTPUT_DIR" ] && [ "$(ls -A $ENV_OUTPUT_DIR 2>/dev/null | wc -l)" -gt 0 ]; then
        FRAME_COUNT=$(ls "$ENV_OUTPUT_DIR"/frame_*.png 2>/dev/null | wc -l)
        ACTION_COUNT=$(ls "$ENV_OUTPUT_DIR"/action_*.txt 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓ Success: Saved $FRAME_COUNT frames and $ACTION_COUNT actions${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "  ${RED}✗ Failed to save frames${NC}"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

# Summary
echo "========================================="
echo "SUMMARY"
echo "========================================="
echo -e "${GREEN}Successful: $SUCCESS${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""
echo -e "Output directory: ${YELLOW}$OUTPUT_BASE_DIR${NC}"
echo ""

# Show sample of what was created
if [ "$SUCCESS" -gt 0 ]; then
    echo "Sample of created directories:"
    ls -la "$OUTPUT_BASE_DIR" | head -10
fi