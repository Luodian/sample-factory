#!/usr/bin/env python3
"""Test script to verify frame saving functionality"""

import os
import sys

# Test the frame saving functionality
print("Testing frame saving functionality...")

# Run with a smaller number of frames for testing
cmd = """python3 -m sf_examples.atari.enjoy_atari \
    --env=atari_alien \
    --experiment=edbeeching_atari_2B_atari_alien_1111 \
    --train_dir=/opt/tiger/sample-factory/checkpoints \
    --device=cpu \
    --save_frames \
    --frames_dir=test_frames \
    --video_frames=50 \
    --max_num_frames=50 \
    --no_render"""

print(f"Running command: {cmd}")
exit_code = os.system(cmd)

if exit_code == 0:
    # Check if frames were saved
    frames_dir = "/opt/tiger/sample-factory/checkpoints/edbeeching_atari_2B_atari_alien_1111/test_frames"
    if os.path.exists(frames_dir):
        frames = [f for f in os.listdir(frames_dir) if f.startswith("frame_")]
        actions = [f for f in os.listdir(frames_dir) if f.startswith("action_")]
        print(f"\n✓ Success! Saved {len(frames)} frames and {len(actions)} action files")
        print(f"Frames directory: {frames_dir}")
        
        # Show first few files
        if frames:
            print(f"\nFirst few frame files:")
            for f in sorted(frames)[:5]:
                print(f"  - {f}")
    else:
        print(f"\n✗ Frames directory not found: {frames_dir}")
else:
    print(f"\n✗ Command failed with exit code: {exit_code}")
    sys.exit(1)